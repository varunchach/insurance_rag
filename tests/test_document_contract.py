"""Source isolation and generalization tests without customer PDFs or annotations."""
import hashlib, json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from document_contract import (bind_runtime_source, validate_source, verify_artifact,
    verify_chunk_binding, seal_manifest, inspect_pdf, policy, sha256)
from preservation import repair_document, validated_options
from docling_core.types.doc import DoclingDocument, Size
import fitz

class UploadContracts(unittest.TestCase):
    def test_runtime_parameters_override_and_remove_sample_settings(self):
        base={"bucket":"approved","source_key":"old.pdf","source_sha256":"a"*64,
              "parsed_artifact_uri":"s3://old/parse","native_artifact_uri":"s3://old/native",
              "kit_prefix":"private-gold/","kit_hashes":{"gold":"hash"}}
        result=bind_runtime_source(base,"uploads/new.pdf","b"*64,"client-document")
        self.assertEqual(result["source_key"],"uploads/new.pdf")
        self.assertEqual(result["source_sha256"],"b"*64)
        self.assertFalse(any(k in result for k in ("parsed_artifact_uri","native_artifact_uri","kit_prefix","kit_hashes")))

    def test_missing_runtime_parameters_never_use_defaults(self):
        for values in [(None,"b"*64,"doc"),("new.pdf",None,"doc"),("new.pdf","b"*64,None)]:
            with self.subTest(values=values),self.assertRaises(ValueError):
                bind_runtime_source({"source_key":"old.pdf","source_sha256":"a"*64},*values)

    def test_invalid_object_keys_and_hashes_rejected(self):
        for key in ("../source.pdf","/source.pdf","a/../b","https://elsewhere/file","a\\b","\nfile"):
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_source({"source_key":key,"source_sha256":"a"*64})
        for checksum in ("", "a"*63, "A"*64, "not-a-hash"):
            with self.subTest(checksum=checksum),self.assertRaises(ValueError):
                validate_source({"source_key":"new.pdf","source_sha256":checksum})

    def fixture(self,root,data=b"%PDF-synthetic-A",document_id="document-A"):
        source=root/"source.pdf";source.write_bytes(data)
        config=bind_runtime_source({},"uploads/input.pdf",sha256(source),document_id)
        (root/"document.json").write_text('{"text":"original parse"}')
        manifest={"source_sha256":config["source_sha256"],"document_id":document_id,
                  "canonical_sha256":sha256(root/"document.json")}
        seal_manifest(root,manifest)
        return config,manifest

    def test_cross_document_artifact_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_=self.fixture(root)
            wrong={**cfg,"source_sha256":"b"*64}
            with self.assertRaisesRegex(ValueError,"source identity"):verify_artifact(wrong,root)
            with self.assertRaisesRegex(ValueError,"document_id"):verify_artifact({**cfg,"document_id":"another-tenant-document"},root)

    def test_source_and_canonical_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_=self.fixture(root)
            (root/"source.pdf").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError,"PDF checksum"):verify_artifact(cfg,root)
            cfg,_=self.fixture(root)
            (root/"document.json").write_text('{"text":"swapped"}')
            with self.assertRaisesRegex(ValueError,"Canonical parse"):verify_artifact(cfg,root,canonical=True)

    def test_chunk_parse_version_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parsed=root/"parse";parsed.mkdir()
            cfg,parent=self.fixture(parsed)
            chunks=root/"chunks";chunks.mkdir()
            manifest={"source_sha256":cfg["source_sha256"],"document_id":cfg["document_id"],
                      "parsed_canonical_sha256":"c"*64}
            seal_manifest(chunks,manifest)
            with self.assertRaisesRegex(ValueError,"parse version"):verify_chunk_binding(cfg,parsed,chunks)
            manifest["parsed_canonical_sha256"]=parent["canonical_sha256"]
            seal_manifest(chunks,manifest)
            verify_chunk_binding(cfg,parsed,chunks)

    def test_payload_checksum_covers_materialized_elements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,m=self.fixture(root)
            (root/"elements.json").write_text("[]");seal_manifest(root,m)
            (root/"elements.json").write_text('[{"ref":"foreign"}]')
            with self.assertRaisesRegex(ValueError,"payload checksum"):verify_artifact(cfg,root)

    def test_unknown_or_invalid_thresholds_rejected(self):
        for options in ({"page_number":7},{"table_match_iou":2},{"expansion_hops":1.5},{"note_gap":-1}):
            with self.subTest(options=options),self.assertRaises(ValueError):validated_options(options)
        with self.assertRaises(ValueError):policy({"input_policy":{"max_pages":0}})

    def make_pdf(self,path,kind):
        pdf=fitz.open();page=pdf.new_page(width=400,height=300)
        if kind in ("native","mixed"):
            page.insert_text((25,35),"Independent client policy with different names and conditions.")
        if kind in ("scan","mixed"):
            src=fitz.open();render=src.new_page(width=300,height=150)
            render.insert_text((10,40),"Scanned condition: minimum 37 and maximum 82.")
            pix=render.get_pixmap()
            page.insert_image(fitz.Rect(20,80,380,260),pixmap=pix)
            src.close()
        pdf.save(path);pdf.close()

    def test_native_pdf_is_parseable_without_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"fresh.pdf";self.make_pdf(path,"native")
            result=inspect_pdf(path,{})
            self.assertTrue(result["parse_allowed"])
            self.assertEqual(result["page_count"],1)

    def test_scan_and_mixed_document_are_explicitly_routed_to_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind,reason in (("scan","ocr_required"),("mixed","visual_content_review_required")):
                path=Path(tmp)/(kind+".pdf");self.make_pdf(path,kind)
                result=inspect_pdf(path,{})
                self.assertFalse(result["parse_allowed"])
                self.assertIn(reason,[x["reason"] for x in result["review_reasons"]])

    def test_empty_and_unsupported_documents_do_not_succeed_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"blank.pdf";self.make_pdf(path,"blank")
            self.assertFalse(inspect_pdf(path,{})["parse_allowed"])
            path.write_bytes(b"not a PDF")
            with self.assertRaisesRegex(ValueError,"PDF documents"):inspect_pdf(path,{})

    def test_size_limit_is_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"fresh.pdf";self.make_pdf(path,"native")
            with self.assertRaisesRegex(ValueError,"byte limit"):
                inspect_pdf(path,{"input_policy":{"max_source_bytes":10}})

    def test_table_recovery_survives_renaming_repagination_and_layout_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index,(name,amount,blanks,offset) in enumerate([
                ("Plan Kestrel","37",0,0),("Service Maple","842",2,35),("Benefit Quartz","126",1,15)]):
                path=Path(tmp)/(str(index)+".pdf")
                pdf=fitz.open()
                for _ in range(blanks):pdf.new_page(width=500,height=400)
                page=pdf.new_page(width=500,height=400)
                x=30+offset;y=100
                for xx in (x,x+120,x+240):page.draw_line((xx,y),(xx,y+80))
                for yy in (y,y+40,y+80):page.draw_line((x,yy),(x+240,yy))
                for px,py,text in [(x+5,y+20,"Contract"),(x+125,y+20,"Amount"),
                                   (x+5,y+60,name),(x+125,y+60,amount)]:
                    page.insert_text((px,py),text)
                pdf.save(path);pdf.close()
                doc=DoclingDocument(name=name)
                for n in range(1,blanks+2):doc.add_page(page_no=n,size=Size(width=500,height=400))
                audit=repair_document(doc,path)
                self.assertTrue(any(t["shape"]==[2,2] for t in audit["table_repairs"]))
                self.assertTrue(all(t["page"]==blanks+1 for t in audit["table_repairs"]))
                text=" ".join(c.text for t in doc.tables for c in t.data.table_cells)
                self.assertIn(name,text);self.assertIn(amount,text)
                self.assertFalse(audit["gold_annotations_consumed"])

if __name__=="__main__":unittest.main(verbosity=2)
