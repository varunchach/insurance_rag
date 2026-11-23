"""Document-independent preservation contracts. No customer PDF or gold fixtures."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from preservation import (lexical, numbered_level, ruled_data, build_context_graph,
                          expand_refs, assign_headings, region, split_text, repair_document)
from docling_core.types.doc import DoclingDocument, DocItemLabel

class Tokenizer:
    def encode(self,text,add_special_tokens=True):
        return list(text)+([0,1] if add_special_tokens else [])

def element(ref,label,text,page=1,top=20,headings=()):
    return {"ref":ref,"label":label,"text":text,"content_layer":"body",
            "heading_candidates":list(headings),"parent_ref":None,
            "provenance":[{"page_no":page,"bbox":{"l":0,"r":100,"t":top,"b":top+10,"coord_origin":"TOPLEFT"}}]}

def table(ref,page,box,cols=3,caption=False):
    return {"ref":ref,"shape":[2,cols],"regions":[{"page":page,"bbox":box}],
            "raw_cells":[{"start_row_offset_idx":0,"start_col_offset_idx":0,
                          "end_col_offset_idx":cols if caption else 1,"text":"Fresh title" if caption else "21"}]}

class PreservationTests(unittest.TestCase):
    def test_numbering_and_consecutive_headings(self):
        doc=DoclingDocument(name="synthetic")
        a=doc.add_heading(text="B. Parent")
        b=doc.add_heading(text="8. Variant Zeta")
        c=doc.add_heading(text="Applicable Criteria:")
        text=doc.add_text(label=DocItemLabel.TEXT,text="Some condition")
        d=doc.add_heading(text="9. Variant Eta")
        lineage=assign_headings(doc)
        self.assertEqual([h["ref"] for h in lineage[text.self_ref]],[a.self_ref,b.self_ref,c.self_ref])
        self.assertEqual([h["ref"] for h in lineage[d.self_ref]],[a.self_ref,d.self_ref])
        self.assertEqual(numbered_level("3.2.7. Conditions"),4)
        self.assertIsNone(numbered_level("Amount 31.5"))

    def test_numeric_normalization_retains_meaning(self):
        self.assertEqual(lexical("12th Product98"),lexical("12 th Product 98"))
        for a,b in [("1.25","125"),("-20","20"),("<=7",">7"),("not covered","covered")]:
            self.assertNotEqual(lexical(a),lexical(b))

    def test_ruled_merged_cell_and_literal_values(self):
        rows=[SimpleNamespace(cells=[(0,0,90,10),None,None]),
              SimpleNamespace(cells=[(0,10,30,20),(30,10,60,20),(60,10,90,20)])]
        candidate=SimpleNamespace(rows=rows,extract=lambda:[["Shared heading",None,None],["NA","Nil",""]])
        data=ruled_data(candidate)
        self.assertEqual((data.num_rows,data.num_cols),(2,3))
        merged=[c for c in data.table_cells if c.text=="Shared heading"][0]
        self.assertEqual(merged.col_span,3)
        self.assertEqual({c.text for c in data.table_cells},{"Shared heading","NA","Nil",""})

    def test_continuation_requires_edges_and_no_new_heading(self):
        a=table("a",7,[0,750,100,990]);b=table("b",8,[0,10,100,200])
        graph=build_context_graph([], [a,b], {7:1000,8:1000})
        self.assertEqual(sum(e["kind"]=="cross_page_table" for e in graph["edges"]),2)
        heading=element("h","section_header","Independent scope",8,0)
        graph=build_context_graph([heading],[a,b],{7:1000,8:1000})
        self.assertFalse(any(e["kind"]=="cross_page_table" for e in graph["edges"]))
        b=table("b",8,[0,10,100,200],caption=True)
        self.assertFalse(build_context_graph([],[a,b],{7:1000,8:1000})["edges"])

    def test_dated_tables_same_page_remain_separate(self):
        a=table("a",3,[0,100,100,200]);b=table("b",3,[0,300,100,400])
        elems=[element("a","table","Effective 2031-02-01: age 40 => 7",3,100),
               element("b","table","Effective 2032-01-02: age 40 => 9",3,300)]
        graph=build_context_graph(elems,[a,b],{3:1000})
        self.assertFalse(any(e["kind"]=="cross_page_table" for e in graph["edges"]))

    def test_footnote_attaches_to_preceding_table(self):
        elems=[element("a","table","Grid A"),element("n","text","* Subject to test",top=35),
               element("b","table","Grid B",top=60)]
        graph=build_context_graph(elems,[],{1:1000})
        pairs={(e["source"],e["target"]) for e in graph["edges"]}
        self.assertIn(("a","n"),pairs)
        self.assertNotIn(("b","n"),pairs)

    def test_expansion_budget_and_hop_limit(self):
        graph={"edges":[{"source":"a","target":"b"},{"source":"b","target":"c"},
                        {"source":"c","target":"d"},{"source":"b","target":"a"}]}
        refs={r:{"text":r*8} for r in "abcd"}
        kept,text,deferred=expand_refs(["a"],graph,refs,Tokenizer(),22,2)
        self.assertEqual(kept,["a","b"])
        self.assertEqual(deferred,["c"])
        self.assertLessEqual(len(Tokenizer().encode(text)),22)

    def test_long_unbroken_text_respects_budget(self):
        source="abcdefghijklmno"*10+" tail"
        pieces=split_text(source,Tokenizer(),16)
        self.assertEqual("".join(pieces),source)
        self.assertTrue(all(len(Tokenizer().encode(p))<=16 for p in pieces))

    def test_unnumbered_note_uses_page_geometry(self):
        elems=[element("a","table","Grid A",top=10),
               element("n","text","An additional restriction applies",top=23),
               element("b","table","Grid B",top=70)]
        graph=build_context_graph(elems,[],{1:1000})
        pairs={(e["source"],e["target"]) for e in graph["edges"]}
        self.assertIn(("a","n"),pairs)
        self.assertNotIn(("b","n"),pairs)

    def test_appendix_resets_numbered_scope(self):
        self.assertEqual(numbered_level("Appendix 8: Independent tests"),1)
        self.assertEqual(numbered_level("Chapter IV Scope"),1)

    def test_native_recovery_is_coordinate_backed_and_markers_not_duplicated(self):
        import fitz,tempfile
        from docling_core.types.doc import Size, ProvenanceItem, BoundingBox, CoordOrigin
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"synthetic.pdf"
            pdf=fitz.open();page=pdf.new_page(width=400,height=300)
            page.insert_text((30,50),"Employer contributes 17% every quarter.")
            pdf.save(path);pdf.close()
            doc=DoclingDocument(name="synthetic")
            doc.add_page(page_no=1,size=Size(width=400,height=300))
            prov=ProvenanceItem(page_no=1,bbox=BoundingBox(l=30,t=35,r=250,b=55,coord_origin=CoordOrigin.TOPLEFT),charspan=(0,10))
            doc.add_text(label=DocItemLabel.TEXT,text="Employer contributes",prov=prov)
            entry=doc.add_list_item(text="A separate numbered condition",enumerated=True,marker="7.")
            audit=repair_document(doc,path)
            self.assertEqual(entry.text,"A separate numbered condition")
            self.assertEqual(entry.marker,"7.")
            self.assertTrue(any(x["ref"]==entry.self_ref for x in audit["list_markers"]))
            self.assertTrue(any("17" in r["text"] for r in audit["native_supplements"]))
            self.assertTrue(all(r["page"]==1 and len(r["bbox"])==4 for r in audit["native_supplements"]))
            self.assertFalse(audit["gold_annotations_consumed"])

    def test_empty_group_has_no_region(self):
        doc=DoclingDocument(name="empty")
        self.assertIsNone(region(doc.body,doc))

if __name__=="__main__":
    unittest.main(verbosity=2)
