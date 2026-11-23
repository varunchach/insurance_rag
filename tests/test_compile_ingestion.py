import json,subprocess,sys,tempfile,unittest
from pathlib import Path

class CompiledUploadContract(unittest.TestCase):
    def test_one_compiled_pipeline_has_required_inputs_and_no_sample_reuse(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config={'runtime_image':'example.invalid/approved/runtime@sha256:'+'a'*64,
                'code_configmap':'approved-code','source_key':'trap/never-use.pdf','source_sha256':'b'*64,
                'parsed_artifact_uri':'s3://trap/old-parse','native_artifact_uri':'s3://trap/old-native',
                'kit_prefix':'trap/gold'}
            (root/'config.json').write_text(json.dumps(config))
            (root/'bootstrap.py').write_text('print(123)')
            process=subprocess.run([sys.executable,str(repo/'pipelines/compile_ingestion.py'),
                str(root/'config.json'),str(root/'bootstrap.py'),str(root/'out')],capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            spec=json.loads((root/'out/pipeline.json').read_text())
            parameters=spec['root']['inputDefinitions']['parameters']
            self.assertEqual(set(parameters),{'source_key','source_sha256','document_id'})
            self.assertTrue(all('defaultValue' not in p and not p.get('isOptional',False) for p in parameters.values()))
            serialized=json.dumps(spec)
            for trap in [config['source_key'],config['source_sha256'],config['parsed_artifact_uri'],
                         config['native_artifact_uri'],config['kit_prefix']]:
                self.assertNotIn(trap,serialized)
            self.assertFalse(any('importer' in executor for executor in spec['deploymentSpec']['executors']))
            self.assertNotIn('qualification-score',serialized)
            self.assertIn('condition-1',spec['root']['dag']['tasks'])

if __name__=='__main__':unittest.main(verbosity=2)
