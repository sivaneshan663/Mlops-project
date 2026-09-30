import os
import tempfile
import unittest
import io
from unittest.mock import patch

TEMP = tempfile.TemporaryDirectory()
os.environ['CONSTRUCTAI_DATA'] = TEMP.name
os.environ.pop('OLLAMA_MODEL', None)
os.environ.pop('EMBEDDING_MODEL', None)
from fastapi.testclient import TestClient
from app import app

class ProjectIsolationTests(unittest.TestCase):
    def test_api_loads_promoted_and_rolled_back_models_without_restart(self):
        import hashlib
        import joblib
        from pathlib import Path
        from forest import RandomForestRegressor
        from lifecycle import FEATURES, promote, rollback
        with tempfile.TemporaryDirectory() as folder, TestClient(app) as client:
            art=Path(folder)/'artifacts'; art.mkdir()
            pid=client.post('/api/projects',json={'name':'Registry integration'}).json()['id']
            inputs={'area_sqft':3000,'floors':3,'quality':2,'location_factor':1,'duration_months':12}
            with patch('app.ROOT',Path(folder)):
                for version, target in [('v1',100.),('v2',200.)]:
                    model=RandomForestRegressor(n_estimators=2).fit([[3000,3,2,1,12]]*4,[target]*4)
                    path=art/(version+'.joblib'); joblib.dump(model,path)
                    promote(art,{'version':version,'model_file':path.name,'features':FEATURES,'gate':{'passed':True},'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                    result=client.post(f'/api/projects/{pid}/estimate',json=inputs).json()
                    self.assertEqual(result['model_version'],version)
                    self.assertEqual(result['estimated_cost_inr'],target)
                rollback(art)
                result=client.post(f'/api/projects/{pid}/estimate',json=inputs).json()
                self.assertEqual(result['model_version'],'v1')
                self.assertEqual(result['estimated_cost_inr'],100.)

    def test_missing_or_invented_citations_fall_back_to_evidence(self):
        with TestClient(app) as client:
            pid=client.post('/api/projects',json={'name':'Citations','budget':100}).json()['id']
            for generated in ['The budget is 100.', 'The budget is 100 [S99].']:
                import json
                with patch.dict(os.environ,{'OLLAMA_MODEL':'test','EMBEDDING_MODEL':''}), patch('app.urllib.request.urlopen',return_value=io.BytesIO(json.dumps({'message':{'content':generated}}).encode())):
                    result=client.post(f'/api/projects/{pid}/chat',json={'message':'What is the budget?'}).json()
                self.assertIn('Citation check failed',result['mode'])
                self.assertIn('closest matching passages',result['content'])
                self.assertNotIn('[S99]',result['content'])

    def test_monitor_reports_missing_evidence_and_http_errors(self):
        with TestClient(app) as client:
            before=client.get('/api/monitoring').json()
            pid=client.post('/api/projects',json={'name':'Monitor test'}).json()['id']
            client.post(f'/api/projects/{pid}/chat',json={'message':'galactic zebras'})
            client.get('/api/projects/not-a-real-id/documents')
            after=client.get('/api/monitoring').json()
            self.assertEqual(after['missing_evidence_answers'],before['missing_evidence_answers']+1)
            self.assertGreater(after['errors'],before['errors'])
            self.assertEqual(client.get('/monitoring').status_code,200)

    def test_empty_model_response_preserves_evidence(self):
        with TestClient(app) as client:
            pid=client.post('/api/projects',json={'name':'Fallback test','budget':500}).json()['id']
            with patch.dict(os.environ,{'OLLAMA_MODEL':'test-model','EMBEDDING_MODEL':''}):
                with patch('app.urllib.request.urlopen',return_value=io.BytesIO(b'{"message":{"content":""}}')):
                    result=client.post(f'/api/projects/{pid}/chat',json={'message':'What is the budget?'}).json()
            self.assertIn('500',result['content'])
            self.assertIn('unavailable',result['mode'])

    def test_project_documents_and_history_are_isolated(self):
        with TestClient(app) as client:
            a = client.post('/api/projects', json={'name': 'Alpha', 'budget': 100}).json()['id']
            b = client.post('/api/projects', json={'name': 'Beta', 'budget': 200}).json()['id']
            for pid, content in [(a, b'Approved project budget is 100. Concrete supplier is AlphaOnly.'), (b, b'Approved project budget is 200. Concrete supplier is BetaOnly.')]:
                response = client.post(f'/api/projects/{pid}/documents', files={'file': ('report.txt', content, 'text/plain')})
                self.assertEqual(response.status_code, 200)
            answer = client.post(f'/api/projects/{a}/chat', json={'message': 'Who is the concrete supplier?'}).json()
            self.assertIn('AlphaOnly', answer['content'])
            self.assertNotIn('BetaOnly', str(answer))
            self.assertTrue(answer['sources'])
            self.assertEqual(client.get(f'/api/projects/{b}/messages').json(), [])
            self.assertEqual(len(client.get(f'/api/projects/{a}/messages').json()), 2)
            follow=client.post(f'/api/projects/{a}/chat',json={'message':'Explain that in more detail'}).json()
            self.assertIn('AlphaOnly',follow['content'])
            self.assertNotIn('BetaOnly',str(follow))
            greeting=client.post(f'/api/projects/{b}/chat',json={'message':'heallow'}).json()
            self.assertEqual(greeting['mode'],'Project assistant')
            self.assertIn('Beta',greeting['content'])
            before=len(client.get(f'/api/projects/{a}/documents').json())
            duplicate=client.post(f'/api/projects/{a}/documents',files={'file':('renamed.txt',b'Approved project budget is 100. Concrete supplier is AlphaOnly.','text/plain')}).json()
            self.assertTrue(duplicate['duplicate'])
            self.assertEqual(len(client.get(f'/api/projects/{a}/documents').json()),before)
            unknown = client.post(f'/api/projects/{a}/chat', json={'message': 'zebras astronomy galaxies'}).json()
            self.assertEqual(unknown['sources'], [])
            self.assertIn('could not find', unknown['content'])
            summary = client.post(f'/api/projects/{a}/chat', json={'message': 'Summarize my uploaded documents'}).json()
            self.assertIn('AlphaOnly', summary['content'])
            self.assertNotIn('BetaOnly', str(summary))
            client.post(f'/api/projects/{a}/documents', files={'file': ('res (2).txt', b'Example resume. Skills: Python and project management.')})
            informal = client.post(f'/api/projects/{a}/chat', json={'message': 'say about this res'}).json()
            self.assertIn('Python', informal['content'])
            self.assertEqual(informal['sources'][0]['document'], 'res (2).txt')
            self.assertEqual(client.get('/api/projects/missing/documents').status_code, 404)
            self.assertEqual(client.post(f'/api/projects/{a}/documents', files={'file': ('x.exe', b'bad')}).status_code, 400)
            self.assertEqual(client.post(f'/api/projects/{a}/documents', files={'file': ('empty.txt', b' ')}).status_code, 400)
            self.assertEqual(client.post(f'/api/projects/{a}/documents', files={'file': ('bad.pdf', b'not a pdf')}).status_code, 400)
            csv = client.post(f'/api/projects/{a}/documents', files={'file': ('cost.csv', b'item,cost\ncement,500')})
            self.assertEqual(csv.status_code, 200)
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/api/system').status_code, 200)
            self.assertEqual(client.post(f'/api/projects/{a}/estimate',json={'area_sqft':-1,'floors':3,'quality':2,'location_factor':1,'duration_months':12}).status_code,422)
            status=client.get('/api/system').json()
            if status['cost']:
                result=client.post(f'/api/projects/{a}/estimate',json={'area_sqft':3000,'floors':3,'quality':2,'location_factor':1,'duration_months':12})
                self.assertEqual(result.status_code,200)
                self.assertGreater(result.json()['estimated_cost_inr'],0)
                self.assertEqual(result.json()['data_kind'],'synthetic')
                explain=client.post(f'/api/projects/{a}/chat',json={'message':'Explain the latest synthetic cost estimate'}).json()
                self.assertTrue(any(s['document']=='Latest synthetic cost estimate' for s in explain['sources']))

if __name__ == '__main__':
    unittest.main()
