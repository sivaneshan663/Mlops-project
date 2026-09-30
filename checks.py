"""Same entry point for local checks and hosted CI, without model downloads."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from lifecycle import atomic_json, now

ROOT = Path(__file__).resolve().parent

def main():
    subprocess.run([sys.executable,'-m','unittest','test_app','test_forest','test_lifecycle','-v'],cwd=ROOT,check=True)
    with tempfile.TemporaryDirectory() as folder:
        env = {**os.environ,'CONSTRUCTAI_DATA':folder,'OLLAMA_MODEL':'','EMBEDDING_MODEL':''}
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        process = subprocess.Popen([sys.executable,'-m','uvicorn','app:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            deadline=time.monotonic()+40
            while True:
                if process.poll() is not None: raise RuntimeError('Deployment exited before readiness')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1) as response:
                        assert json.load(response)['status']=='ok'
                    break
                except OSError:
                    if time.monotonic()>deadline: raise
                    time.sleep(.2)
            for route in ['/', '/monitoring', '/api/monitoring', '/api/system']:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}{route}',timeout=5) as response:
                    assert response.status==200
        finally:
            process.terminate(); process.wait(timeout=15)
    report={'at':now(),'unit_tests':'passed','local_process_deployment':'passed: actual uvicorn startup, health, chat page, monitoring page/API, system API','python':sys.version,'execution_environment':'GitHub Actions' if os.environ.get('GITHUB_ACTIONS') else 'local','hosted_ci':'executed' if os.environ.get('GITHUB_ACTIONS') else 'not run: local execution only','docker':'not tested by this check script'}
    atomic_json(ROOT/'artifacts'/'checks_report.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
