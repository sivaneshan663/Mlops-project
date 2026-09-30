"""Cached local neural embeddings; never shares documents with remote APIs."""
import hashlib
import os
import json
import math
import sqlite3
import urllib.request

def semantic_scores(texts, question, cache_path, model='all-minilm'):
    conn = sqlite3.connect(cache_path)
    try:
        conn.execute('CREATE TABLE IF NOT EXISTS embeddings(key TEXT PRIMARY KEY, vector TEXT)')
        vectors = []
        for text in texts + [question]:
            key = hashlib.sha256((model + '\n' + text).encode()).hexdigest()
            row = conn.execute('SELECT vector FROM embeddings WHERE key=?', (key,)).fetchone()
            if row:
                vector = json.loads(row[0])
            else:
                body = json.dumps({'model':model, 'input':text, 'keep_alive':'5m'}).encode()
                req = urllib.request.Request(os.environ.get('OLLAMA_URL','http://127.0.0.1:11434')+'/api/embed', data=body, headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    vector = json.load(response)['embeddings'][0]
                conn.execute('INSERT OR REPLACE INTO embeddings VALUES(?,?)', (key, json.dumps(vector)))
            vectors.append(vector)
        conn.commit()
        q = vectors.pop()
        norm = lambda v: math.sqrt(sum(x*x for x in v))
        return [sum(a*b for a,b in zip(v,q))/(norm(v)*norm(q) or 1) for v in vectors]
    finally:
        conn.close()
