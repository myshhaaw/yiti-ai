from __future__ import annotations
import hashlib, json, os, re, shutil, sqlite3
from pathlib import Path
import numpy as np
from yiti_ai.semantic_retrieval import eligible_records, records_fingerprint, MODEL_ID, MODEL_DIMENSION

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ['YITI_SOURCE_INDEX']).resolve()
SOURCE_VECTORS=Path(os.environ['YITI_SOURCE_VECTORS']).resolve()
OUT=ROOT/'deploy_artifacts'
DB=OUT/'search.sqlite3'
VECTORS=OUT/'policy-vectors.npz'
DOC_COLUMNS=('document_id','title','file_path','department','source_url','policy_date','verification_status','processing_status','privacy_flags_json','course','user_supplied_date','policy_date_status','date_precision','source_url_status','metadata_provenance','file_paths_json','source_file_count')
CHUNK_COLUMNS=('content','title','department','heading','document_id','file_path','source_url','page','policy_date','course')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=sqlite3.connect(f'file:{SOURCE.as_posix()}?mode=ro',uri=True)
    src.row_factory=sqlite3.Row
    docs=src.execute("SELECT * FROM documents WHERE privacy_flags_json='[]' AND processing_status LIKE 'indexed%' ORDER BY document_id").fetchall()
    ids=[row['document_id'] for row in docs]
    if len(ids)!=26: raise SystemExit(f'Expected 26 eligible docs; found {len(ids)}')
    marks=','.join('?' for _ in ids)
    chunks=src.execute(f"SELECT rowid AS chunk_id,{','.join(CHUNK_COLUMNS)} FROM chunks WHERE document_id IN ({marks}) ORDER BY rowid",ids).fetchall()
    if not chunks: raise SystemExit('No eligible chunks')
    if len(chunks)!=1600: raise SystemExit(f'Expected 1600 eligible chunks; found {len(chunks)}')
    if DB.exists(): DB.unlink()
    dst=sqlite3.connect(DB)
    dst.execute('CREATE TABLE documents (document_id TEXT PRIMARY KEY,title TEXT,file_path TEXT,department TEXT,source_url TEXT,policy_date TEXT,verification_status TEXT,processing_status TEXT,privacy_flags_json TEXT,course TEXT,user_supplied_date TEXT,policy_date_status TEXT,date_precision TEXT,source_url_status TEXT,metadata_provenance TEXT,file_paths_json TEXT,source_file_count INTEGER)')
    dst.execute("CREATE VIRTUAL TABLE chunks USING fts5(content,title,department UNINDEXED,heading UNINDEXED,document_id UNINDEXED,file_path UNINDEXED,source_url UNINDEXED,page UNINDEXED,policy_date UNINDEXED,course UNINDEXED,tokenize='unicode61 remove_diacritics 2')")
    dst.execute('CREATE TABLE extraction_pages (document_id TEXT NOT NULL,page INTEGER NOT NULL,method TEXT NOT NULL,ocr_confidence REAL,suspicious_output INTEGER NOT NULL,PRIMARY KEY(document_id,page))')
    for d in docs: dst.execute(f"INSERT INTO documents ({','.join(DOC_COLUMNS)}) VALUES ({','.join('?' for _ in DOC_COLUMNS)})",[d[c] for c in DOC_COLUMNS])
    for row in chunks: dst.execute('INSERT INTO chunks(rowid,'+','.join(CHUNK_COLUMNS)+') VALUES ('+','.join('?' for _ in range(len(CHUNK_COLUMNS)+1))+')',[row['chunk_id']]+[row[c] for c in CHUNK_COLUMNS])
    pages=src.execute(f"SELECT document_id,page,method,ocr_confidence,suspicious_output FROM extraction_pages WHERE document_id IN ({marks})",ids).fetchall()
    dst.executemany('INSERT INTO extraction_pages VALUES (?,?,?,?,?)',[tuple(p) for p in pages])
    dst.commit(); dst.execute('PRAGMA integrity_check').fetchone(); integrity=dst.execute('PRAGMA integrity_check').fetchone()[0]
    if integrity!='ok': raise SystemExit('SQLite integrity check failed')
    dst.close(); src.close()
    records=eligible_records(DB)
    with np.load(SOURCE_VECTORS,allow_pickle=False) as z:
        vectors=np.asarray(z['vectors'],dtype=np.float32)
        meta=json.loads(str(z['metadata'].item()))
    if vectors.shape!=(len(records),MODEL_DIMENSION) or len(meta.get('records',[]))!=len(records): raise SystemExit('Vector rows do not match public chunks')
    if meta.get('model_id')!=MODEL_ID: raise SystemExit('Embedding model mismatch')
    if records_fingerprint(records)!=meta.get('source_fingerprint'): raise SystemExit('Chunk ids/content do not match vector metadata')
    meta['source_index']=DB.name
    temp=VECTORS.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez_compressed(f,vectors=vectors,metadata=np.asarray(json.dumps(meta,ensure_ascii=False)))
    temp.replace(VECTORS)
    secret=re.compile(r'(AIza[0-9A-Za-z_-]{30,}|(?:GEMINI|OPENAI|AZURE)[A-Z0-9_]*KEY\s*[:=]\s*[^\s#]{8,}|gh[pousr]_[A-Za-z0-9]{30,})',re.I)
    scan='\n'.join(str(r.get('content','')) for r in records)
    print(json.dumps({'sqlite':str(DB),'sqlite_bytes':DB.stat().st_size,'vectors':str(VECTORS),'vector_bytes':VECTORS.stat().st_size,'documents':len(ids),'chunks':len(records),'restricted_docs_in_output':dst_count(DB),'sqlite_integrity':integrity,'vector_rows':int(vectors.shape[0]),'vector_dim':int(vectors.shape[1]),'model_id':meta['model_id'],'vector_fingerprint_match':True,'secret_pattern_hits_in_public_passages':len(secret.findall(scan))},indent=2))

def dst_count(path):
    c=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
    n=c.execute("SELECT count(*) FROM documents WHERE privacy_flags_json!='[]' OR processing_status NOT LIKE 'indexed%'").fetchone()[0]
    c.close(); return n

if __name__=='__main__': main()