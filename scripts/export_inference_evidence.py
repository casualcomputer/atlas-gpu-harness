"""Publish complete local Atlas exports and a compact inference summary.

Usage: python scripts/export_inference_evidence.py RUN_ID [RUN_ID ...]
Large weights, runtime binaries, credentials, and server logs are not copied.
"""
import argparse
import json, hashlib, shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_ids", nargs="+")
args = parser.parse_args()
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'evidence'/'inference'
out.mkdir(parents=True, exist_ok=True)
rows=[]
for run_id in args.run_ids:
    source=root/'artifacts'/'inference'/run_id/'report.json'
    report=json.loads(source.read_text())
    destination=out/f'{run_id}.json'
    shutil.copyfile(source,destination)
    run=report['run']
    trials=[]
    for experiment in report['experiments']:
        result=experiment['result']
        trials.append({'id':experiment['_id'], 'candidate':experiment['candidate'], 'status':experiment['status'], 'attempt':experiment['attempt'], 'stage':result.get('stage'), 'recipe':result.get('recipe'), 'correct':result.get('correct'), 'quality':result.get('quality'), 'device_peak_mib':result.get('memory',{}).get('device_peak_mib'), 'process_rss_peak_mib':result.get('memory',{}).get('process_rss_peak_mib'), 'median_decode_tokens_per_second':result.get('median_decode_tokens_per_second'), 'median_request_latency_seconds':result.get('median_request_latency_seconds'), 'load_seconds':result.get('load_seconds'), 'error':result.get('error')})
    rows.append({'run_id':run_id,'status':run['status'],'model':run['context']['model'],'gpu':run['context']['hardware']['gpu'],'gpu_budget_mib':run['context']['fixed']['gpu_budget_mib'],'context_hash':run['context_hash'],'recommendation':run.get('recommendation'),'proposals':run.get('proposals'), 'proposal_errors':run.get('proposal_errors',{}),'trials':trials,'report_path':str(destination.relative_to(root)),'report_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
(root/'evidence'/'inference-summary.json').write_text(json.dumps({'scope':'Short exact-answer tests on one desktop, not general quality or production throughput; full records exported from MongoDB Atlas.', 'runs':rows},indent=2,default=str)+'\n')
print(json.dumps([{'run_id':r['run_id'],'status':r['status'],'trials':len(r['trials']),'recipe':(r.get('recommendation') or {}).get('recipe')} for r in rows]))
