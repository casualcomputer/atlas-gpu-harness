"""One native-Windows 32K request per engine; descriptive comparison, not promotion."""
import gzip
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filelock import FileLock
from atlas_gpu_harness.store import AtlasStore, now
from atlas_gpu_harness.strata_experiment import Resources, digest, evaluate, export, save, score

STRATA = ROOT / 'artifacts/Strata'
sys.path[:0] = [str(STRATA), str(STRATA / 'tools')]
from serve.server import StrataEngine, engine_args, child_env
from strata_tokenizer import Tokenizer


def post(base, route, payload, timeout=900):
    req = urllib.request.Request(base + route, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def llama_trial(exe, model, prompt, tok, directory):
    base = 'http://127.0.0.1:8977'
    command = [str(exe), '-m', model, '-c', '33792', '--fit-ctx', '33792',
               '--parallel', '1', '--host', '127.0.0.1', '--port', '8977',
               '-t', '13', '-tb', '13', '--reasoning', 'off', '--reasoning-budget', '0',
               '--flash-attn', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0',
               '--load-mode', 'mmap', '--lazy-mode', 'on', '--no-warmup']
    result = {'engine': 'llama.cpp', 'policy': 'engine_autofit', 'command': command,
              'prompt_tokens': len(prompt['input_ids']), 'correct': False,
              'speculative_decoding': False, 'engine_success': False}
    proc = sampler = None
    with (directory / 'llama.log').open('w', encoding='utf-8') as log:
        try:
            start = time.monotonic()
            proc = subprocess.Popen(command, stdout=log, stderr=log)
            deadline = start + 360
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'llama.cpp exited during loading ({proc.returncode}); see llama.log')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError('llama.cpp load exceeded 360 seconds')
            result['load_seconds'] = time.monotonic() - start
            with post(base, '/detokenize', {'tokens': prompt['input_ids']}) as response:
                decoded = json.load(response)['content']
            result['tokenizer_match'] = decoded == tok.decode(prompt['input_ids'])
            result['tokenizer_check'] = 'Identical saved token IDs decode to identical text in both engines; re-encoding concatenated token sections can merge boundaries'
            if not result['tokenizer_match']:
                save(directory/'tokenizer-mismatch.json', {'llama': decoded, 'strata': tok.decode(prompt['input_ids'])})
                raise ValueError('Token IDs decode differently; comparison aborted')
            warm = tok.encode('<|im_start|>user\nSay ready.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
            with post(base, '/completion', {'prompt': warm, 'n_predict': 16, 'temperature': 0, 'cache_prompt': False}) as response:
                result['warmup'] = json.load(response).get('timings')
            sampler = Resources(proc.pid)
            sampler.start()
            start = time.monotonic()
            first = None
            chunks = []
            final = None
            with post(base, '/completion', {'prompt': prompt['input_ids'], 'n_predict': 768,
                      'temperature': 0, 'seed': prompt['seed'], 'cache_prompt': False, 'stream': True}) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    event = json.loads(line[6:])
                    if event.get('error'):
                        raise RuntimeError(str(event['error']))
                    if event.get('content'):
                        if first is None:
                            first = time.monotonic() - start
                        chunks.append(event['content'])
                    if event.get('stop'):
                        final = event
            if final is None:
                raise RuntimeError('Missing final timing event')
            answer = ''.join(chunks)
            timing = final['timings']
            result.update(engine_success=True, request_seconds=time.monotonic()-start,
                          first_token_seconds=first, answer_text=answer, telemetry=final,
                          decode_tps=timing['predicted_per_second'], prefill_tps=timing['prompt_per_second'])
            result['quality'] = score(answer, prompt['expected'])
            # tokens_cached can describe the final populated slot; prompt_n is the evaluated-input counter.
            result['input_verified'] = timing['prompt_n'] == 32768
            result['correct'] = result['quality']['passed'] and result['input_verified']
        except Exception as exc:
            result['error'] = {'type': type(exc).__name__, 'message': str(exc)[:1500]}
        finally:
            if sampler:
                result['resources'] = sampler.stop()
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
    return result


def main():
    cfg = json.loads((STRATA / 'strata-iq2_xs.json').read_text(encoding='utf-8-sig'))
    tpath = Path(cfg['tokenizer'])
    vocab = json.loads((tpath / 'vocab.json').read_text(encoding='utf-8'))
    tokens = [None] * len(vocab)
    for word, idx in vocab.items():
        tokens[idx] = word
    tok = Tokenizer(tokens, (tpath/'merges.txt').read_text(encoding='utf-8').split('\n'), json.loads((tpath/'token_type.json').read_text(encoding='utf-8')))
    artifact = ROOT/'evidence/strata/e5359ac4eb6d42358187f0530d66b1d9/prompt-tuning-32768-0.json.gz'
    prompt = json.loads(gzip.decompress(artifact.read_bytes()))
    exe = ROOT/'artifacts/engines/llama-b11429-windows/llama-server.exe'
    model = cfg['args'][cfg['args'].index('--native')+1]
    fixed = {'protocol': 'quick-engine-comparison-32k-v1', 'platform': sys.platform,
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'prompt_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(), 'prompt_artifact': str(artifact),
             'prompt_tokens': 32768, 'max_new': 768, 'temperature': 0, 'seed': prompt['seed'],
             'reasoning': False, 'strata_config': cfg, 'llama_version': 'b11429',
             'llama_exe_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
             'limitations': ['One request per engine; descriptive only', 'Strata uses shipped MTP4; llama.cpp has no draft head',
                             'Same quantized weights and token IDs; backend kernels and cache formats differ',
                             'Fixed order: llama.cpp then Strata; warmup per engine, no prompt reuse']}
    with FileLock(str(ROOT/'artifacts/gpu.lock'), timeout=0):
        store = AtlasStore()
        store.client.admin.command('ping')
        run_id = store.create_run(fixed, digest(fixed), 2,
                                  objective='Quick Qwen3.8-Flash-Next llama.cpp versus Strata 32K comparison',
                                  policy_version=fixed['protocol'])
        directory = ROOT/'evidence/engine-comparison'/run_id
        directory.mkdir(parents=True)
        latest = directory.parent/'latest.json'
        print('ATLAS_RUN_ID='+run_id, flush=True)
        for arm in ['llama_cpp', 'strata_default']:
            history = [e['_id'] for e in store.experiments_for(run_id)]
            store.decide(run_id, history, arm, 'Measure the same saved 32K task on this engine', 'User-requested quick comparison; no tuning or promotion', fixed['protocol'])
            owner = store.start(run_id, arm, digest(fixed), arm)
            print('Starting '+arm, flush=True)
            if arm == 'llama_cpp':
                result = llama_trial(exe, model, prompt, tok, directory)
            else:
                result = evaluate(cfg, StrataEngine, engine_args, child_env, tok, prompt, 'default', directory/'strata.log')
                result.update(engine='Strata', speculative_decoding='MTP4')
            result['prompt_artifact'] = str(artifact)
            save(directory/(arm+'.json'), result)
            store.finish(run_id, arm, owner, result)
            export(store, run_id, latest)
            print(json.dumps({k: result.get(k) for k in ['engine','engine_success','decode_tps','prefill_tps','request_seconds','quality','error']}), flush=True)
            if arm == 'llama_cpp' and not result.get('engine_success'):
                print('Baseline failed: skipping unmatched Strata rerun.', flush=True)
                break
        store.runs.update_one({'_id': run_id}, {'$set': {'status': 'completed', 'finished_at': now(), 'recommendation': None, 'improvement_supported': False, 'conclusion': 'Quick descriptive engine comparison; inspect individual execution and quality outcomes'}})
        export(store, run_id, latest)
        print('Saved '+str(latest), flush=True)

if __name__ == '__main__':
    main()
