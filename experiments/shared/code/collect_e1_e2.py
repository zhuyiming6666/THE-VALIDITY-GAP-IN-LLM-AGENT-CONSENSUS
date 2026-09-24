"""Resumable E1 calibration and E2 temperature collection; no manuscript edits."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import dotenv_values
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'grain-iclr11'
OLD = ROOT / 'llmbft'
OUT = PAPER / 'results/e1_e2_20260919'
sys.path.insert(0, str(OLD / 'exp'))
from instrument_error import JUDGE
from measure_pmax import SCHEMA_COT
from common import parse
sys.path.insert(0, str(OLD / 'v5/code'))
from annotation_normalization import answer_class, norm_key


def read(path):
    return json.loads(path.read_text())


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    reference = PAPER / 'audit/annotation_scoring.rechecked.json'
    ref = read(reference)
    pairs_path = OLD / 'v5/annotation/pairs_blind.tsv'
    assert sha(pairs_path) == ref['input_sha256']['blank_template']
    pairs = list(csv.DictReader(pairs_path.open(), delimiter='\t'))
    assert {r['pair_id'] for r in pairs} == {r['pair_id'] for r in ref['per_pair']}
    all_tasks = {r['id']: r for r in map(json.loads, (OLD / 'data/tasks.jsonl').read_text().splitlines())}
    baseline = read(OLD / 'v3/results/e1_analysis.json')
    tids = baseline['common_tasks']['ids']
    assert len(tids) == 123
    tasks = [all_tasks[t] for t in tids]
    return ref, pairs, tasks


def client():
    env = dotenv_values(OLD / 'v3/code/.env')
    key = os.getenv('OPENAI_API_KEY') or env.get('OPENAI_API_KEY') or env.get('AICODING_API_KEY')
    url = os.getenv('OPENAI_BASE_URL') or env.get('OPENAI_BASE_URL') or env.get('AICODING_BASE_URL')
    if not key:
        raise RuntimeError('No configured API credential')
    if url and urlsplit(url).path in ('', '/'):
        url = url.rstrip('/') + '/v1'
    return OpenAI(api_key=key, base_url=url, timeout=60, max_retries=1)


def chat(c, model, system, user, temperature, limit):
    r = c.chat.completions.create(model=model, temperature=temperature, max_tokens=limit,
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
    return {'raw': r.choices[0].message.content, 'returned_model': r.model,
            'usage': r.usage.model_dump() if r.usage else None,
            'finish_reason': r.choices[0].finish_reason, 'timestamp': time.time()}


def collect(mode, workers):
    ref, pairs, tasks = inputs()
    c = client()
    if mode == 'preflight':
        statuses = {}
        for model in ('gpt-4.1-nano', 'gpt-4.1-mini'):
            try:
                r = chat(c, model, 'Reply with OK.', 'API availability check.', 0, 5)
                statuses[model] = {'ok': True, 'returned_model': r['returned_model']}
            except Exception as e:
                statuses[model] = {'ok': False, 'error_type': type(e).__name__, 'status': getattr(e, 'status_code', None)}
        try:
            r = c.embeddings.create(model='text-embedding-3-small', input=['availability check'])
            statuses['embedding'] = {'ok': True, 'dimension': len(r.data[0].embedding)}
        except Exception as e:
            statuses['embedding'] = {'ok': False, 'error_type': type(e).__name__, 'status': getattr(e, 'status_code', None)}
        save('preflight.json', statuses)
        print(json.dumps(statuses), flush=True)
        return
    jobs = []
    if mode == 'e1':
        for p in pairs:
            for model in ('gpt-4.1-nano', 'gpt-4.1-mini'):
                jobs.append({'id': p['pair_id'] + ':' + model, 'pair': p, 'model': model})
        for p in pairs:
            jobs.append({'id': p['pair_id'] + ':embedding', 'pair': p, 'model': 'embedding'})
    else:
        jobs = [{'id': f'{t}:{p["id"]}:{i}', 'task': p, 'temperature': t, 'rep': i}
                for t in (0.3, 0.7) for p in tasks for i in range(16)]
        random.Random(20260919).shuffle(jobs)
    path = OUT / (mode + '_raw.jsonl')
    existing = {r['id']: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
    jobs = [j for j in jobs if not existing.get(j['id'], {}).get('ok')]
    def one(j):
        try:
            if mode == 'e1':
                p = j['pair']
                if j['model'] == 'embedding':
                    texts = [f'{p[s+"_answer"]} || {p[s+"_key"]}' for s in ('A', 'B')]
                    response = c.embeddings.create(model='text-embedding-3-small', input=texts)
                    vectors = np.array([d.embedding for d in sorted(response.data, key=lambda d: d.index)])
                    sim = float(np.dot(*vectors) / (np.linalg.norm(vectors[0])*np.linalg.norm(vectors[1]) + 1e-12))
                    r = {'cosine': sim, 'prediction': sim > 0.9, 'usage': response.usage.model_dump()}
                else:
                    u = f'Question:\n{p["question"]}\n\nAgent 1 -- ANSWER: {p["A_answer"]} | KEY: {p["A_key"]}\nAgent 2 -- ANSWER: {p["B_answer"]} | KEY: {p["B_key"]}'
                    r = chat(c, j['model'], JUDGE, u, 0, 5)
                    label = (r['raw'] or '').strip().upper().rstrip('.')
                    if label not in ('SAME', 'DIFFERENT'):
                        return {'id': j['id'], 'ok': False, 'error_type': 'InvalidJudgeLabel', **r}
                    r['prediction'] = label == 'SAME'
            else:
                r = chat(c, 'gpt-4.1-nano', SCHEMA_COT, j['task']['question'], j['temperature'], 700)
                a, k = parse(r['raw'])
                r.update(task_id=j['task']['id'], temperature=j['temperature'], rep=j['rep'], answer=a, key=k)
            return {'id': j['id'], 'ok': True, **r}
        except Exception as e:
            return {'id': j['id'], 'ok': False, 'error_type': type(e).__name__, 'status': getattr(e, 'status_code', None)}
    print(f'{mode}: {len(jobs)} pending', flush=True)
    with path.open('a') as fh, ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        failed = 0
        for n, future in enumerate(as_completed(futures), 1):
            r = future.result()
            fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
            failed += not r['ok']
            if n % 50 == 0 or n == len(jobs):
                print(f'{mode}: {n}/{len(jobs)}, failures={failed}', flush=True)
            if n >= 20 and failed == n:
                for f in futures: f.cancel()
                raise RuntimeError('All initial calls failed; stopped collection')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'preflight', 'e1', 'e2'])
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    ref, pairs, tasks = inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == 'prepare':
        save('manifest.json', {'seed': 20260919, 'pairs': len(pairs), 'joint_resolved': sum(r['joint'] is not None for r in ref['per_pair']),
            'tasks': [t['id'] for t in tasks], 'bench_counts': dict(collections.Counter(t['bench'] for t in tasks)),
            'judge_prompt': JUDGE, 'embedding_text': 'answer || KEY', 'embedding_guard': 'cosine > 0.9',
            'generation_prompt': SCHEMA_COT, 'temperatures_new': [0.3,0.7], 'k': 16,
            'reference_sha256': sha(PAPER / 'audit/annotation_scoring.rechecked.json'),
            'pairs_sha256': sha(OLD / 'v5/annotation/pairs_blind.tsv'),
            'normalizer_sha256': sha(OLD / 'v5/code/annotation_normalization.py'),
            'baseline_files': ['llmbft/results/samples.jsonl','llmbft/results/samples2.jsonl'],
            'limitations': ['Historical T=1 baseline has a different collection date/backend snapshot.',
                'Original judges target joint answer-and-operation equivalence, not operation alone.',
                'No transitivity inference from incomplete pair triangles.']})
        print('Prepared 300 matched calibration pairs and 123 fixed tasks.')
    else:
        collect(a.mode, a.workers)
