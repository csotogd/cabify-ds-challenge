"""Un viaje por fila; particiones anteriores a cualquier ajuste aprendido."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261014
CENTERS = {
    'Buenos Aires': (-34.60, -58.45), 'Bogotá': (4.65, -74.10),
    'Lima': (-12.07, -77.04), 'Madrid': (40.42, -3.70),
    'Santiago': (-33.45, -70.65), 'Barcelona': (41.39, 2.17),
    'Medellín': (6.24, -75.58), 'Córdoba': (-31.42, -64.19),
    'Quito': (-0.18, -78.48), 'Valencia': (39.47, -0.38),
    'Sevilla': (37.39, -5.98), 'Mendoza': (-32.89, -68.84),
    'Montevideo': (-34.90, -56.16), 'Ciudad de México': (19.43, -99.13),
    'Cali': (3.45, -76.53),
}

def region_of(route):
    lat, lon = route[0]
    ds = {name: 111 * np.hypot(lat-a, (lon-b)*np.cos(np.radians(lat)))
          for name, (a,b) in CENTERS.items()}
    best = min(ds, key=ds.get)
    return best if ds[best] < 65 else 'Otras zonas'

def load_data():
    raw = ROOT/'data/raw/challenge_dataset.json'
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    prior_audit=ROOT/'results/routes/audit.json'
    if prior_audit.exists():
        expected=json.loads(prior_audit.read_text()).get('sha256')
        if expected and expected!=digest:
            raise ValueError('The dataset has changed. Do not reuse splits, caches or models from the previous benchmark.')
    records = json.loads(raw.read_text())
    grouped = defaultdict(list)
    for row in records:
        grouped[row['journey_id']].append(row)
    routes, rows = {}, []
    for jid, annotations in sorted(grouped.items()):
        first = annotations[0]
        for r in annotations:
            assert r['estimated_route'] == first['estimated_route']
            assert r['real_route'] == first['real_route']
        # One person gets one vote per journey, even if they appear several times.
        by_annotator=defaultdict(set)
        for r in annotations: by_annotator[r['annotator']].add(r['annotation'])
        votes=[];conflicted=0
        for labels in by_annotator.values():
            decisions=labels-{"I don't know"}
            if len(decisions)==1: votes.append(int('They differ' in decisions))
            elif len(decisions)>1: conflicted+=1
        p = float(np.mean(votes)) if votes else np.nan
        status = 'unknown' if not votes else 'tie' if p==.5 else 'different' if p>.5 else 'same'
        a, b = np.asarray(first['estimated_route'],float), np.asarray(first['real_route'],float)
        assert a.ndim==b.ndim==2 and a.shape[1]==b.shape[1]==2
        assert np.isfinite(a).all() and np.isfinite(b).all()
        route_hash = hashlib.sha256(json.dumps([first['estimated_route'],first['real_route']]).encode()).hexdigest()
        routes[jid] = (a,b)
        rows.append(dict(journey_id=jid, n_annotations=len(annotations), n_votes=len(votes),
                         n_unknown=sum(labels=={"I don't know"} for labels in by_annotator.values()),
                         n_unique_annotators=len(by_annotator),n_conflicted_annotators=conflicted,target_soft=p,
                         target_hard=int(p>.5) if votes and p!=.5 else np.nan,
                         status=status, region=region_of(b), route_hash=route_hash,
                         n_estimated=len(a), n_real=len(b)))
    df = pd.DataFrame(rows).set_index('journey_id')
    assert df.route_hash.is_unique, 'Duplicate geometries: group them before splitting.'
    manifest = ROOT/'results/routes/split.json'
    if manifest.exists():
        split = json.loads(manifest.read_text())
        assert set(split)==set(df.index), 'El dataset ha cambiado: no reutilizar el split.'
    else:
        # Exactly 1,432 / 477 / 477 for the available file.
        n_holdout = round(.2*len(df))
        train, rest = train_test_split(df.index, test_size=2*n_holdout, random_state=SEED,
                                      stratify=df.status)
        val, test = train_test_split(rest, test_size=n_holdout, random_state=SEED+1,
                                    stratify=df.loc[rest,'status'])
        split = {j:s for s, ids in [('train',train),('val',val),('test',test)] for j in ids}
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(split,indent=2))
    df['split'] = pd.Series(split)
    audit = dict(sha256=digest, n_records=len(records),
                 n_journeys=len(df), n_annotators=len({r['annotator'] for r in records}),
                 annotation_counts=dict(Counter(r['annotation'] for r in records)),
                 status_counts=dict(df.status.value_counts()),
                 split_counts=pd.crosstab(df.split,df.status).to_dict(),
                 duplicated_journey_annotator_records=len(records)-int(df.n_unique_annotators.sum()),
                 conflicted_journey_annotator_pairs=int(df.n_conflicted_annotators.sum()),
                 seed=SEED, split_policy='60/20/20 por journey_id, estratificado por estado inicial de etiqueta; se conserva tras deduplicar votos, antes de evaluar modelos.',
                 duplicate_policy='One vote per annotator and journey. Identical repeats collapse; a decided vote plus an abstention keeps the decision; contradictory decisions from the same person count as an abstention.')
    (ROOT/'results/routes/audit.json').write_text(json.dumps(audit,indent=2,default=int))
    return df, routes
