"""Exploratory road features: local OSRM matching + directed OSM distances.

No human labels are used here. The node graph is deliberately simpler than
OSRM's turn-aware routing graph; its distances must not be called legal ETAs.
"""
import json
import time
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
import osmium
from pyproj import Geod
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.csgraph import dijkstra, connected_components
from .data import ROOT, load_data
from .road_graph import DRIVABLE

CACHE = ROOT/'data/cache/roads'
REPORT = ROOT/'resultados/rutas_grafo'
GEOD = Geod(ellps='WGS84')
MATCH_CONFIG = dict(max_points=300, radius_m=15, tidy=False, timestamps=False)


def sample_observed(route, maximum=300):
    """Choose observed points by arc length, never invent intermediate GPS fixes."""
    p=np.asarray(route,float)
    p=p[np.r_[True,np.any(np.diff(p,axis=0)!=0,axis=1)]]
    if len(p)<=maximum:return p
    lengths=GEOD.inv(p[:-1,1],p[:-1,0],p[1:,1],p[1:,0])[2]
    along=np.r_[0,np.cumsum(lengths)]
    ids=np.unique(np.r_[0,np.searchsorted(along,np.linspace(0,along[-1],maximum)),len(p)-1])
    return p[ids]


def match_one(item, server='http://127.0.0.1:5057'):
    jid,kind,route=item
    path=CACHE/'matches'/f'{jid}_{kind}.json'
    if path.exists():return json.loads(path.read_text())
    p=sample_observed(route,MATCH_CONFIG['max_points'])
    coords=';'.join(f'{lon:.6f},{lat:.6f}' for lat,lon in p)
    params=dict(overview='full',geometries='geojson',annotations='nodes,distance,duration',
                gaps='split',tidy='false',radiuses=';'.join(['15']*len(p)),generate_hints='false')
    for attempt in range(3):
        try:
            response=requests.get(f'{server}/match/v1/driving/{coords}',params=params,timeout=90)
            value=response.json()
            if value.get('code') not in {'Ok','NoMatch','NoSegment'}:
                raise RuntimeError(str(value)[:250])
            break
        except (requests.RequestException,ValueError) as exc:
            if attempt==2:raise RuntimeError(f'OSRM transport failure {jid}/{kind}') from exc
            time.sleep(1)
    value.update(journey_id=jid,kind=kind,input_points=p.tolist(),config=MATCH_CONFIG)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
    return value


def run_matching():
    df,routes=load_data();(CACHE/'matches').mkdir(exist_ok=True)
    tasks=[(jid,kind,p) for jid,(a,b) in routes.items() for kind,p in [('estimated',a),('real',b)]]
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(match_one,tasks),1):
            if i%100==0:print('Map matching',i,'/',len(tasks),'segundos',round(time.monotonic()-start),flush=True)
    (REPORT/'matching_config.json').write_text(json.dumps(MATCH_CONFIG,indent=2))


def build_node_graph():
    """OSM node graph: static access/oneway, no turn restrictions or traffic."""
    graph_path=CACHE/'directed_lengths.npz';nodes_path=CACHE/'graph_nodes.npz'
    metadata=REPORT/'node_graph.json'
    if graph_path.exists() and nodes_path.exists() and metadata.exists() and json.loads(metadata.read_text()).get('version')==2:
        z=np.load(nodes_path)
        return load_npz(graph_path),z['ids'],z['coords']
    node_coords={};edges={};ways=0
    processor=(osmium.FileProcessor(CACHE/'challenge.osm.pbf').with_locations()
               .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
               .with_filter(osmium.filter.KeyFilter('highway')))
    for way in processor:
        tags=dict(way.tags)
        if tags.get('highway') not in DRIVABLE or tags.get('area')=='yes':continue
        access=next((tags[k] for k in ['motorcar','motor_vehicle','vehicle','access'] if k in tags),'yes')
        if access in {'no','private','agricultural','forestry'}:continue
        ns=[n for n in way.nodes if n.location.valid()]
        if len(ns)!=len(way.nodes) or len(ns)<2:continue
        coords=np.array([(n.lon,n.lat) for n in ns]);ids=[n.ref for n in ns]
        lengths=GEOD.inv(coords[:-1,0],coords[:-1,1],coords[1:,0],coords[1:,1])[2]
        oneway=next((tags[k] for k in ['oneway:motorcar','oneway:motor_vehicle','oneway:vehicle','oneway'] if k in tags),
            'yes' if tags.get('junction') in {'roundabout','circular'} or tags.get('highway')=='motorway' else 'no')
        for i,(u,v,length) in enumerate(zip(ids[:-1],ids[1:],lengths)):
            if length<=0:continue
            node_coords[u]=coords[i];node_coords[v]=coords[i+1]
            directions=[]
            if oneway!='-1':directions.append((u,v))
            if oneway not in {'yes','1','true'}:directions.append((v,u))
            for edge in directions:edges[edge]=min(edges.get(edge,np.inf),float(length))
        ways+=1
    ids=np.array(sorted(node_coords),np.int64);lookup={n:i for i,n in enumerate(ids)}
    coords=np.array([node_coords[n] for n in ids])
    rows=np.fromiter((lookup[u] for u,v in edges),np.int32)
    cols=np.fromiter((lookup[v] for u,v in edges),np.int32)
    data=np.fromiter(edges.values(),np.float64)
    graph=csr_matrix((data,(rows,cols)),shape=(len(ids),len(ids)))
    save_npz(graph_path,graph);np.savez_compressed(nodes_path,ids=ids,coords=coords)
    (REPORT/'node_graph.json').write_text(json.dumps(dict(version=2,nodes=len(ids),directed_edges=len(data),ways=ways,
        limitations=['No turn restrictions','No conditional access','No ferries','Current OSM snapshot'],
        distance_limit_m=10000),indent=2))
    return graph,ids,coords


def weighted_overlap(a,b):
    keys=set(a)|set(b)
    union=sum(max(a.get(e,0),b.get(e,0)) for e in keys)
    shared=sum(min(a.get(e,0),b.get(e,0)) for e in keys)
    return shared/union if union else np.nan


def to_target_distances(reversed_graph,targets,limit=10000):
    """Distances source->any target; reverse first, NOT target->source."""
    return dijkstra(reversed_graph,directed=True,indices=np.unique(targets),
                    min_only=True,limit=limit)


def summarize_match(raw,lookup,coords):
    matches=raw.get('matchings',[]);points=raw.get('tracepoints',[])
    fraction=sum(p is not None for p in points)/len(raw['input_points'])
    length=sum(m['distance'] for m in matches)
    confidence=np.average([m['confidence'] for m in matches],weights=[max(m['distance'],1) for m in matches]) if matches else 0.
    good=bool(len(matches)==1 and fraction>=.9 and confidence>=.5)
    f=dict(matched_fraction=fraction,confidence=float(confidence),subtraces=len(matches),
           reliable=int(good),partial_length_m=length,
           partial_duration_s=sum(m['duration'] for m in matches))
    # OSRM reports the parent OSM nodes and the length actually traversed for
    # each segment (including partial segments at GPS waypoints). Weight the
    # distance summaries by those partial lengths, not full edge lengths per fix.
    weights=defaultdict(float);edges={};seen=set();missing=set()
    for m in matches:
        for leg in m['legs']:
            annotation=leg.get('annotation',{})
            ns=annotation.get('nodes',[]);partial=annotation.get('distance',[])
            if ns and len(partial)!=len(ns)-1:
                raise ValueError('OSRM node/distance annotations are misaligned')
            seen.update(ns);missing.update(n for n in ns if n not in lookup)
            for u,v,traversed in zip(ns[:-1],ns[1:],partial):
                if traversed<=0 or u==v or u not in lookup or v not in lookup:continue
                a,b=coords[lookup[u]],coords[lookup[v]]
                d=GEOD.inv(a[0],a[1],b[0],b[1])[2]
                if d<=0:continue
                edges[(u,v)]=max(edges.get((u,v),0.),d)
                weights[lookup[u]]+=traversed/2;weights[lookup[v]]+=traversed/2
    f['fraction_nodes_missing']=len(missing)/max(len(seen),1)
    return f,edges,weights


def distance_summary(distances,weights,scale):
    d=np.asarray(distances);w=np.asarray(weights,float);w=w/w.sum()
    # Capped means are LOWER BOUNDS when there is censoring, not exact means.
    capped=np.minimum(d,10000.)
    order=np.argsort(capped);p95=capped[order][min(np.searchsorted(np.cumsum(w[order]),.95),len(d)-1)]
    result=dict(mean_capped_m=float(w@capped),p95_capped_m=float(p95),max_capped_m=float(capped.max()),
                fraction_unreachable_or_over_10km=float(w[~np.isfinite(d)].sum()))
    result.update({f'fraction_over_{x}m':float(w[d>x].sum()) for x in [50,100,500]})
    result.update(mean_capped_relative=result['mean_capped_m']/scale,max_capped_relative=result['max_capped_m']/scale)
    return result


def build_graph_features():
    out=CACHE/'graph_features.json'
    if out.exists():return pd.read_json(out,orient='index')
    graph,ids,coords=build_node_graph();lookup={n:i for i,n in enumerate(ids)}
    # Independent countries/cities cannot be reached: compute on exactly the
    # relevant weak components, without changing any finite shortest distance.
    _,labels=connected_components(graph,directed=False)
    by_component=np.argsort(labels,kind='stable');counts=np.bincount(labels)
    starts=np.r_[0,np.cumsum(counts)]
    @lru_cache(maxsize=8)
    def component_graph(key):
        nodes=np.sort(np.concatenate([by_component[starts[k]:starts[k+1]] for k in key]))
        return nodes,graph[nodes][:,nodes].T.tocsr()
    df,_=load_data();rows={};start=time.monotonic()
    for i,jid in enumerate(df.index):
        pair=[summarize_match(json.loads((CACHE/'matches'/f'{jid}_{kind}.json').read_text()),lookup,coords)
              for kind in ['estimated','real']]
        a,b=pair;f={f'{kind}_{k}':v for kind,result in zip(['est','real'],pair) for k,v in result[0].items()}
        scale=max((a[0]['partial_length_m']+b[0]['partial_length_m'])/2,1.)
        f['directed_edge_iou']=weighted_overlap(a[1],b[1]) if a[1] and b[1] else np.nan
        # Expose the quality of these approximations. No journey is filtered out.
        f['both_reliable']=a[0]['reliable']*b[0]['reliable']
        for quantity in ['length_m','duration_s']:
            x=a[0]['partial_'+quantity];y=b[0]['partial_'+quantity]
            f['relative_'+quantity+'_delta']=(y-x)/max((x+y)/2,1.) if f['both_reliable'] else np.nan
        for source,target,name in [(a,b,'est_to_real'),(b,a,'real_to_est')]:
            template=distance_summary(np.array([0.]),np.array([1.]),1.)
            for key in template:f[name+'_'+key]=np.nan
            if source[2] and target[2]:
                sources=np.array(list(source[2]));targets=np.array(list(target[2]))
                components=tuple(np.unique(labels[np.r_[sources,targets]]))
                nodes,reverse=component_graph(components)
                d=to_target_distances(reverse,np.searchsorted(nodes,targets))
                summary=distance_summary(d[np.searchsorted(nodes,sources)],list(source[2].values()),scale)
                f.update({name+'_'+k:v for k,v in summary.items()})
        rows[jid]=f
        if (i+1)%100==0:print('Variables de grafo',i+1,'/',len(df),'segundos',round(time.monotonic()-start),flush=True)
    features=pd.DataFrame.from_dict(rows,orient='index')
    features.to_json(out,orient='index',double_precision=12)
    coverage=[]
    for split in ['train','val','test']:
        subset=features.loc[df.index[df.split==split]]
        coverage.append(dict(split=split,n=len(subset),both_reliable=int(subset.both_reliable.sum()),
            both_any_match=int(((subset.est_subtraces>0)&(subset.real_subtraces>0)).sum()),
            median_est_confidence=float(subset.est_confidence.median()),median_real_confidence=float(subset.real_confidence.median()),
            mean_est_matched_fraction=float(subset.est_matched_fraction.mean()),mean_real_matched_fraction=float(subset.real_matched_fraction.mean())))
    (REPORT/'feature_manifest.json').write_text(json.dumps(dict(n=len(features),n_features=len(features.columns),
        columns=list(features),coverage=coverage,missing_per_column=features.isna().sum().to_dict(),
        distance='Directed node-graph shortest distance, not turn-aware OSRM distance; capped at 10 km',
        overlap='Length weighted unique directed node-edge IoU; touching part of an edge counts the full edge (endpoint approximation)',
        time='OSRM car-profile estimate on matched portions, not observed elapsed time',
        quality_rule='One subtrace, >=90% sampled points matched, OSRM confidence >=0.5; descriptive, not calibrated'),indent=2))
    return features


if __name__=='__main__':
    run_matching();build_graph_features()
