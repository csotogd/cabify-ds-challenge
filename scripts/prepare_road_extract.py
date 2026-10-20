"""Recorta los PBF a los viajes + margen; conserva referencias y restricciones."""
from pathlib import Path
import sys
import json
import heapq
import time
import hashlib
import osmium
import numpy as np
import shapely
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from routes.data import load_data

CACHE=ROOT/'data/cache/geofabrik'
OUT=ROOT/'data/cache/roads'
REPORT=ROOT/'results/routes_graph'
COUNTRIES=['spain','colombia','argentina','chile','peru','ecuador','uruguay','mexico']


def crop_country(name,area):
    source=CACHE/f'{name}.osm.pbf'
    target=OUT/f'{name}.osm.pbf'
    report_path=REPORT/f'extract_{name}.json'
    if target.exists() and report_path.exists():
        info=json.loads(report_path.read_text())
        if info.get('area_hash')==hashlib.sha256(area.wkb).hexdigest():
            print(name, 'extracto existente',flush=True)
            return target
        raise ValueError('Coverage has changed; do not mix caches.')
    if not source.exists():
        raise FileNotFoundError(source)
    start=time.monotonic();selected=set();restrictions=0
    print(name,'selecting ways',flush=True)
    # Pyosmium ejecuta los filtros de entidades/tags en C++.
    with osmium.BackReferenceWriter(target,source,remove_tags=False) as writer:
        processor=(osmium.FileProcessor(source).with_locations()
                   .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
                   .with_filter(osmium.filter.KeyFilter('highway','route')))
        for way in processor:
            if 'highway' not in way.tags and way.tags.get('route')!='ferry':
                continue
            coords=[(n.lon,n.lat) for n in way.nodes if n.location.valid()]
            if len(coords)<2:continue
            x,y=zip(*coords)
            if not area.intersects(box(min(x),min(y),max(x),max(y))):
                continue
            writer.add_way(way);selected.add(way.id)
        print(name,len(selected),'ways; adding restrictions',flush=True)
        for rel in osmium.FileProcessor(source,entities=osmium.osm.RELATION):
            if rel.tags.get('type')!='restriction':continue
            if any(m.type=='w' and m.ref in selected for m in rel.members):
                writer.add_relation(rel);restrictions+=1
        print(name,'completando nodos de referencia',flush=True)
    info=dict(country=name,selected_ways=len(selected),restriction_relations=restrictions,
              bytes=target.stat().st_size,seconds=time.monotonic()-start,
              area_hash=hashlib.sha256(area.wkb).hexdigest(),
              sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),
              margin_degrees=.08,source='Geofabrik / OpenStreetMap',
              method='Ways whose bbox intersects the union of journey bboxes plus a margin; complete references and related restriction relations.')
    report_path.write_text(json.dumps(info,indent=2))
    print(name,'listo',round(info['bytes']/1e6,1),'MB',flush=True)
    return target


def ordered_objects(path):
    for obj in osmium.FileProcessor(path):
        rank=0 if obj.is_node() else 1 if obj.is_way() else 2
        yield (rank,obj.id,obj)


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    df,routes=load_data()
    boxes=[]
    for a,b in routes.values():
        p=np.concatenate([a,b]);south,west=p.min(axis=0)-.08;north,east=p.max(axis=0)+.08
        boxes.append(box(west,south,east,north))
    area=shapely.union_all(boxes)
    (REPORT/'coverage.geojson').write_text(json.dumps(shapely.geometry.mapping(area)))
    targets=[]
    for name in COUNTRIES:
        targets.append(crop_country(name,area))
    merged=OUT/'challenge.osm.pbf'
    if not merged.exists():
        print('Uniendo extractos en orden OSM',flush=True)
        counts=[0,0,0];duplicates=0;prior=None
        with osmium.SimpleWriter(merged) as writer:
            for rank,oid,obj in heapq.merge(*(ordered_objects(p) for p in targets),key=lambda r:r[:2]):
                key=(rank,oid)
                if key==prior:
                    duplicates+=1
                    continue
                writer.add(obj);counts[rank]+=1;prior=key
        info=dict(nodes=counts[0],ways=counts[1],relations=counts[2],duplicates_skipped=duplicates,
                  bytes=merged.stat().st_size,
                  sha256=hashlib.file_digest(merged.open('rb'),'sha256').hexdigest(),
                  journeys=len(df),countries=COUNTRIES,
                  snapshot_note='Current map downloaded with date and hash; not a historical reconstruction of the trip.')
        (REPORT/'extract_manifest.json').write_text(json.dumps(info,indent=2))
        print(json.dumps(info),flush=True)

if __name__=='__main__':main()
