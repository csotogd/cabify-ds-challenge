"""Optional OSM network download for a geographic pilot; it uses no labels."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import requests
from routes.data import load_data, ROOT

df,routes=load_data()
city=sys.argv[1] if len(sys.argv)>1 else 'Madrid'
selected=df.index[df.region==city]
points=np.concatenate([p for j in selected for p in routes[j]])
south,west=points.min(axis=0)-.015
north,east=points.max(axis=0)+.015
query=f'''[out:json][timeout:90];way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"]({south},{west},{north},{east});out tags geom;'''
slug=city.lower().replace(' ','_')
dest=ROOT/f'data/cache/osm_{slug}.json'
report=dict(city=city,n_journeys=len(selected),bbox=[south,west,north,east],
            source='OpenStreetMap contributors / Overpass API',query=query,errors=[])
for endpoint in ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter']:
    if dest.exists():
        report['status']='cached'; break
    try:
        response=requests.post(endpoint,data={'data':query},timeout=110,
                               headers={'User-Agent':'CabifyInterviewResearch/1.0'})
        response.raise_for_status()
        data=response.json()
        if not data.get('elements'):raise ValueError('No road geometries returned')
        dest.write_text(json.dumps(data,separators=(',',':')))
        report.update(status='downloaded',endpoint=endpoint,n_ways=len(data['elements']),
                      osm_timestamp=data.get('osm3s',{}).get('timestamp_osm_base'))
        break
    except Exception as e:
        report['errors'].append(f'{endpoint}: {type(e).__name__}: {e}')
else: report['status']='failed'
(ROOT/f'results/routes/osm_{slug}_download.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,ensure_ascii=False),flush=True)
