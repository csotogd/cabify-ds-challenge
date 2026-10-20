"""Piloto de importación OSM. No es un map matcher ni un modelo ya evaluado."""
import json
import re
import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
from pyproj import Geod
from .data import ROOT,load_data

DRIVABLE={'motorway','trunk','primary','secondary','tertiary','unclassified','residential',
          'living_street','service','motorway_link','trunk_link','primary_link','secondary_link','tertiary_link'}
# Supuestos ilustrativos para una demostración, no velocidades observadas.
FALLBACK_KPH={'motorway':90,'trunk':70,'primary':50,'secondary':40,'tertiary':35,
              'residential':25,'living_street':10,'service':15}

def load_graph(path=None):
    path=path or ROOT/'data/cache/osm_madrid_tiny.osm'
    if not path.exists():return None
    root=ET.parse(path).getroot()
    coords={int(n.attrib['id']):(float(n.attrib['lat']),float(n.attrib['lon'])) for n in root.findall('node')}
    g=nx.MultiDiGraph();geod=Geod(ellps='WGS84')
    for way in root.findall('way'):
        tags={t.attrib['k']:t.attrib['v'] for t in way.findall('tag')}
        highway=tags.get('highway')
        if highway not in DRIVABLE or tags.get('area')=='yes':continue
        if tags.get('access') in {'private','no'} or tags.get('motor_vehicle') in {'private','no'}:continue
        raw=tags.get('maxspeed','');match=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(mph|km/h)?\s*',raw)
        imputed=match is None
        speed=float(match[1])*(1.609344 if match[2]=='mph' else 1) if match else FALLBACK_KPH.get(highway,30)
        if speed<=0:continue
        nodes=[int(n.attrib['ref']) for n in way.findall('nd')]
        oneway=tags.get('oneway','yes' if tags.get('junction')=='roundabout' else 'no')
        for u,v in zip(nodes[:-1],nodes[1:]):
            if u not in coords or v not in coords:continue
            la,lo=coords[u];lb,lp=coords[v]
            length=geod.inv(lo,la,lp,lb)[2]
            if length<=0:continue
            for n in [u,v]:g.add_node(n,lat=coords[n][0],lon=coords[n][1])
            attrs=dict(length=length,speed_kph=speed,travel_time=length/(speed/3.6),
                       imputed_speed=imputed,highway=highway,way_id=int(way.attrib['id']))
            if oneway!='-1':g.add_edge(u,v,**attrs)
            if oneway not in {'yes','1','true'}:g.add_edge(v,u,**attrs)
    g.graph['source']='OpenStreetMap contributors, ODbL; https://www.openstreetmap.org/copyright'
    return g

def audit_graph():
    g=load_graph();df,routes=load_data()
    bounds=[40.415,-3.705,40.425,-3.695]
    complete=[]
    for jid,(a,b) in routes.items():
        p=np.concatenate([a,b]);s,w,n,e=bounds
        if ((p[:,0]>=s)&(p[:,0]<=n)&(p[:,1]>=w)&(p[:,1]<=e)).all():complete.append(jid)
    report=dict(status='pilot_only_not_evaluated',google_maps='No hay credenciales configuradas; no se han hecho consultas de pago.',
                source='OpenStreetMap contributors, ODbL',source_url='https://www.openstreetmap.org/copyright',
                bbox=bounds,nodes=len(g) if g else 0,edges=g.number_of_edges() if g else 0,
                complete_journeys_in_bbox=complete,
                reason='Las consultas amplias de Overpass agotaron tiempo o fueron rechazadas. Solo se obtuvo un extracto pequeño. No hay cobertura suficiente para comparar un modelo basado en calles y tiempos.',
                road_time_status='Solo demostración longitud/velocidad. Límites OSM o valores supuestos por tipo de vía; no duración real, tráfico histórico ni penalizaciones de giro.',
                missing=['Cobertura de todas las ciudades','Map matching de ambas rutas con confianza y control de continuidad',
                         'Restricciones de giro y permisos completos','Tiempos comparables, versionados y validados',
                         'Benchmark de características viarias sobre exactamente los mismos viajes'],
                no_road_features_in_trained_models=True)
    if g:
        report['fraction_edges_with_imputed_speed']=float(np.mean([d['imputed_speed'] for _,_,d in g.edges(data=True)]))
    (ROOT/'resultados/rutas/road_graph_status.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return g,report

if __name__=='__main__':audit_graph()
