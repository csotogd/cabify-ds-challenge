"""Build and start the pinned local road engine; no public routing requests."""
from pathlib import Path
import subprocess,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/cache/roads'
REPORT=ROOT/'resultados/rutas_grafo'
IMAGE='ghcr.io/project-osrm/osrm-backend@sha256:55dbfd47d984c2cacd64e32901b4321c86b5ec761438e69aa053dc78698a786f'
NAME='cabify-challenge-osrm'
VOLUME='cabify-challenge-osrm-data'

def run(args):
    print('OSRM:',args[0],flush=True)
    subprocess.run(['docker','run','--rm','-v',f'{VOLUME}:/data',IMAGE]+args,check=True)

def main():
    exists=subprocess.run(['docker','volume','inspect',VOLUME],capture_output=True).returncode==0
    subprocess.run(['docker','volume','create',VOLUME],check=True)
    if not exists:
        for name in ['import','extract','partition','customize']:
            (DATA/f'{name}.done').unlink(missing_ok=True)
    if not (DATA/'import.done').exists():
        helper='cabify-osrm-import'
        subprocess.run(['docker','create','--name',helper,'-v',f'{VOLUME}:/data',IMAGE,'true'],check=True)
        try:
            subprocess.run(['docker','cp',str(DATA/'challenge.osm.pbf'),f'{helper}:/data/challenge.osm.pbf'],check=True)
        finally:subprocess.run(['docker','rm',helper],check=True)
        (DATA/'import.done').write_text(hashlib.file_digest((DATA/'challenge.osm.pbf').open('rb'),'sha256').hexdigest())
    stages=[('extract',['osrm-extract','-t','2','-p','/opt/car.lua','/data/challenge.osm.pbf']),
            ('partition',['osrm-partition','-t','2','/data/challenge.osrm']),
            ('customize',['osrm-customize','-t','2','/data/challenge.osrm'])]
    for name,args in stages:
        marker=DATA/f'{name}.done'
        if not marker.exists():run(args);marker.write_text(IMAGE)
    existing=subprocess.run(['docker','ps','-q','--filter',f'name=^/{NAME}$'],capture_output=True,text=True,check=True).stdout.strip()
    if not existing:
        subprocess.run(['docker','run','--rm','-d','--name',NAME,'-p','127.0.0.1:5057:5000',
            '-v',f'{VOLUME}:/data:ro',IMAGE,'osrm-routed','--algorithm','mld','--threads','4',
            '--max-matching-size','1000','/data/challenge.osrm'],check=True)
    (REPORT/'engine.json').write_text(json.dumps(dict(image=IMAGE,version='26.10.0',
        profile='/opt/car.lua',algorithm='MLD',traffic='none',host='127.0.0.1',port=5057,docker_volume=VOLUME,
        time_note='Model-based car-profile durations, not historical trip durations'),indent=2))

if __name__=='__main__':main()
