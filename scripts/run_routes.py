"""Pipeline completo: descargar datos, extraer, preentrenar, seleccionar y evaluar."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rutas.data import ROOT,load_data
from rutas.features import build_features
from rutas.t2vec import train_encoder
from rutas.train import run

if __name__=='__main__':
    if not (ROOT/'data/raw/challenge_dataset.json').exists():
        import requests
        url='https://www.dropbox.com/s/rdm11r99pmi13if/challenge_dataset.json?dl=1'
        response=requests.get(url,timeout=120);response.raise_for_status()
        (ROOT/'data/raw').mkdir(parents=True,exist_ok=True)
        (ROOT/'data/raw/challenge_dataset.json').write_bytes(response.content)
    for p in ['data/cache','modelos/rutas','resultados/rutas','figuras/rutas']:(ROOT/p).mkdir(parents=True,exist_ok=True)
    df,routes=load_data();build_features()
    if not (ROOT/'data/cache/t2vec_embeddings.npz').exists():train_encoder(df,routes)
    run()
