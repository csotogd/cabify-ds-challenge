"""Reproducible public downloads; no API queries carrying the private routes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/cache/geofabrik"
REPORT = ROOT / "results/routes_graph"
SOURCES = {
    "madrid": "europe/spain/madrid",
    "spain": "europe/spain",
    "colombia": "south-america/colombia",
    "argentina": "south-america/argentina",
    "chile": "south-america/chile",
    "peru": "south-america/peru",
    "ecuador": "south-america/ecuador",
    "uruguay": "south-america/uruguay",
    "mexico": "north-america/mexico",
}


def download(name):
    DEST.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    target = DEST / f"{name}.osm.pbf"
    manifest = REPORT / f"download_{name}.json"
    if target.exists() and manifest.exists():
        info = json.loads(manifest.read_text())
        actual = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        if actual == info.get("sha256"):
            print(f"{name}: cache verified, {target.stat().st_size / 1e6:.1f} MB", flush=True)
            return info
        raise ValueError(f"El hash de {name} no coincide; no reutilizar.")
    url = f"https://download.geofabrik.de/{SOURCES[name]}-latest.osm.pbf"
    partial = target.with_suffix(".pbf.part")
    started = time.monotonic()
    print(f"{name}: descargando {url}", flush=True)
    with requests.get(url, stream=True, timeout=(20, 60),
                      headers={"User-Agent": "CabifyInterviewResearch/1.0 (offline OSM analysis)"}) as response:
        response.raise_for_status()
        digest = hashlib.sha256()
        size = 0
        last_report = time.monotonic()
        with partial.open("wb") as output:
            for block in response.iter_content(1024 * 1024):
                if not block:
                    continue
                output.write(block)
                digest.update(block)
                size += len(block)
                if time.monotonic() - last_report > 20:
                    print(f"{name}: {size / 1e6:.1f} MB recibidos", flush=True)
                    last_report = time.monotonic()
        expected = response.headers.get("Content-Length")
        if expected and int(expected) != size:
            raise ValueError(f"Descarga incompleta de {name}: {size}/{expected}")
        info = dict(name=name, url=url, bytes=size, sha256=digest.hexdigest(),
                    last_modified=response.headers.get("Last-Modified"),
                    etag=response.headers.get("ETag"),
                    downloaded_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    seconds=time.monotonic() - started,
                    attribution="© OpenStreetMap contributors; ODbL; extracts by Geofabrik")
    partial.replace(target)
    manifest.write_text(json.dumps(info, indent=2))
    print(f"{name}: listo, {size / 1e6:.1f} MB, hash registrado", flush=True)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("regions", nargs="*", default=[x for x in SOURCES if x != "madrid"])
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(download, name): name for name in args.regions}
        errors = []
        for future in as_completed(jobs):
            try:
                future.result()
            except Exception as exc:
                errors.append((jobs[future], str(exc)))
                print(f"ERROR {jobs[future]}: {exc}", flush=True)
    if errors:
        raise SystemExit(str(errors))
