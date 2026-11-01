"""Public VEC metadata/acquisition; never downloads the whole cohort by default."""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://data-proxy.ebrains.eu/api/v1/datasets/c1702a5b-bc8f-486e-a0ee-4758a707ff00'
CAP = 2 * 1024**3

def get(url):
    with urllib.request.urlopen(url, timeout=45) as r:
        if r.status != 200:
            raise RuntimeError(f'Unexpected HTTP {r.status}')
        return r.read()

def inventory():
    rows, marker = [], ''
    while True:
        q = urllib.parse.urlencode({'limit':1000, 'marker':marker})
        page = json.loads(get(BASE+'?'+q))['objects']
        if not page:
            break
        rows.extend(page)
        marker = page[-1]['name']
    (ROOT/'data/metadata/vec-inventory.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows

def acquire(rows, names):
    logpath=ROOT/'manifests/acquisition.json'
    log=json.loads(logpath.read_text()) if logpath.exists() else []
    used=sum(x['bytes'] for x in log)
    index={x['name']:x for x in rows}
    for name in names:
        target=ROOT/'data/vec'/name
        if target.exists():
            continue
        size=index[name]['bytes']
        if used+size>CAP:
            raise RuntimeError('Dataset download cap reached')
        url=BASE+'/'+urllib.parse.quote(name, safe='/')
        payload=get(url)
        if len(payload)!=size:
            raise RuntimeError(f'Size mismatch for {name}')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(payload)
        used+=len(payload)
        log.append(dict(provider_url=url, dataset_doi='10.25493/XWG2-S8X',
                        version='v1.0', license='CC-BY-4.0',
                        retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                        path=str(target.relative_to(ROOT)),bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(),
                        provider_hash=index[name]['hash']))
        logpath.write_text(json.dumps(log,indent=2)+'\n')
        print(name,len(payload),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',action='store_true');p.add_argument('names',nargs='*');a=p.parse_args()
    rows=inventory() if a.inventory else json.loads((ROOT/'data/metadata/vec-inventory.json').read_text())
    acquire(rows,a.names)
