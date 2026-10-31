"""Fetch and verify the two public source commits without executing downloaded code."""
import argparse
from pathlib import Path
import shutil
import subprocess


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--git',default='git')
    args=p.parse_args();project=args.project.resolve()
    sources=[('https://github.com/norrisjamie23/Localising_SOZ_from_SPES.git','vendor/norris2024','90062e8c5fe5593713125eaf5f573faedb739f65'),
             ('https://github.com/OpenNeuroDatasets/ds004080.git','data/ds004080','c4fd7418883e33b024292468eb14da1649f51aae')]
    for url,relative,commit in sources:
        folder=project/relative
        if not folder.exists():
            folder.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run([args.git,'clone','--depth','1',url,str(folder)],check=True)
        actual=subprocess.check_output([args.git,'-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        if actual!=commit:
            raise RuntimeError(f'{relative}: expected pinned commit {commit}, found {actual}; do not silently reset a user checkout')
        print(relative,actual)
