"""Fetch the pinned public author code; optionally fetch ~8 GB of Git LFS data."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

REPOSITORY='https://github.com/norrisjamie23/SOZ_localisation_SPES_HAT.git'
COMMIT='11f65182a2114440f1425656f91d0ebec51ee726'

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--download-data',action='store_true');a=p.parse_args()
    git=shutil.which('git')
    if not git:raise RuntimeError('Install Git first')
    target=a.project.resolve()/'vendor/norris2026_hat';target.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
    created=not target.exists()
    if created:subprocess.run([git,'clone','--no-checkout',REPOSITORY,str(target)],check=True,env=env)
    remote=subprocess.check_output([git,'-C',str(target),'remote','get-url','origin'],text=True).strip()
    if remote!=REPOSITORY:raise ValueError('Existing checkout has an unexpected origin')
    if not created and subprocess.check_output([git,'-C',str(target),'status','--porcelain'],text=True).strip():raise ValueError('Do not overwrite modified upstream files')
    subprocess.run([git,'-C',str(target),'checkout','--detach',COMMIT],check=True,env=env)
    if a.download_data:subprocess.run([git,'-C',str(target),'lfs','pull','--include=data/*'],check=True)
    print('Pinned multi-trial SPES author repository ready. Data are downloaded only with --download-data.')

if __name__=='__main__':main()
