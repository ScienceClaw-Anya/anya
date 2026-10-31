"""Continue the approved reconstruction after training, with explicit stage status."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from report import document


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--max-wait-hours',type=float,default=5)
    p.add_argument('--agent-workers',type=int,default=2)
    args=p.parse_args()
    project=args.project.resolve()
    track=project/'reproduction/norris2024'
    output=project/'outputs/reproduction/norris2024'
    upstream=project/'vendor/norris2024'
    report_path=project/'outputs/reports/paper_reproduction.html'
    status={'stage':'waiting_for_75_training_runs','cohort_difference_disclosed_and_user_approved':True}

    def update():
        status['updated_unix']=time.time()
        temporary=output/f'pipeline_status.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(status,indent=2));temporary.replace(output/'pipeline_status.json')
        temporary=report_path.with_suffix('.tmp')
        temporary.write_text(document(output),encoding='utf-8');temporary.replace(report_path)

    def stage(name, command, expected):
        if expected.exists():
            return
        status['stage']=name;update();print('STAGE',name,flush=True)
        with (output/'logs'/f'pipeline_{name}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                     env=dict(os.environ,PYTHONHASHSEED='0'))
            status['child_pid']=process.pid
            while process.poll() is None:
                update();time.sleep(20)
            status.pop('child_pid',None)
            if process.returncode:
                status['stage']='failed_'+name;status['exit_code']=process.returncode;update()
                raise RuntimeError(f'{name} failed; inspect its stage log. No demonstration fallback.')
        if not expected.exists():
            raise RuntimeError(f'{name} exited without its expected evidence file')
        update()

    started=time.monotonic()
    while True:
        table_path=output/'table1_reproduction.json'
        table=json.loads(table_path.read_text()) if table_path.exists() else {}
        count=sum(row['finished_folds'] for row in table.values())
        status['completed_primary_runs']=count
        update()
        if count==75:
            break
        if time.monotonic()-started>args.max_wait_hours*3600:
            status['stage']='training_wait_timeout';update()
            raise TimeoutError('Training did not produce all 75 completed evidence files within the explicit wait limit')
        time.sleep(20)
    stage('channel_sensitivity',[sys.executable,'-u','-X','utf8',str(track/'channel_sensitivity.py'),
          '--upstream',str(upstream),'--output',str(output)],output/'channel_sensitivity.json')
    stage('agent_comparison',[sys.executable,'-u','-X','utf8',str(track/'agent_comparison.py'),
          '--project',str(project),'--upstream',str(upstream),'--output',str(output),
          '--workers',str(args.agent_workers),'--allow-count-mismatch'],output/'agent_results.json')
    status['stage']='public_code_runs_and_agent_comparison_complete'
    status['exact_paper_reproduction']=False
    status['remaining_paper_gaps']=['cohort counts','original RNG/fold manifests','independent outcome labels and exact modified model']
    update();print('PIPELINE COMPLETE',flush=True)


if __name__=='__main__':main()
