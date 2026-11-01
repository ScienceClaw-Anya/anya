"""Local native-tool check and exact payload preview; makes ZERO model calls."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_physical.environment import PhysicalAcquisition
from .experiment import verified
from .logic import prompts,baseline_ranking
from .public import public_state


def build(project,output):
    import torch
    from neurotarget.scienceclaw_runtime import Framework
    project,output=Path(project),Path(output); p=verified(project,output); torch.set_num_threads(2)
    case=next(c for c in p['cases'] if c['pilot']); folder=output/'local_review'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():
        print('LOCAL_REVIEW_ALREADY_EXISTS; use its saved source/preview hashes rather than overwriting it.'); return
    env=PhysicalAcquisition(project,case['outer'],case['subject']); catalog=read_json(output/'catalog'/f'{case["case_id"]}.json')
    raw=load_array(project/f'vendor/norris2026_hat/data/X_{case["subject"]}.npz')[:,:20]
    framework=Framework(project,folder/'scienceclaw'); parents=[]; artifacts=[]
    def artifact(agent,payload,kind):
        result=framework.artifact(agent,'neurotarget-spes-v16',payload,kind,parents.copy()); artifacts.append(result.artifact_id); return result.artifact_id
    prefix=fixed_queries(env.available,case['outer'],int(case['subject'][-2:]),0)[:4]
    for q in prefix:
        obs=env.observe(q); parents=[artifact('local_warmup',obs,'local_only_physical_response')]
    previous=None; tool_calls=0; first=None; snapshots=[]
    # The common tool accepts only public snapshots inside its actual whitelisted root.
    snapshot_folder=output/'trajectories/_local_data_review'; snapshot_folder.mkdir(parents=True,exist_ok=True)
    for r in range(4):
        public=public_state(env,catalog,raw,previous); path=snapshot_folder/f'{r:02d}_public.json'; write_json(path,public); snapshots.append(sha(path))
        if r==0:
            first=prompts(public,'single',0,[])[0]
            for policy in p['policies']:
                for slot,prompt in enumerate(prompts(public,policy,0,[])): write_json(folder/f'first_{policy}_{slot}.json',prompt)
        common=[]
        for slot in [0,1]:
            metadata=framework.registry.get_skill('neurotarget-spes-v16')
            result=framework.executor.execute_skill('neurotarget-spes-v16',metadata,{'project':str(project),'state_path':str(path),'format':'json'},timeout=90)
            if result.get('status')!='success' or result.get('result')!=public: raise RuntimeError('Actual local common tool mismatch')
            tool_calls+=1; common.append(artifact('local_tools',public,'actual_native_tool_no_model'))
        choice=baseline_ranking(public)[0]; parents=common; parents=[artifact('local_numeric',{'choice':choice,'scope':'Fixed score-focus choice; NO language-agent decision.'},'local_numeric_choice')]
        previous=env.predictions['hat'].copy(); obs=env.observe(int(choice[1:])); parents=[artifact('local_feedback',obs,'local_only_actual_feedback')]
    native_count=framework.verify_artifacts(); actual=env.final_judge(); old=read_json(project/'outputs/spes_agent_v11/numeric'/case['case_id']/'score_focus/metrics.json')
    if actual['queries']!=old['queries']: raise ValueError('Preview changed the existing numerical control')
    np.testing.assert_allclose(actual['prediction'],old['prediction'],rtol=0,atol=1e-7)
    encoded=json.dumps(first,indent=2,allow_nan=False)
    proposed={'destination':'https://generativelanguage.googleapis.com/v1beta/openai/','model':'gemini-4-argon',
      'purpose':'Five-patient diagnostic physical-pair acquisition comparison; single, multi and rationale-hidden arms.',
      'data_to_transmit':['Public deidentified recording-electrode coordinates, names and atlas regions.','Recorded stimulation condition metadata.',
                          'Only already selected response-group quality statistics and model scores/ranks.','Past agent reasoning, selected pairs and observed descriptive feedback.'],
      'excluded_from_request':['Clinical contact SOZ labels, resection status and surgical outcomes.','Raw EEG waveforms and unselected response summaries.','Patient/subject identifiers, user credentials and local source paths.'],
      'request_cap':120,'completion_token_cap_per_request':32768,'retries':0,
      'cost_note':'Paid API or workshop credits may be consumed; actual token use and charges depend on responses. No monetary price estimate is asserted.',
      'future_expansion':'Not included in this specific data-transmission approval; pilot outcome and any later approval determine expansion.'}
    # The proposed approval is bounded to the pilot; expansion is not triggered by this local check.
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Gemini pilot data review</title>
<style>body{max-width:1000px;margin:40px auto;padding:0 20px;font:16px/1.55 system-ui;color:#18283d}h1{line-height:1.2}pre{background:#f1f4f8;padding:18px;white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}strong{color:#6c2e19}</style>
<h1>Gemini pilot: exact data-transmission review</h1><p><strong>No new Gemini request has been sent.</strong> Automatic approval rejected the proposed run because this specific patient-derived payload and paid request scope require explicit authorization.</p>
<p>The source is a publicly released, deidentified SPES research dataset. The proposed recipient is Google's official Gemini model API. This review concerns five patients and at most 120 attempts, including the first eight-call smoke trajectory.</p>
<h2>Proposed scope</h2><pre>'''+html.escape(json.dumps(proposed,indent=2))+'''</pre>
<h2>Local verification completed</h2><p>Eight real response groups and eight native ScienceClaw tool executions were checked locally, with no model calls. The fixed numerical trajectory exactly reproduces the frozen v11 score-focus queries and scores. Native artifact integrity and lineage passed.</p>
<p>Language-agent reasoning and cross-round behavior remain untested on these real physical groups. The local numeric check is not a multi-agent result. All future requests retain the same field definitions; measured values and past reasoning vary by trajectory.</p>
<details><summary>Exact first proposed request (local preview only)</summary><pre>'''+html.escape(encoded)+'''</pre></details>
<p>This self-contained file makes no external requests. Raw data, credentials and generated request previews remain outside Git.</p></html>'''
    path=output/'data_review.html'; path.write_text(document,encoding='utf8')
    usages=list((output/'trajectories').glob('*/*/*_usage.json'))
    if usages: raise ValueError('Unexpected model-attempt records before explicit approval')
    manifest={'status':'local_pipeline_verified_external_approval_pending','model_calls':0,'actual_native_tool_calls':tool_calls,'native_artifacts_verified':native_count,
       'physical_response_groups':8,'matches_v11_score_focus':True,'data_review_sha256':sha(path),'protocol_sha256':sha(output/'protocol.json'),
       'first_request_sha256':sha(folder/'first_single_0.json'),'snapshot_sha256':snapshots,'proposed_transmission':proposed,
       'limits':['This local check uses a numerical rule, not a model substitute.','No paid model decision or multi-agent performance is inferred.']}
    write_json(folder/'manifest.json',manifest); print({k:v for k,v in manifest.items() if k not in ['proposed_transmission','snapshot_sha256']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); build(a.project,a.output)
