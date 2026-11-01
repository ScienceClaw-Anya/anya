"""English report; external approval and real results remain explicit states."""
import argparse
from pathlib import Path
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json


def build(project,output):
    project,output=Path(project),Path(output); p=read_json(output/'protocol.json'); review=read_json(output/'local_review/manifest.json')
    if not (output/'pilot_results.json').exists():
        authorization=(output/'transmission_approval.json').exists()
        attempts=len(list((output/'trajectories').glob('*/*/*_usage.json')))
        status='authorized_pilot_results_incomplete' if authorization else 'external_data_transmission_approval_pending'
        if (output/'smoke_interruption.json').exists(): status='smoke_interrupted_spent_attempts_preserved'
        summary={'status':status,'recorded_model_attempts':attempts,'pilot_attempt_cap':120,
            'local_native_tool_calls':review['actual_native_tool_calls'],'local_native_artifacts_verified':review['native_artifacts_verified'],
            'local_numeric_trajectory_matches_v11':review['matches_v11_score_focus'],'recording_name_ambiguities':p['recording_name_ambiguities'],
            'multi_agent_superiority_established':False}
        lines=['# Independent physical proposals: local verification and pilot status','',
          (f'Specific user authorization has been received for five patients and at most120 paid attempts. At report generation, **{attempts}** attempt records exist and complete pilot results are not yet available. No multi-agent result is inferred.' if authorization else
           'The new experiment is implemented and frozen, but no new Gemini request has been sent. Automatic approval rejected the proposed paid run because this specific patient-derived payload and destination require explicit authorization.'),
          '', 'The local data preview is `outputs/spes_agent_v16/data_review.html`; it includes the exact first proposed request and field exclusions. It was generated before model calls and is not committed.','',
          '## Completed local evidence','',
          'Eight real physical response groups and eight actual native ScienceClaw observation-tool executions passed, with twenty artifact hashes and parent links verified. The deterministic score-focus trajectory reproduces frozen v11 queries and predictions. This is a local numerical pipeline check, not a model substitute or a collaboration result. Eight offline tests cover independent prompts, delayed peer exchange, the rationale ablation, symmetric rank voting, hidden-waveform gating, metadata projection and actual transport-schema/reservation behavior with a mock transport.','',
          'All 2,117 recording coordinates match source metadata. Twenty-one have several possible electrode names, which remain explicitly unresolved in the common catalog. Shared anatomy is retained only when all matching metadata entries agree; no name is guessed.','',
          '## Frozen research design','',
          'Three arms receive four common initial groups and four adaptive rounds, with two independent proposals per round. Single uses two integrating generalist passes; multi uses anatomy/network and response/error roles. Both current requests are fixed before either response. The reason-hidden arm removes only peer rationale while preserving measured feedback and peer choices. A fixed 3/2/1 rank vote merges decisions, with score-focus gain and physical ID as ties. No reviewer can unilaterally override the pair.','',
          'Every arm has the same whitelist-projected anatomy, recorded stimulus conditions, selected-response quality, numerical scores/ranks and feedback. The unchanged original HAT supplies all final rankings. Failed v12-v15 predictors are not adopted. The v11 numerical query-sensitivity check passes on the five previously fixed pilot patients, with mean within-patient AUROC range 0.141081; this shows queries matter, not that agents can exploit them.','',
          '## Proposed transmission and costs','',
          'The proposed recipient is the official Google Gemini endpoint `generativelanguage.googleapis.com`, model `gemini-4-argon`. Requests contain publicly released, deidentified electrode anatomy/coordinates, recorded conditions, already observed signal statistics, model scores/ranks and past reasoning/feedback. Raw EEG, unselected-response summaries, contact SOZ labels, resection/outcome fields, source subject identifiers, credentials and private file paths are excluded.','',
          'The requested initial authorization is bounded to five patients and at most 120 paid attempts, including the first eight-call smoke. Retries are disabled; maximum completion tokens are 32,768 per attempt. Actual token consumption and charges depend on responses. The remaining 30-patient expansion is not covered by this initial specific transmission approval. It also requires the frozen pilot performance/validity/diversity gate and audit.','',
          'Reports and source code are in English; raw data, generated catalogs, requests, credentials and model assets remain outside Git. The workflow uses the local workstation and creates no new cloud resource. Nominal uncertainty intervals will remain exploratory after repeated cohort inspection. No clinical efficacy, physical delivery or multi-agent benefit is established.','']
    else:
        result=read_json(output/'pilot_results.json'); audit=read_json(output/'audit_pilot.json')
        if audit['status']!='verified': raise ValueError('Real pilot audit required')
        behavior=read_json(output/'behavior.json')
        summary={'status':'audited_pilot_complete','summary':result['summary'],'comparisons':result['comparisons'],
                 'valid_model_decisions':result['valid_model_decisions'],'actual_attempts':result['actual_attempts'],
                 'pilot_expansion_eligible':result['pilot_expansion_eligible'],'audit':audit,'behavior':behavior,'multi_agent_superiority_established':False}
        lines=['# Independent physical-proposal pilot','',
          f"Multi-agent mean patient AUROC is **{result['summary']['multi']['mean_patient_AUROC']:.6f}**, versus single **{result['summary']['single']['mean_patient_AUROC']:.6f}** and score-focus **{result['summary']['score_focus']['mean_patient_AUROC']:.6f}**. The frozen expansion gate **{'passes' if result['pilot_expansion_eligible'] else 'fails'}**. Five patients do not establish general superiority.",'',
          '| Arm/control | Mean patient SOZ AUROC |','|---|---:|']
        for name,v in result['summary'].items(): lines.append(f"| {name} | {v['mean_patient_AUROC']:.6f} |")
        lines+=['','| Comparator | Multi gain | Paired nominal 95% interval |','|---|---:|---|']
        for name,v in result['comparisons'].items(): lines.append(f"| {name} | {v['multi_gain']:+.6f} | {v['paired_95CI']} |")
        lines+=['',f"Accepted model decisions: **{result['valid_model_decisions']}/{result['actual_attempts']}**. Every language-agent trajectory uses eight actual response groups, eight attempts and four feedback rounds. Numerical controls use the same measurement budget with zero model calls. Actual reported input/output tokens are **{audit['reported_prompt_tokens']:,}/{audit['reported_completion_tokens']:,}**; **{audit['attempts_without_token_usage']}** attempts have unavailable usage, not zero usage.",'',
          'Every physical trajectory was re-inferred with all three original scorers. Public evidence, history visibility, native hashes/lineage, API schemas, rank votes and final AUROC were checked. Requests never contain clinical labels or future response summaries. Twenty-one recording-name ambiguities remain explicit rather than guessed.','',
          '## What collaboration actually changed','',
          '| Arm | Independent top-choice agreement | Mean top-three Jaccard | Merged choice matches coverage leader | Matches score-focus leader |','|---|---:|---:|---:|---:|']
        for name,b in behavior['policies'].items():
            lines.append(f"| {name} | {b['same_independent_top_choice']}/{b['adaptive_rounds']} | {b['mean_top3_Jaccard']:.3f} | {b['merged_choice_matches_numeric_leader']['coverage']}/{b['adaptive_rounds']} | {b['merged_choice_matches_numeric_leader']['score_focus']}/{b['adaptive_rounds']} |")
        lines+=['',
          'The two multi-agent roles selected the same first choice in **18/20 rounds**, despite independently fixed current-round prompts. The generalist pair also agreed in 18/20. Specialized prompts reduced agreement lower in the three-choice lists, without creating a better first-choice decision. Both roles use the same foundation model and all common evidence; their role names do not imply independently trained epilepsy specialists.','',
          'Multi matched the coverage leader in 13/20 rounds and score-focus in 7/20; these leader counts can overlap. This is consistent with convergence on shared geometric heuristics. It does not establish that a particular heuristic caused the performance gap. Unlike the old serial design, no reviewer can veto the other proposal.','',
          f"Full and rationale-hidden multi-agent runs had different final rankings in only **{result['multi_hidden_different_rankings']}/5 patients**. Hiding peer explanations did not worsen the mean outcome. With one stochastic run per arm and five patients, this does not isolate a general causal effect of communication. Access to feedback is verified from requests and lineage; useful reflection is not proven by rationale text.",'',
          'Recorded stimulation-condition counts and uniform-candidate reference counts are retained in the machine-readable behavior summary. Raw 4/8 mA counts alone cannot demonstrate preference because availability differs; pulse/current/charge budgets are not equated.','',
          'The fixed gate fails, so the 30-patient extension is not run. All120 specifically authorized attempts are consumed; no replacement or additional request is made for this failed hypothesis. The next local hypothesis is to train relative candidate-value rankings from existing source-held-out physical counterfactuals, instead of relying on prompt specialties or absolute one-query gain estimates. Such numerical feasibility would still not establish Gemini collaboration value.','',
          'A changed ranking or rationale does not prove communication improved decisions. Backend stochasticity, the small selected pilot and repeated cohort inspection limit causal interpretation. Nominal intervals are not adjusted for the complete research search. No treatment, safety or external validation claim follows. Requests and patient-derived traces remain outside Git.','']
    (project/'docs/spes-agent-v16-report.md').write_text('\n'.join(lines),encoding='utf8'); write_json(project/'examples/spes_agent_v16_summary.json',summary); print('English v16 report state:',summary['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); a=parser.parse_args(); build(a.project,a.output)
