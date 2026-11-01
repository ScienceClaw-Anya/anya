"""English action-grain correction from inspected source metadata."""
import argparse
from pathlib import Path
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json


def build(project,output):
    project,output=Path(project),Path(output);a=read_json(output/'alignment_audit.json');s=read_json(output/'two_query_smoke.json');v=read_json(output/'remap_invariance.json');stop=read_json(project/'outputs/spes_agent_v9/interruption.json')
    if a['mapped_patients']!=35 or v['status']!='verified':raise ValueError('Full audit and actual-model verification required')
    fraction=a['unambiguous_cells']/a['valid_cells']
    text=f'''# Physical action-grain correction and verified replacement

The processed arrays contain compact response sets for individual recording
contacts. **The same array column is not the same physical stimulation pair
across contacts.** This was an implementation error in our adaptation from
contact classification to active selection. The shared-action assumption should
have been checked before the agent experiments.

The audit confirms mixed physical pairs within columns in **35/35 patients**.
Historical v4-v10 subset-selection results remain preserved as **array-column
ablation**. Their interpretation as eight physical stimulations, anatomical
target selection or a stimulation protocol is withdrawn. They cannot establish
whether multi-agent collaboration helps real stimulation selection.

## Evidence and scope

The check matches processed recording coordinates to the public electrode TSV,
then matches each retained response distance to the midpoint of actual recorded
stimulation pairs. The source coordinates match; the provider declares
**fsaverage, millimetres**, not an interchangeable MNI152 registration. Only
geometry, anatomical labels and stimulation fields are read. SOZ, resection,
outcomes and clinical notes are excluded.

Across **{a['valid_cells']:,} recording-contact / compact-column cells**,
**{a['unambiguous_cells']:,} ({fraction:.4%})** have one unique anatomical-pair
match within 0.0001 mm. **{a['ambiguous_cells']} ambiguous cells are excluded**.
There are {a['duplicate_recording_pair_assignments']} duplicate assignments.
All 35 mapped patients have at least 41 verified single-condition pairs, and the
largest pool has 93. No pair had multiple recorded parameter tuples in this
metadata check; this does not validate prospective clinical safety.

These counts concern the **first twenty trials actually used by all current
neural fits and evaluations**. An initial full-forty-trial probe found changing
geometry at fixed row/column positions; it is retained as an unresolved probe.
A row-only map must not be extended to those extra trials. The replacement
adapter rejects changing geometry within selected trials.

The full-response classification task does not require a shared physical
action axis. Its patient-separated neural results and static model comparisons
are unaffected by this action-grain finding. It does affect the physical
interpretation of partial-column acquisition and tools trained on that task.

## Stopped work and retained evidence

The v9 outer evaluation stopped after **{stop['completed_numeric_trajectories']}
completed trajectories**, leaving **{stop['incomplete_or_unstarted']} canceled
or unfinished**. All recorded choices, observations, metrics, frozen protocols
and the stop lock remain. No partial-cohort winner is reported. Its numerical
tools are not adopted for physical selection.

The ten v10 contact-model fits and training banks are retained, but outer
evaluation was blocked before execution. They used column-ablation prefixes
and require new physical-query training states before active use. No new Gemini
request was made in either version. Historical scores are not silently rewritten.
Interpretation notices are added to the old English reports and READMEs.

## Verified replacement

`PhysicalAcquisition` queries a named anatomical pair and gathers its verified
response from the appropriate, potentially different, compact column at each
recording contact. Unresolved cells remain zero/absent. Only those actual
responses enter model inference. Hidden responses and labels stay private;
the scoring guard requires eight complete physical queries.

The minimum now runs on patient {s['case']['subject']}: two distinct pairs,
{s['observations'][0]['contacts']} and {s['observations'][1]['contacts']}.
The second query needs different compact columns across recording contacts,
demonstrating why the old global-column action was incorrect. This is an
extraction/inference check, with no localization-effect claim.

An actual held-out HAT check compares physically aligned responses with the
identical retained response set in original compact layout. Maximum absolute
score difference is **{v['maximum_absolute_HAT_score_difference']:.9g}**
(tolerance {v['tolerance']}). Reindexing therefore preserves this model's
full-response calculation for the tested patient. Full predictions are not
exposed to agents in the active task.

## Next valid experiment

Freeze new physical-pair numerical controls before an agent comparison. Rebuild
development-patient query-value or contact-posterior training states using the
verified maps; old column-utility models are unsuitable. The agent interface can
now share real pair identities and whitelisted anatomy, in addition to measured
waveforms and trained model scores. Independent proposals, feedback timing and
total model/measurement budgets must match single-agent controls; include direct
numerical selection with the same tools.

Model/version changes or more debate do not repair an invalid action definition.
No multi-agent superiority or treatment benefit has been established. All new
evaluation on this repeatedly inspected cohort remains exploratory; a stronger
claim needs independent confirmation.

Source context: [public processed author arrays and model code](https://github.com/norrisjamie23/SOZ_localisation_SPES_HAT)
and [OpenNeuro ds004080 v1.2.2](https://openneuro.org/datasets/ds004080/versions/1.2.2).
Per-file metadata, array, action-map and source-code hashes are retained locally
under `outputs/spes_physical/`. Raw data and private credentials stay outside Git.
All mapping/model checks ran locally on the workstation; no cloud resource or
model API request was used.
'''
    (project/'docs/spes-physical-action-audit.md').write_text(text,encoding='utf8')
    write_json(project/'examples/spes_agent_v9_summary.json',{'status':'stopped_invalid_physical_action_mapping','completed_trajectories':stop['completed_numeric_trajectories'],'remaining_canceled_or_unfinished':stop['incomplete_or_unstarted'],'new_API_calls':0,'physical_interpretation':'withdrawn; column-ablation only','multi_agent_superiority_established':False})
    write_json(project/'examples/spes_agent_v10_summary.json',{'status':'stopped_before_outer_evaluation_invalid_action_assumption','fit_models':10,'outer_evaluations':0,'new_API_calls':0,'physical_interpretation':'column-ablation prefix training; not adopted for physical queries'})
    print('English physical-action audit and interruption summaries saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.project,a.output)
