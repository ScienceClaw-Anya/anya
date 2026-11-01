"""File-presence eligibility only. Never opens held-out recordings/parameters."""
import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'data/metadata/vec-inventory.json').read_text())
names={r['name'] for r in records}
split=json.loads((ROOT/'manifests/split-v1.json').read_text())
rows=[]
for subject in sorted(split['development']+split['held_out_reserved']):
    p=f'VirtualEpilepticCohort/derivatives/tvb/{subject}'
    b=f'VirtualEpilepticCohort/{subject}'
    required={
        'connectome':f'{p}/struct/{subject}_connectome.zip',
        'gain':f'{p}/struct/{subject}_gain.txt',
        'electrodes':f'{b}/{subject}_electrodes.tsv',
        'clinical_generator':f'{p}/ses-01/clinicalhypothesis/parameters/{subject}_epileptor_parameters_run-01.tsv',
        'clinical_simulator':f'{p}/ses-01/clinicalhypothesis/parameters/{subject}_simulator_parameters_run-01.tsv',
        'clinical_source':f'{p}/ses-01/clinicalhypothesis/{subject}_simulated_source_timeseries_run-01.npz',
        'clinical_sensors':f'{b}/ses-01/ieeg/{subject}_ses-01_task-simulatedseizure_acq-clinicalhypothesis_run-01_ieeg.eeg',
        'clinical_sensor_header':f'{b}/ses-01/ieeg/{subject}_ses-01_task-simulatedseizure_acq-clinicalhypothesis_run-01_ieeg.vhdr',
    }
    row={'subject':subject,'split':'development' if subject in split['development'] else 'held_out_reserved'}
    row.update({k:int(v in names) for k,v in required.items()})
    row['required_files_present']=int(all(row[k] for k in required))
    rows.append(row)
target=ROOT/'manifests/vec-file-eligibility.tsv'
with target.open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(json.dumps({'subjects':len(rows),'required_files_present':sum(r['required_files_present'] for r in rows),'limitation':'File presence only; contents, mapping, units and usable dynamics unverified for held-out patients.'}))
