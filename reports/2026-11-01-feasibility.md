# Feasibility audit — development, not a scientific result

The independent branch was checked at root 8a3f855655fc83cc498907c309c8d1ac64f019c6.
No intervention runs preceded `manifests/split-v1.json`. Six subjects are reserved
for development (including previously exposed sub-003), 24 for untouched evaluation.
Asset eligibility remains to be verified; this is a split reservation, not a frozen
confirmatory protocol. Listing: 2,290 files, 4,674,138,585 bytes; do not fetch all.

VEC data DOI: https://doi.org/10.25493/XWG2-S8X (v1.0; CC BY 4.0).
Authors: Dollomaja, Wang and Jirsa (2024). Paper:
https://doi.org/10.1371/journal.pcbi.1012911 .
Generation code inspected at a99c88354015f4c961d49a92c076ed0b675c740b:
https://github.com/BalanceKey/virtual_epilepsy_patient_cohort . No license file was
present in its tree. Upstream scripts were read, not executed or redistributed.
They contain author-specific paths and empirical inputs unavailable in the release.
We use TVB's GPL-3.0-or-later equations and retain attribution for adapted code.

Sub-003 clinicalhypothesis run-01 assets acquired: 28,436,577 bytes including
three reference PNGs inadvertently included by the first acquisition selector;
future selectors must explicitly exclude `/img/`. Every saved asset is hashed in
`manifests/acquisition.json`; clinical assets and raw arrays remain ignored.
Its connectome includes 162 parcel volumes and areas. Their physical units and
reconstruction must still be verified: a filename alone does not establish mm³.
The released parameter values differ from current upstream script defaults; use
released parameters. Source NPZ has 4,800 by 162 values. The BrainVision header
and file length correspond to 4,600 by 185 values, requiring alignment verification.
The coordinate JSON says T1w/vox, but no affine is yet verified. Do not infer mm
or a contact's parcel purely from those coordinates.

## Compute placement amendment

Before any local experiment, the user explicitly required GCP execution. All
simulation, controls and experimental tests will run in the isolated directory
`/home/devstar3022/anya-intervention-research` on existing `scienceclaw-gcp`.
Local work is editing, provenance, metadata inspection and Git only. The previously
created local `.venv` has not been used for simulations. No new cloud provision.
GCP at 2026-11-01 16:18 UTC: 32 CPUs, 125 GiB RAM, 122 GiB available, 175 GiB
free disk; load 12.51 with existing vep_v3 jobs. Use one worker, one numerical
thread, nice priority 10. Do not stop or alter existing jobs. Retain the initial
60-minute experiment wall-time and 2-GiB dataset-download ceilings.

## IDEAS audit boundary

Primary sources: https://www.cnnp-lab.com/ideas-data ,
https://doi.org/10.1111/epi.18192 , https://doi.org/10.1002/epi.70186 .
The authors report IDEAS I: 442 surgical patients, 433 resection masks; IDEAS II:
216 patients with raw DWI and 191 with processed connectomes. Identifiers are
explicitly aligned across the two releases. These are published counts, not
locally verified joint eligibility. The author page links clinical metadata,
resected-region percentages and atlas-ordered connectomes. It does not establish
matched SEEG recordings or contact SOZ labels for those imaging patients.
The linked clinical table is 39.31 kB plus a metadata PDF and control table, with
CC0 displayed on its page. HTTP downloads returned empty 202 WAF challenges;
the browser download was then denied by the user's permission control. No
workaround will be attempted. No clinical table was acquired or joined, no
12-month endpoint completeness was computed, and no retrospective prediction
was run. The concrete dependency is authorized access to the clinical dictionary
and table plus subject-level processed-connectome and resection inventories.

### Additional source checks

IDEAS I Table 1 reports 427 patients with 12-month ILAE outcomes. Its narrative
analysis uses 422 after excluding five multilobar resections. These denominators
must not be conflated. IDEAS II's published 191 processed-network patients put an
upper bound of 191 on a connectome-based retrospective cohort before further
exclusions; this is an inference from published counts, not a joined cohort.
A candidate endpoint is ILAE 1 at 12 months, conditional on verifying dictionary
coding. ILAE 1–2 measures freedom from disabling seizures and is a different
endpoint. Postoperative pathology must not enter a strictly preoperative model.

The VEC metadata inventory shows all eight required file types for all 30
clinicalhypothesis run-01 cases (`manifests/vec-file-eligibility.tsv`). This checks
names only and does not open reserved subjects' parameters or recordings.

A separate pipeline source check found that `get_region_volumes` multiplies
label counts by the product of NIfTI voxel dimensions, then writes `volumes.txt`:
https://github.com/ins-amu/tvb-pipeline/blob/70d35cea1828e2919b5fe6a696c0860492ef6aa2/util/create_tvb_dataset.py#L349 .
This supports a likely physical-volume interpretation, but the VEC release does
not establish the generating pipeline revision or the individual NIfTI header
units. The pilot remains a parcel-count proxy comparison.
