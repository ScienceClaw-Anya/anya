# Resource map and acquisition audit

Reviewed 2026-11-01. Availability in a paper is not a verified local download.
No dataset has yet been acquired by this new track. Pin revisions and record
access/license terms, hashes, anatomical conventions and usable cases when acquired.

| Resource | Inputs and useful role | Current evidence and limitations |
|---|---|---|
| [VEC paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012911), [EBRAINS data](https://doi.org/10.25493/XWG2-S8X), [generation code](https://github.com/BalanceKey/virtual_epilepsy_patient_cohort) | 30 patient-derived virtual cases; connectivity, electrodes, gain matrices, known generating parameters, simulated spontaneous/stimulated seizures and interictal activity. Main simulation benchmark. | Known truth is generator parameters. Audit parcel volumes and modality coverage. Anya's sub-003 experiment is development exposure. Multiple seizures/EZ hypotheses from one person are not independent patients. |
| [TVB](https://www.thevirtualbrain.org/), Epileptor via VEC | Brain-network dynamics and source-to-sensor simulation. Reproduce baseline activity and implement explicitly defined virtual interventions. | A simulator is not a dataset or a therapeutic efficacy result. Pin the version compatible with upstream code. Changing excitability is not automatically equivalent to surgery. |
| [BVEP](https://github.com/ins-amu/BVEP) | Bayesian inversion of individualized seizure-network models; uncertainty and hidden-parameter inference. | Candidate fitting framework, not integrated yet. Use a tractable development configuration before a cohort-scale inference run. |
| [IDEAS I paper](https://arxiv.org/abs/2406.06731), [ds005602](https://openneuro.org/datasets/ds005602); [IDEAS II paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13285235/), [ds007401](https://openneuro.org/datasets/ds007401) | MRI, actual resections and postoperative outcomes; diffusion-derived connectivity. Main candidate for retrospective validation. | I: 442 surgical patients. II: DWI from 216 patients/98 controls, released processed networks for 191 patients/97 controls. Eligible overlap with outcomes is not yet audited. No assumption of complete matched SEEG/SOZ. Raw and derived assets need their own provenance/terms. |
| [HUP ds004100](https://github.com/OpenNeuroDatasets/ds004100), [ds003029](https://github.com/OpenNeuroDatasets/ds003029), [ds003876](https://github.com/OpenNeuroDatasets/ds003876) | Ictal/interictal iEEG, clinical localization and resection/outcome metadata; dynamics and retrospective validation. | Earlier workspace exercised HUP060; Anya exercised one ds003029 seizure. Check patient overlap, contact/region mapping and coordinate frames. Do not assume structural connectivity exists for each case. |
| [SPES ds004080](https://github.com/OpenNeuroDatasets/ds004080), [PRIOS ds004370](https://github.com/OpenNeuroDatasets/ds004370) | Recorded evoked responses; diagnostic SOZ inference, response prediction and acquisition policy evaluation. | Anya's main work uses selected 35-patient processed SPES cohorts. Their versions are not independent datasets. Only recorded physical actions can be replayed. PRIOS example has no independently verified SOZ reference in the inspected adapter. Neither is automatically a therapeutic stimulation outcome bank. |
| [Virtual resection predicts surgical outcome](https://pmc.ncbi.nlm.nih.gov/articles/PMC6885672/), [EpiVR code](https://github.com/ieeg-portal/EpiVR) | Functional-network virtual-resection comparator and retrospective evaluation precedent. | 28-patient study; audit executable assets, access and HUP overlap. Reported literature scores are not our replication results. |
| [Controlling seizure propagation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006805) | Network-disconnection modeling precedent and comparison ideas. | Disconnection costs and assumptions differ from resection. Propagation control is distinct from eliminating onset activity. |
| [VBT_INS_Stimulation](https://github.com/HuifangWang/VBT_INS_Stimulation), [paper](https://doi.org/10.1038/s43588-025-00841-6) | High-resolution virtual brains, SEEG/TI stimulation, Bayesian EZN inference; named in the user's project document. | Demonstrated stimulation focus is diagnosis via induced seizures. Suppression requires a justified additional mechanism. Author-reported Code Ocean assets: [capsule 1](https://doi.org/10.24433/CO.3316132.v1), [capsule 2](https://doi.org/10.24433/CO.2143670.v1); this track has not downloaded them. |
| [SimNIBS](https://simnibs.github.io/simnibs/build/html/) | Electric-field simulation and montage optimization, including temporal interference; part of Wang's toolchain. | Target/off-target field delivery can be evaluated separately. Field strength is not equivalent to seizure suppression or clinical safety. Later extension. |

## Lower-priority supporting resources

The earlier workspace integrated HCP/ENIGMA normative connectivity, Allen Human
Brain Atlas regional expression and Open Targets epilepsy gene evidence, and
cataloged ClinVar. These are population-reference biological priors, not a matched
patient's multimodal measurements. They are optional for the first benchmark.

Anya's [resource registry](https://github.com/ScienceClaw-Anya/anya/blob/d666bd356d944fb363dbf60537b5682e44b269a7/docs/real-datasets.md)
also lists MELD, Harvard EEG, Epi25 and Ontario TLE/EpLink. They may support lesion
priors, EEG features, genetic context or external outcome validation, respectively.
Access and usefulness depend on the particular release; prioritize the core
VEC/IDEAS pathway before widening acquisition.

## Required acquisition manifest

For each acquired asset record: provider URL, version/revision, retrieval time,
license/access conditions, size, checksum, local ignored path, patient identifier,
modality, coordinate frame, atlas/region order, units, observed vs simulated status,
label meaning, generating hypothesis, and missingness/exclusions. Unknown fields
stay unknown. Public code availability does not imply permission to redistribute
individual-level clinical data or every derived asset.

The user's HTML plan remains contextual evidence: it motivates surgery/stimulation
planning but is not a measured result or an execution instruction. Its local source
was `/Users/step5/Downloads/epilepsy-scienceclaw-implementation-plan-complete.html`.
