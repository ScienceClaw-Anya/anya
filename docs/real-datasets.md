# Real-data integration and resource registry

Checked on 2026-10-31. This is a focused resource check for the diagnostic-selection
prototype, not an exhaustive literature review. The supplied
`More possible datasets.txt` was a starting list; provider documentation determines
the actual access and available measurements. Do not assume every resource contains
stimulation responses, connectomes or a clinical localization gold standard.

## Implemented real-recording checks

| Resource | What is executed | Reference and limitation |
|---|---|---|
| [Multicenter iEEG / ds003029](https://github.com/OpenNeuroDatasets/ds003029) | One seizure from sub-jh101; preictal versus first-five-second ictal contact ranking | Clinical SOZ contacts from the public clinical workbook, with bad-contact exclusions. Passive contact ranking; no counterfactual stimulation or full clinical EZ accuracy. |
| [PRIOS / ds004370](https://github.com/OpenNeuroDatasets/ds004370) | Five recorded pulses at each of two distinct anatomical stimulation pairs from sub-PRIOS01 | Evoked RMS normalized by baseline standard deviation. No independent SOZ reference was verified for this example, so localization accuracy is unavailable. |

The provider READMEs describe clinical epilepsy recordings and SPES recordings,
respectively. Both dataset descriptions state CC0. Small metadata and annotation
assets use pinned Git revisions and their published annex checksums (MD5 for some
older assets, SHA-256 for others); an additional local SHA-256 is recorded.
Signals are read by exact HTTP byte ranges from the public
[OpenNeuro AWS bucket](https://registry.opendata.aws/openneuro/), with ETag locking.
Per-range checksums do not claim verification of a full recording's checksum.

For ds003029, contact names are joined case-insensitively, numeric ranges are
expanded, literal bipolar names are preserved, missing SOZ mappings cause an error,
and both BIDS and clinical bad-contact annotations are excluded. The fixed scoring
rule uses variance ratios and clinician-provided onset timing; contact labels and
postoperative outcomes do not enter the ranking rule. The source recording's header
and channel ordering are checked before reading samples.

For PRIOS, polarity reversal is not counted as a new anatomical site. Annotated
artifact intervals, bad/non-iEEG channels, stimulated contacts and the first 10 ms
after a pulse are excluded. Header amplitude units are omitted and channel cutoff
columns appear reversed, so the prototype reports dimensionless normalized responses
and does not infer or apply a filter from those cutoff values. This is a small
extraction check, not a validated N1 detector or active-localization benchmark.

## Resources from the supplied list

| Resource | Suitable contribution | Current status and constraint |
|---|---|---|
| [IDEAS](https://onlinelibrary.wiley.com/doi/10.1111/epi.18192) and [IDEAS II](https://onlinelibrary.wiley.com/doi/10.1002/epi.70186) | MRI, resection masks, structural connectomes and surgical outcomes; a useful next anatomical scaffold | Provider records checked. Raw releases are OpenNeuro ds005602/ds007401; derived assets need their own license check. Resection is not an exact EZ label, and these are not recorded stimulation-response banks. Not downloaded in this run. |
| [Multicenter iEEG](https://github.com/OpenNeuroDatasets/ds003029) | Seizure timing, contact-level SOZ and passive external validation | Implemented small check above. The public release omits the Cleveland Clinic center, which requires a separate DUA. No such agreement was accepted. |
| [Harvard EEG v4.1](https://bdsp.io/content/harvard-eeg-db/4.1/) | Large EEG feature-learning or seizure-detection development | Provider page states credentialed access. No login, agreement or download was attempted. It does not by itself provide the within-patient SEEG stimulation/gain map needed here. |
| [Epi25](https://epi25.broadinstitute.org/) | Gene/variant context for later biological priors | Public summary browser; individual-level sequencing access is controlled. Genetic associations do not directly identify an electrode stimulation site. Not part of the current numerical benchmark. |
| [MELD](https://meldproject.github.io/studies/FocalEpilepsies/) | Structural lesion detection and anatomical priors | Cohort and access depend on the specific MELD study; public code is not equivalent to unrestricted participant-level MRI access. Lesion masks and clinical EZ are different labels. No participant-level data downloaded. |
| [Ontario TLE / EpLink](https://portal.conp.ca/dataset?id=projects%2Fimaging_in_TLE_DR011) | Multimodal imaging and surgical-outcome research | Provider lists 118 patients, 72 with surgical outcomes, and access approval through Brain-CODE. No agreement or access request was submitted. Limited EEG availability does not imply recorded stimulation responses. |

## Why these scores stay separate

The virtual benchmark evaluates 162 region-level generator-reference labels.
The multicenter example evaluates recorded contacts against clinician SOZ annotations.
PRIOS currently has no verified localization reference in this adapter. IDEAS
resection masks and outcomes answer another question. Pooling these into a single
"accuracy" would mix different tasks, anatomical units and reference definitions.

The next real active-selection study needs verified stimulation identifiers,
repeated responses, compatible anatomical mapping and an independent localization
reference in the same patients. Logged stimulation data can support replay only
over recorded actions; causal claims about unseen stimulation require additional
modeling assumptions or prospective data.
