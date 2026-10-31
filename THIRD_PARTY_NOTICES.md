# Third-party components and data

- **ScienceClaw**, https://github.com/lamm-mit/scienceclaw, Apache-2.0.
  The project imports the pinned upstream code from an independently cloned,
  ignored vendor directory. It extends the registry/store paths and local save
  policy without copying the upstream library into this repository.
- **Virtual Epileptic Patient Cohort**, DOI 10.25493/XWG2-S8X, CC BY 4.0.
  Cite Dollomaja et al., *Virtual epilepsy patient cohort: Generation and
  evaluation*, PLOS Computational Biology (2025), DOI 10.1371/journal.pcbi.1012911.
  Downloads require the user's acceptance of the EBRAINS access terms.
- **VEP author code**, https://github.com/BalanceKey/virtual_epilepsy_patient_cohort.
  Kept in an independent ignored checkout, with commit and file checksums. This
  repository does not relicense it or assert an upstream license that is absent.
  The coupling-index correction is implemented in our subclass, not by editing
  the downloaded author source.
- **Team literature review**, https://github.com/ScienceClaw-Anya/epilepsy-virtual-brain.
  The attributed method cards in `research/evidence.json` identify the source
  commit, file hashes and URLs. They are context from the team review, not a claim
  that the integration independently repeated its full literature audit.

Generated results are research simulations. No clinical efficacy or independently
validated seizure-localization performance is claimed.
