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
- **Multicenter epilepsy iEEG**, OpenNeuro ds003029,
  https://github.com/OpenNeuroDatasets/ds003029. The dataset description states CC0
  and attributes Adam Li and collaborators. Raw recordings and the public clinical
  annotation workbook remain outside Git. Derived contact-level checks cite the
  dataset and do not implement or claim the proprietary neural-fragility algorithm.
- **PRIOS**, OpenNeuro ds004370,
  https://github.com/OpenNeuroDatasets/ds004370. The dataset description states CC0
  and attributes van Blooijs, Blok, Huiskamp and Leijten. Retain the provider's
  requested acknowledgment for *The effect of propofol on local effective brain
  networks*. Raw recordings remain outside Git; this repository contains only a
  small derived extraction check and its public-source provenance.

- **Norris et al. SPES SOZ models**, https://github.com/norrisjamie23/Localising_SOZ_from_SPES, MIT. Pinned commit 90062e8c5fe5593713125eaf5f573faedb739f65 is imported from an independent ignored checkout. Cite Norris et al., *Localising the Seizure Onset Zone from Single-Pulse Electrical Stimulation Responses with a CNN Transformer*, MLHC / PMLR 252 (2024), https://proceedings.mlr.press/v252/norris24a.html. The reconstruction records differences between the paper and public code/data; it does not substitute paper values for executed results.
- **CCEP ECoG dataset across age 4-51**, OpenNeuro ds004080 v1.2.4, CC0. Cite van Blooijs et al., *Developmental trajectory of transmission speed in the human brain*, Nature Neuroscience (2023), DOI 10.1038/s41593-023-01272-0. Raw recordings, clinical electrode tables and derived tensors/checkpoints remain outside Git; complete public recording hashes are checked before processing.

Research outputs include simulations and retrospective public-recording analyses. No clinical efficacy or independently
validated seizure-localization performance is claimed.

- **Norris 2026 multi-trial SPES models**, MIT code,
  https://github.com/norrisjamie23/SOZ_localisation_SPES_HAT, pinned commit
  11f65182a2114440f1425656f91d0ebec51ee726. Kept in an ignored independent
  checkout. Cite the preprint *Localising the epileptogenic zone from single-pulse
  electrical stimulation responses using cross-trial attention*,
  https://doi.org/10.64898/2026.07.27.26358819. Derived arrays originate from
  OpenNeuro ds004080 v1.2.2 according to the author README. Do not conflate this
  processing cohort with the earlier project's ds004080 v1.2.4 reconstruction.
  No raw data or model checkpoints are redistributed in this repository.
