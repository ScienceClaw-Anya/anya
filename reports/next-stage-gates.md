# Gates after CPU development v2

See `2026-11-01-cpu-development-v2.md` for completed results. Ordinary bounded
development on the existing GCP CPUs remains authorized; use a new run manifest,
one thread at nice 10, and no new resources or changes to existing jobs. No
experiment is currently running, and no held-out evaluation has been started.

1. **Improve observation prediction.** Sub-010's mapping is verified across three
   recordings. The observation-only linear baseline loses to last-observation
   persistence, and stationary Epileptor covariance fitting still has large
   prediction errors. Fit seizure phase, onset/offset or time-varying envelopes
   and quantify which parameters are identifiable. Keep generators and source
   arrays evaluator-only. Do not optimize interventions with the failed fits.
2. **Resolve sub-003 provenance.** NPZ/TSV source excerpts disagree at matching
   timestamps, and source-to-sensor reconstruction retains 18.6% raw error.
   Alternative gains and affine scaling do not repair it. This is a development
   data-quality exclusion from mechanistic fitting, not evidence that all VEC
   cases are inconsistent. Do not silently replace the release with fitted gains.
3. **Establish informative network effects.** Released settings in both cases
   show small effects of network removal/replacement compared with the sampled
   noise variation. Even synthetic 1000x coupling in sub-003 does not recruit
   additional regions under the detector. Choose a scientifically justified
   propagation model/regime and validate observations; do not tune for a desired
   personalization result. Include a training-only reference and an established
   virtual-resection method with compatible information and compute budgets.
4. **Verify costs and units.** Parcel volumes are present but individual geometry
   and physical units remain unverified. Keep parcel-count/model-time labels until
   resolved, or explicitly freeze a proxy-cost benchmark with restricted claims.
   Exporter amplitude scaling and 1-kHz sample metadata do not establish biological
   amplitude or time calibration.
5. **Freeze before cohort evaluation.** Finalize detector, horizons, baseline
   eligibility, candidate feasibility, cost, practical effect threshold and
   patient-level analysis before reading outcomes for the 24 reserved cases.
   All v1/v2 outcomes and temporal/other-recording checks are development evidence.
6. **Retrospective study separately.** The IDEAS clinical-table download remains
   denied. An authorized table/dictionary and subject inventories are needed to
   verify actual overlap of connectivity, delivered surgery and a consistent
   follow-up outcome. Do not retry the denied download by another route or
   assume imaging records contain SEEG/SOZ.

The next stage should advance fitting and model validity, with explicit failure
criteria, before investing in more intervention search. CPU throughput is
adequate for such bounded development; GPUs are not required for this step.
