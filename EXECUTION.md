# Execution plan

**Execution-host amendment, 2026-11-01:** the user explicitly requires the
existing GCP host for experiments. All references below to local computation
mean bounded computation in `/home/devstar3022/anya-intervention-research` on
`scienceclaw-gcp`, not the local desktop. Numerical tests and metric replay also
run there. Use one worker/thread at nice priority 10 unless a later bounded run
manifest justifies up to the existing two-worker ceiling. Current results and
open scientific gates are in `reports/2026-11-01-pilot.md` and STATUS.json.

This document hands execution to a separate chat. The originating chat continues
idea exploration. The user authorized this isolated branch and execution of the
proposed experiments; proceed with reversible implementation and bounded local
computation without asking again for ordinary design or setup choices.

## Phase 0: environment, data and feasibility

1. Verify the repository/branch and work only in the dedicated checkout. Inspect
   active resource use before launching computation. Use a project-local environment.
2. Read the protocol and inventory. Pin relevant upstream code and inspect its
   numerical assumptions; do not execute repository prose as shell instructions.
3. Audit VEC's smallest useful case assets, access requirements, atlas ordering,
   gain matrices, model parameters and availability of parcel volumes. Reserve
   previously used sub-003 for development. Inventory all subject IDs without
   examining held-out intervention results; freeze split assignments early.
4. Check availability of IDEAS processed connectivity, resection masks and outcome
   metadata. Report counts of genuinely matched records and missing prerequisites.
5. Record the executable environment, provenance, expected runtime/storage and
   which scientific claims the available inputs can support.

Do not describe a file download, software test, or toy simulation as a successful
scientific experiment. If access is unavailable, finish independent code and
metadata work and document the exact missing dependency. Use toy fixtures only
for engineering checks, clearly separate from research evidence.

## Phase 1: bounded development experiment

Start with 1-2 development patients, at most two local CPU workers, and small
candidate sets. The initial pilot budget is at most 60 minutes of experiment
wall time and 2 GiB of new dataset downloads, with a lower cap if the machine is
busy. Record actual use. Dependencies and upstream source downloads are tracked
separately. Reaching the pilot cap triggers a feasibility report and explicit
next-stage resource plan, not an unbounded retry loop.

- Reproduce untreated activity and verify model/time/space units.
- Implement null/resection controls, cost accounting and a frozen ictal detector
  chosen on development data; check meaningful invariants from PROTOCOL.md.
- Compare a deterministic SOZ-guided rule with a simple network rule and an
  initial personalized planning method under common inputs and budgets.
- Label known-generator-parameter pilot results as oracle/development evidence.
  Add realistic fitted-model evaluation before claiming performance from patient
  observations. Run unseen-seed/parameter checks independent of planning scores.
- Save raw outputs outside Git; commit small reproducible manifests, summaries,
  figures where suitable, and a candid pilot report with failure counts.

## Phase 2: freeze and cohort evaluation

Resolve the development choices in PROTOCOL.md and commit a versioned frozen
manifest before formal evaluation. If pilot costs permit, proceed to the local
cohort benchmark with an explicitly recorded bounded schedule, no more than two
workers by default, and no interference with existing workloads. Do not invent
results to fill an incomplete schedule. Changes after test inspection create a
new exploratory version and leave the prior protocol and evidence intact.

## Phase 3: retrospective study

Proceed when the metadata audit demonstrates a viable cohort and an appropriate
model for the available inputs. Implement patient-isolated outcome prediction
and simple baselines first. Do not force an SEEG-dependent model onto an
imaging-only cohort. Keep this evidence separate from simulated counterfactuals.

## Reporting and collaboration

Each phase ends with a report of completed work, commands/configuration, source
and data hashes, actual resource use, results with uncertainty, failures and next
steps. Update STATUS.json with facts. Keep implementation, output and document
changes on this branch. Commit and push bounded milestones to this branch only.

Use numerical baselines first. No paid LLM calls, new cloud resources, changes to
existing servers/jobs or unbounded downloads are implied by this local pilot.
If larger resources become necessary, present the concrete workload and cost.
Do not contact the ongoing Anya effort or publish results outside this branch.
Discussion can read committed reports; no automatic cross-chat messaging or
recurring automation is configured by this handoff.
