# Instructions for this independent research branch

Work only on `codex/personalized-intervention-research` in this dedicated checkout.
This is an independent root history inside `ScienceClaw-Anya/anya`, not the
ongoing diagnostic/SPES branch. Never switch or modify another checkout, merge
into main, rewrite main, change the default branch, or stop an existing process
to free resources. Do not open an integration PR unless the user later asks.

The user explicitly requires experiments on the existing `scienceclaw-gcp`
host, not the local machine. Run simulations, numerical tests and experimental
metric replay only in `/home/devstar3022/anya-intervention-research`. This remote
directory is an isolated execution mirror, not another existing checkout; its
`outputs/source-commit.txt` identifies the source snapshot. Keep local work to
editing, metadata inspection, provenance, reporting and Git. Do not provision
new cloud resources or change other jobs. Default to one worker/thread at nice
priority 10; preserve the bounded pilot limits and record any new run schedule.

Read README.md, PROTOCOL.md, RESOURCES.md, EXECUTION.md and STATUS.json before
research work. The mode is user-directed research execution. These documents
record hypotheses and plans, not evidence that the hypotheses are true.

Keep raw data, clinical annotations, large arrays, model weights, environments,
credentials and runtime output outside Git. Track only permitted compact
manifests, code, summaries and reports. Use explicit file lists when staging.
Inspect the staged diff and avoid accidental publication of patient data or keys.
Use a local environment and bounded resources; public metadata inspection and
ordinary reversible setup can proceed under the user's existing authorization.

Separate simulated truth, observed recordings, clinical labels, delivered actions
and outcomes. Never fabricate matched patients, missing labels, parcel volumes or
clinical responses. Preserve source identities and anatomical/physical units.
Tests of code and replay of metrics are not biological or clinical validation.

Freeze formal evaluation choices before held-out outcomes, split by patient, and
isolate evaluator-only parameters. Any deviation or reused test set is recorded
as a versioned amendment or exploratory analysis. Preserve failures and negative
results. Do not equate more agents, simulations, contacts or seizures with more
independent patients or scientific improvement.

Tests should verify scientific invariants and leakage/budget/mapping boundaries,
not mirror implementation. Benchmark the proposed method against simple numerical
baselines before adding language-agent orchestration. Read upstream licenses and
retain required attribution when adapting code. Do not run downloaded code until
its relevant execution paths and dependencies have been inspected.

Commit and push only this branch with an explicit destination. Never force-push
or push all branches/tags. Other branches and the main experimental track are
read-only references. The idea-exploration chat and execution chat coordinate
through committed reports; user authorization is needed for any new messaging
workflow or external publication beyond these branch updates.
