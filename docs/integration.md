# Integration contract and scope

The framework pin in `config/upstream-lock.json` matches the ScienceClaw version
used by the team's source guide. The project imports the real upstream registry,
executor, artifact and Gemini client; they are not stand-in classes.

## Execution path

1. Native SkillRegistry discovers the three project skills.
2. Native SkillExecutor obtains a public candidate catalog and attributed review cards.
3. Four independent role profiles select experiments, using either explicit deterministic
   demo policies or the native Gemini client. Candidate count exceeds the budget.
4. A deterministic adapter validates IDs, enforces budgets and passes the selected
   contact pair to the native executor's simulation tool.
5. Native artifacts record catalog/evidence -> proposal -> measurement dependencies.
   The next prompt consumes the visible measurement summaries.
6. Artifact hashes and parent order are checked before a success report is written.

This is a controlled case workflow, not a claim of unrestricted emergent discovery.
The upstream ArtifactReactor, publication service, remote execution, heartbeat,
cross-run learning and model-ensemble optimization remain separate future work.

## Local persistence policy

Registry and artifact-store subclasses redirect state into `outputs/scienceclaw`.
The store uses native artifact creation/hash/lookup and native global-index writing,
with a local-only save hook that omits optional Infinite publication and PQC key
generation. No global HOME variables or user ScienceClaw profiles are replaced.

Simulation arrays stay in numerical run folders. Agent prompts receive only approved
SEEG summaries, geometry/network scores and attributed evidence. The numerical
result record and report may contain model diagnostic readouts for offline inspection;
those readouts are not passed into the agent decision prompt.

## Review alignment

The team's architecture brief distinguishes planning, tool execution, artifacts,
need matching and publishing. This integration preserves those boundaries. The
longer implementation plan describes a multi-week research system; it is not a
claim that every proposed endpoint or validation gate already exists.

Implemented: explicit tool contract, bounded execution, input-ID validation,
artifact lineage, fixed upstream versions, attributed team-review context,
separate deterministic and real Gemini modes, local provenance checks.

Pending: independent patient model fitting, credible model/hypothesis ensembles,
published-method parity benchmarks, prospective stimulation-control validation,
held-out historical outcomes, transactional multi-worker claim leases and independent
release review. Software tests and one successful agent loop do not validate these.

## Infinite access gateway

An optional read-only gateway check authenticates only against the configured
HTTPS origin, with redirect following disabled. The gateway session is kept in an
ignored private file. A successful gateway check is not proof of agent API identity.
`infinite-draft` prepares a reviewable local post from a real Gemini report and
performs no network write. Registration, new API-key creation and publication are
separate actions; none are enabled by the numerical collective's save hook.
