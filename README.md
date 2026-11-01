# Personalized intervention planning beyond SOZ localization

This is an independent research track in `ScienceClaw-Anya/anya`, created on
2026-11-01 from the user's research discussion. It studies virtual intervention
selection and retrospective outcome prediction without recruiting new patients.

**Status: bounded development pilot completed on the existing GCP host. No
personalization advantage, equal-information intervention advantage or clinical
prediction improvement has been established.**

Read the [first pilot report](reports/2026-11-01-pilot.md) for results, limitations,
resource use and the remaining scientific gates. All numerical experiments and
tests run on GCP; this checkout holds code, provenance and compact reports.

## Research question

Does a patient-specific network model select virtual resections that reduce
simulated seizure activity more effectively than SOZ-guided targeting at the
same intervention budget? Do model-derived predictions also improve prediction
of outcomes from surgeries actually performed?

The first question is the primary experiment. The second is an independent
retrospective validation objective, conditional on acquiring compatible records.
The initial intervention is virtual resection. Disconnection and stimulation
are later extensions with their own mechanisms, cost measures and protocols.

## What changes relative to localization

Recordings and anatomy lead to a localization/network estimate. A planner then
compares candidate interventions, predicts their consequences and selects an
action. Localization and planning are evaluated separately:

| Comparison | Contribution tested |
|---|---|
| Same localization estimate, SOZ-guided vs network-based planner | Value of intervention modeling beyond localization |
| Same planner, different localization estimates | Whether better localization improves intervention outcomes |
| Same localization and planner, individual vs generic connectivity | Value of personalization |

Clinical SOZ contact annotations, model-generating epileptogenicity parameters,
resection masks and postsurgical outcomes are distinct references. A resection
mask describes an action; it is not ground-truth SOZ. VEC's known parameters are
simulation truth, not verified biological truth.

## Quantitative objectives

- **Primary:** greater reduction in simulated seizure burden at a common
  intervention budget, measured on held-out virtual patients against SOZ-guided
  targeting. Measure initiation and propagation separately.
- **Secondary:** intervention efficiency, finite-candidate selection regret,
  robustness to uncertainty and the value of individual connectivity.
- **Retrospective objective:** improved held-out prediction and calibration for
  a fixed postsurgical outcome using records of the operation actually delivered.

A simulated benefit supports a claim about the tested models. Prediction of
observed surgical outcomes supports an empirical prediction claim. Neither alone
verifies outcomes of alternative operations never performed.

## Start here

1. [Protocol](PROTOCOL.md): endpoints, baselines, leakage controls and freeze gates.
2. [Resources](RESOURCES.md): datasets, models, provenance and access checks.
3. [Execution plan](EXECUTION.md): bounded first run and deliverables.
4. [Status](STATUS.json): machine-readable completion state.
5. [Agent instructions](AGENTS.md): isolation and scientific reporting rules.

## Branch and checkout isolation

- Remote: `git@github.com:ScienceClaw-Anya/anya.git`.
- Branch: `codex/personalized-intervention-research`.
- This branch starts from its own root commit and has no parent from `main`.
- Its checkout is separate from all existing Anya checkouts and running jobs.
- Ongoing diagnostic/SPES work is a read-only reference. The inspected reference
  snapshot was `d666bd356d944fb363dbf60537b5682e44b269a7`; it is not this branch's parent.
- Do not merge this branch into `main`, open an integration PR, change the default
  branch, or edit existing experiment records as part of this track.

The idea-exploration chat remains the place for research direction. A separate
execution chat owns implementation and bounded experiments in this checkout.
Record execution progress in this branch so discussions do not depend on chat
memory. New ideas require a versioned protocol amendment before they affect a run.

## Relationship to Anya

The inspected Anya work emphasizes SOZ ranking and diagnostic measurement
selection on retrospective SPES data, with a smaller virtual-patient experiment.
Its estimates may become inputs to our planner after anatomical/unit mapping is
verified. We do not interpret its diagnostic scores as treatment efficacy.
Its virtual sub-003 case is prior development exposure and must not be advertised
as a pristine held-out patient here.

This track does not assume language agents outperform numerical methods. First
establish the numerical benchmark. Any subsequent agent study must use equal
information and resource budgets and evaluate intervention outcomes directly.
