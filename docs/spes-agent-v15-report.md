# Sparse pretrained readouts: ranking gain not established

The fixed context readout scored **0.734324** mean patient SOZ AUROC on the same eight random0 physical groups, versus original HAT **0.740463**. Gain **-0.006139**, paired 95% interval **[-0.046473408707469716, 0.035096462110097836]**. The next-tool gate fails. These readouts are not adopted as improved localization scorers.

| Groups | Scorer | Mean patient AUROC | Gain over original HAT | Paired 95% interval | Mean patient Brier |
|---:|---|---:|---:|---|---:|
| 4 | hat | 0.705031 | +0.000000 | [0.0, 0.0] | 0.604688 |
| 4 | constant | 0.500000 | -0.205031 | [-0.2560376897834946, -0.1511297597060985] | 0.115629 |
| 4 | linear | 0.725147 | +0.020117 | [-0.029267214197639484, 0.07401318100119578] | 0.111095 |
| 4 | context | 0.698035 | -0.006996 | [-0.05306698818154918, 0.038768378986028564] | 0.114154 |
| 8 | hat | 0.740463 | +0.000000 | [0.0, 0.0] | 0.600342 |
| 8 | constant | 0.500000 | -0.240463 | [-0.28756216186391953, -0.1888088602727886] | 0.115629 |
| 8 | linear | 0.743970 | +0.003507 | [-0.03246264786370757, 0.04293321355102729] | 0.111570 |
| 8 | context | 0.734324 | -0.006139 | [-0.046473408707469716, 0.035096462110097836] | 0.115533 |

The simpler linear readout has a small positive eight-group mean difference, but its interval crosses zero and it remains secondary. The four-group linear result is also uncertain. Neither replaces the primary. Better absolute output scale is distinct from better within-patient ranking; the development-prevalence constant is included as a proper Brier comparator.

## Adaptation and patient separation

Each of five pretrained outer HAT encoders remains fixed. Its own two separately fitted readouts use only its 21 original fitting patients, with three seeded physical paths at four, six and eight response groups. The seven original early-stop patients and seven outer test patients never enter readout fitting. All ten fixed C=0.1 logistic fits are frozen before new outer outcomes. Patient weights balance total contribution while preserving natural class prevalence.

The 16-dimensional features are extracted immediately before the original classifier; their linear reconstruction is checked against original HAT outputs. The context readout adds count and missingness information and representation/count interactions. Each encoder has its own readout coordinate system. Latent dimensions from independently trained neural models are never pooled into one classifier. These fitting features are in-sample for their frozen encoder, explicitly not out-of-fold neural features.

There are **105** fitting banks and **57,159** correlated contact/state rows. The audit independently reconstructs physical availability and context features, checks all lineage hashes and patient exclusion, refits all ten readouts, and replays **280** held-out scorer/state outcomes. Five representative source encoder states were re-inferred to verify feature capture. Remaining neural features were checked from stored outputs, not all re-inferred. Original HAT outputs reproduce the frozen v12 physical-input reference.

## Implication for further collaboration research

Replacing the last readout alone does not establish better localization. This does not rule out adapting the encoder, obtaining additional independent data, or a different acquisition strategy. No new Gemini call was used. A new physically valid language-agent pilot must use a separately frozen protocol and common tools, evidence, response groups, API attempts, feedback cadence and final scorer; it must not adopt failed v12-v15 predictors as established improvements.

All computation ran locally. No new neural fit or cloud resource was required. Repeated inspection of this 35-patient cohort makes these exploratory outcomes; nominal bootstrap intervals are not adjusted for the entire research search. Eight response groups do not equate pulse/charge/time budgets. SOZ ranking does not establish seizure-control benefit.
