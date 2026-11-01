# Sparse physical readouts from pretrained HAT representations

This new numerical study keeps each outer HAT encoder frozen and trains two
small readouts on its own 21 fitting patients, using actual subsets of four,
six and eight verified physical response groups. The original seven early-stop
and seven test patients never enter readout fitting. Three fixed random paths
create nine correlated input states per fitting patient.

The 16-dimensional representation is captured immediately before the original
classifier. Its linear reconstruction must match the original HAT score. Each
encoder has a separate readout; latent axes from differently trained encoders
are never concatenated. These fitting features are in-sample for the encoder,
and are explicitly not called out-of-fold neural predictions.

The context readout adds query count, available-trial fraction and interactions
between representation and query count. Both scaled logistic models use fixed
C=0.1 and equal patient weight at natural class prevalence. All ten fits are
frozen before same-query test outcomes. The primary is context versus original
HAT at eight groups; four-group and simpler readout results stay secondary.

```powershell
python -m unittest experiments.spes_agent_v15.test_readout
python -m experiments.spes_agent_v15.experiment --project . --output outputs/spes_agent_v15 --stage freeze
python -m experiments.spes_agent_v15.experiment --project . --output outputs/spes_agent_v15 --stage banks --workers 2
python -m experiments.spes_agent_v15.experiment --project . --output outputs/spes_agent_v15 --stage fit
python -m experiments.spes_agent_v15.experiment --project . --output outputs/spes_agent_v15 --stage evaluate
```

This stage uses local computation, zero new neural fits and zero Gemini calls.
It is a representation/readout adaptation test, not a completed LLM comparison.
No clinical efficacy, external validation or multi-agent advantage is claimed.
Raw data, labels and model assets stay outside Git.
