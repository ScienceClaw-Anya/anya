# Sparse-observation scorer training

This exploratory follow-up was frozen after the negative v4 agent pilot.
It tests a concrete training/inference mismatch before spending more LLM calls.
It is not yet evidence that sparse training or collaboration improves accuracy.

The existing author's augmentation removes trial-by-site observations. With
many trials, this often leaves some response at nearly every site. The active
benchmark instead removes entire stimulation columns. Ten new fits train the
same fixed baseline/HAT architectures with **1-8 whole stimulus columns per
patient in each training batch**. The best epoch uses two fixed eight-column
subsets of seven validation patients. Each outer test patient remains absent
from normalization, fitting and early stopping.

The fixed evaluation replays all five v4 numerical strategies' existing eight-
measurement sets. It changes only the scorer. It does not show that a new
controller chooses better queries. The primary comparison is sparse HAT versus
the old HAT on the `random_0` sequence; all other results are retained. Fixed
presets, 25 epochs, patience 5, no new hyperparameter search or API calls.

```powershell
python -m experiments.spes_agent_v5.neural --project . --output outputs/spes_agent_v5 --workers 2
# Freeze before opening the new outer predictions:
python -m experiments.spes_agent_v5.evaluate --project . --output outputs/spes_agent_v5 --freeze-only
# After all ten fits finish:
python -m experiments.spes_agent_v5.evaluate --project . --output outputs/spes_agent_v5
python -m unittest experiments.spes_agent_v5.test_v5 -v
```

Training runs locally on the RTX 4090. Model checkpoints and public recordings
stay outside Git. The repeatedly inspected patient cohort supports exploratory
comparisons; a fresh cohort is still needed for independent confirmation.

Separately, v4's sequential single-agent controller had seven opportunities to
react between queries, while its batch collaboration controller had one.
Any later collaboration redesign should match the feedback schedule as well
as request and measurement budgets. The historical v4 comparison stays frozen.
