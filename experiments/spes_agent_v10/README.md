# Contact-model prototype: stopped before outer evaluation

The action-grain audit invalidated the physical interpretation of column-masked
training states. Ten small contact classifiers and 42,340 correlated contact/state
rows were generated, but no outer evaluation was run and no weights are adopted
for physical target selection. See the
[English action audit](../../docs/spes-physical-action-audit.md).

The prototype extracts 26 observed-only morphology, sample-index, FFT-bin and HAT
features. Time bins and FFT bins are numerical indices; no unverified clinical
time window, frequency band or meaning of the unused 128 inputs is assigned.
Training and normalization exclude each outer patient. These safeguards do not
repair a wrong physical action definition.

An expected-ranking helper computes a ratio of independent-label expected pair
counts. Its unit tests pass, but it was never used for physical acquisition.
It is not exact expected AUROC or a clinically calibrated risk calculation.

Training banks, fitted weights, protocols and a stop marker are retained locally.
The outer-evaluation lock stays in place. Rebuild states with verified physical
pairs in a fresh experiment before adopting a contact-posterior tool. This version
made no model API request and created no cloud resources.
