# Attribution and scope

The compiled numerical kernel in `src/intervention/model.py` adapts the
six-variable Epileptor equations and integration/noise conventions from The
Virtual Brain library 2.10.0. Copyright (c) 2012–2025 Baycrest Centre for Geriatric
Care and others; GPL-3.0-or-later. The upstream license and citation instructions
are retained in LICENSE. This is an independently implemented benchmark wrapper,
not an endorsed TVB treatment-planning product.

TVB: Sanz Leon et al. (2013), https://doi.org/10.3389/fninf.2013.00010 .
Epileptor: Jirsa et al. (2014), https://doi.org/10.1093/brain/awu133 .
VEC: Dollomaja et al. (2025), https://doi.org/10.1371/journal.pcbi.1012911 .
Data: Dollomaja, Wang and Jirsa (2024), Virtual Epileptic Patient Cohort v1.0,
https://doi.org/10.25493/XWG2-S8X , CC BY 4.0. The dataset was transformed by
normalizing connectivity, deriving bipolar gains and onset proxies, and running
new virtual interventions. Data authors do not endorse these changes.

VEC's generation scripts were inspected at
a99c88354015f4c961d49a92c076ed0b675c740b. No license file was present in that
repository's tree. They remain ignored reference material and were not executed
or copied into tracked implementation files.

Neither toy invariant tests, reconstruction of simulated source features, nor
oracle-parameter intervention comparisons establish biological or clinical
validity. No raw clinical files, patient time series or generating parameters
are redistributed by this branch.
