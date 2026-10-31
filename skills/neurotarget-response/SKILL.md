---
name: neurotarget-response
description: Read one hash-checked simulated SEEG response from a frozen diagnostic bank. Returns only the queried experiment, a covariance reference and public observation features.
metadata:
  category: scientific
  tags: [neuroscience, simulation, diagnostic-stimulation]
---

# Diagnostic response query

Use this tool for a single permitted contact pair. The experiment scheduler
enforces the query budget and forbids repeated targets within a policy.
The bank is a computational cache, not information available to the agent.
No hidden generator parameters, reference labels or accuracy scores are returned.

## Parameters
- `--project`: repository root.
- `--bank-path`: complete bank.json under this project's outputs/response_banks.
- `--target-id`: one permitted contact pair.
- `--format json`: structured output.

## Example
`python {baseDir}/scripts/response.py --bank-path BANK_JSON --target-id CONTACT_PAIR --format json`
