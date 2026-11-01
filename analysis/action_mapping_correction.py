"""Persistent interpretation correction; historical numerical results stay intact."""
ACTION_NOTICE='''> **Action-mapping correction (2026-11-01):** Processed array columns are not shared physical stimulation sites across recording contacts. Metadata matching confirms that one column can mix several anatomical pairs. Historical v4-v10 subset-selection counts represent array-column ablation, **not eight physical stimulations or validated target selection**. That physical interpretation is withdrawn; recorded results remain preserved. Full-response SOZ classification does not require this alignment and is unaffected. See the [action-grain audit](https://github.com/ScienceClaw-Anya/anya/blob/main/docs/spes-physical-action-audit.md).'''


def corrected(text):
    return text if 'Action-mapping correction (2026-11-01)' in text else ACTION_NOTICE+'\n\n'+text
