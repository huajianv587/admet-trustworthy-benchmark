"""Audited compatibility repairs for legacy ADMET SMILES encodings.

The TDC Solubility_AqSolDB test split contains two pyridine N-oxides in a
legacy charge/valence notation accepted by RDKit 2023.09--2024.03 but rejected
by newer RDKit releases.  Each replacement below is parseable by RDKit 2026.03
and was verified to have the same Standard InChIKey as the legacy string under
RDKit 2024.03.6.  They are used only as model inputs; all published prediction
artifacts retain the original official SMILES, labels, indices, order, and
split membership.
"""

from __future__ import annotations


SMILES_COMPATIBILITY_POLICY = "tdc_legacy_pyridine_n_oxide_v1"

LEGACY_VALENCE_REPAIRS: dict[str, str] = {
    # Drug_ID: 3-methyl-n-oxidepyridine (3-methylpyridine N-oxide)
    "CC1=CC=C[NH+2]([O-])[CH-]1": "CC1=C[NH+]([O])[CH-]C=C1",
    # Drug_ID: n-oxidenicotinic acid (nicotinic acid N-oxide)
    "O=C(O)C1=C[NH+2]([O-])[CH-]C=C1": "[O][NH+]1C=C(C(=O)O)C=C[CH-]1",
}

LEGACY_VALENCE_REPAIR_INCHIKEYS: dict[str, str] = {
    "CC1=CC=C[NH+2]([O-])[CH-]1": "FHJYMRFPLWFYPD-UHFFFAOYSA-N",
    "O=C(O)C1=C[NH+2]([O-])[CH-]C=C1": "GDILJONVBZVAFH-UHFFFAOYSA-N",
}


def model_input_smiles(smiles: str) -> str:
    """Return a parser-compatible model input without changing sample identity."""

    value = str(smiles)
    return LEGACY_VALENCE_REPAIRS.get(value, value)


def representation_repair_record(
    smiles: str,
    *,
    split: str,
    sample_index: int,
) -> dict[str, str | int] | None:
    """Describe a compatibility repair for inclusion in a run manifest."""

    source = str(smiles)
    repaired = model_input_smiles(source)
    if repaired == source:
        return None
    return {
        "split": str(split),
        "sample_index": int(sample_index),
        "source_smiles": source,
        "model_input_smiles": repaired,
        "standard_inchikey": LEGACY_VALENCE_REPAIR_INCHIKEYS[source],
        "policy": SMILES_COMPATIBILITY_POLICY,
        "reason": "legacy_valence_notation_same_standard_inchikey",
    }
