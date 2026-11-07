"""Gene panels (section 6 of the plan)."""

RETINA = [
    "RHO", "CEP290", "USH2A", "RPE65", "ABCA4", "GUCY2D", "CRX",
    "PRPH2", "BEST1", "NR2E3", "RPGR", "OPA1", "PAX6",
]

CARDIAC = [
    "SCN5A", "KCNQ1", "KCNH2", "KCNJ2", "CACNA1C",
    "MYH7", "MYBPC3", "LMNA", "PKP2",
]

FABRY = ["GLA"]

# Too large for one AlphaFold model, so no 3D edges
NO_3D_EDGES = {"USH2A", "RYR2", "TTN"}

# Related proteins for matching-residue edges
FAMILIES = {
    "SCN5A": ["SCN1A", "SCN2A", "SCN3A", "SCN4A", "SCN8A", "SCN9A", "SCN10A", "SCN11A"],
    "KCNQ1": ["KCNQ2", "KCNQ3", "KCNQ4", "KCNQ5"],
}
