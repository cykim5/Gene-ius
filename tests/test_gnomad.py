import pandas as pd

from geneius.features.gnomad import agreement, common_not_in_clinvar, match, parse_hgvsp


def _variants():
    return pd.DataFrame({
        "variant_id": [1, 2, 3],
        "gene": ["KCNQ1", "KCNQ1", "RHO"],
        "position": [591, 100, 23],
        "ref_aa": ["R", "A", "P"],
        "alt_aa": ["C", "V", "H"],
        "key": ["chr11:2848000:C:T", "chr11:2700000:C:T", "chr3:129528801:C:A"],
        "protein_variant": ["R591C", "A100V", "P23H"],
    })


def _gnomad():
    return pd.DataFrame([
        # v3 exact DNA match for variant 1
        {"gene": "KCNQ1", "release": "r3", "gnomad_id": "11-2848000-C-T", "key": "chr11:2848000:C:T",
         "ref_aa": "R", "position": 591, "alt_aa": "C", "ac": 2, "an": 150000, "af": 2 / 150000,
         "hom": 0, "faf95_popmax": None},
        # v2 protein match for variant 1 (different coordinates, GRCh37)
        {"gene": "KCNQ1", "release": "r2", "gnomad_id": "11-2869000-C-T", "key": "chr11:2869000:C:T",
         "ref_aa": "R", "position": 591, "alt_aa": "C", "ac": 4, "an": 250000, "af": 4 / 250000,
         "hom": 0, "faf95_popmax": 1e-6},
        # common variant in v2 only, for variant 2
        {"gene": "KCNQ1", "release": "r2", "gnomad_id": "11-2721000-C-T", "key": "chr11:2721000:C:T",
         "ref_aa": "A", "position": 100, "alt_aa": "V", "ac": 5000, "an": 250000, "af": 0.02,
         "hom": 40, "faf95_popmax": 0.03},
        # common variant not in ClinVar at all
        {"gene": "KCNQ1", "release": "r2", "gnomad_id": "11-2722000-G-A", "key": "chr11:2722000:G:A",
         "ref_aa": "G", "position": 200, "alt_aa": "S", "ac": 1000, "an": 250000, "af": 0.004,
         "hom": 2, "faf95_popmax": 0.005},
    ])


def test_parse_hgvsp():
    assert parse_hgvsp("p.Arg591Cys") == ("R", 591, "C")
    assert parse_hgvsp("p.Arg591Ter") is None
    assert parse_hgvsp("p.Arg591=") is None
    assert parse_hgvsp(None) is None


def test_match():
    m = match(_variants(), _gnomad()).set_index("variant_id")
    assert m.loc[1, "r3_found"] and m.loc[1, "r2_found"]
    assert m.loc[1, "r3_ac"] == 2 and m.loc[1, "r2_ac"] == 4
    assert not m.loc[2, "r3_found"] and m.loc[2, "r2_found"]
    assert m.loc[2, "max_af"] == 0.02 and m.loc[2, "hom_total"] == 40
    assert not m.loc[3, "in_gnomad"] and m.loc[3, "max_af"] == 0.0


def test_common_not_in_clinvar():
    extra = common_not_in_clinvar(_variants(), _gnomad(), min_af=0.001)
    assert list(extra.position) == [200]


def test_agreement():
    a = agreement(match(_variants(), _gnomad()))
    assert a["both_found"] == 1 and a["within_5x"] == 1.0 and a["only_r2"] == 1
