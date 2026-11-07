"""Make simulated patients from PrimeKG.

For each sampled disease: keep a fraction of its HPO terms, add a few random noise
terms, and record the disease's genes as the answer. Also writes the list of held-out
diseases so the GNN can drop their phenotype edges before training.

    python eval/simulate_patients.py --n 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geneius.data import get_kg  # noqa: E402

HERE = Path(__file__).resolve().parent


def simulate(n: int, keep: float, noise: int, min_terms: int, seed: int) -> list[dict]:
    kg = get_kg()
    rng = random.Random(seed)
    eligible = sorted(d for d, ts in kg.disease_hpo.items() if len(ts) >= min_terms and kg.disease_genes.get(d))
    pool = sorted({t for ts in kg.disease_hpo.values() for t in ts})
    patients = []
    for i, d in enumerate(rng.sample(eligible, min(n, len(eligible)))):
        terms = sorted(kg.disease_hpo[d])
        kept = rng.sample(terms, max(1, round(len(terms) * keep)))
        others = [t for t in pool if t not in kg.disease_hpo[d]]
        hpo_ids = kept + rng.sample(others, min(noise, len(others)))
        rng.shuffle(hpo_ids)
        patients.append({
            "id": f"sim-{i:04d}",
            "disease": d,
            "true_genes": sorted(kg.disease_genes[d]),
            "hpo_ids": hpo_ids,
        })
    return patients


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--keep", type=float, default=0.6, help="fraction of the disease's terms to keep")
    ap.add_argument("--noise", type=int, default=2, help="random unrelated terms to add")
    ap.add_argument("--min-terms", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "data" / "simulated.jsonl"))
    ap.add_argument("--heldout-out", default=str(HERE / "data" / "heldout_diseases.txt"))
    args = ap.parse_args(argv)

    patients = simulate(args.n, args.keep, args.noise, args.min_terms, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in patients:
            f.write(json.dumps(p) + "\n")
    with open(args.heldout_out, "w") as f:
        f.write("\n".join(sorted({p["disease"] for p in patients})) + "\n")
    print(f"Wrote {len(patients)} patients to {args.out}")
    print(f"Wrote held-out diseases to {args.heldout_out}")


if __name__ == "__main__":
    main()
