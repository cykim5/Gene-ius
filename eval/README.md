# Evaluation (Stages 8 and 9)

| File | Test |
|---|---|
| `go_no_go.py` | Stage 2 check: enough reclassified VUS? Run first. |
| `time_saved.py` | Time-saved curve (test 3) |
| `reclassification.py` | Headline test: A, B, C on 2023 VUS reclassified since (to do) |
| `rank_today.py` | Stage 9: retrain on all data, rank today's VUS (to do) |

Full test list (from the plan):

1. VUS reclassification (headline)
2. Same test without AlphaMissense
3. Time-saved curve
4. HPO time-based test
5. Symptom scoring with partial credit
6. Telling sibling diseases apart
7. Mechanism vs LoGoFunc
8. Treatment vs the migalastat list
9. Held-out genes
10. Results by data amount
11. Remove one layer at a time
12. Confidence check
13. Agent benchmark
14. Cardiac panel

Rules: report n for every number, add bootstrap confidence intervals, and keep weak results next to the baselines.
