"""
Two-threshold sweep: pass / review / fail, costed at production prevalence.

Reads outputs/test_scores.csv and src/config.py. For every pair
(t_low, t_high) on a 0.02 grid with t_low <= t_high, each test image is
decided with src.scoring.decide: score < t_low pass, t_low <= score < t_high
review, score >= t_high fail. Outcome cost per unit:

    pass    0 if intact, COST_FN_DOP if damaged
    review  COST_REVIEW_DOP either way (the person catches it, no FN)
    fail    COST_FAIL_CONFIRM_DOP if damaged (claim filed on the photo)
            COST_FAIL_CONFIRM_DOP + COST_FALSE_CLAIM_DOP if intact (a false claim)

Shares of positives and negatives landing in each outcome are rescaled to
1,000 units at BASE_RATE exactly as in scripts/sweep.py.

The grid is run once per value in COST_FALSE_CLAIM_SWEEP. Writes to outputs/:
    review_grid_fc<X>.csv      every pair at false-claim cost X, within-band flag
    review_grid.png            heatmap at COST_FALSE_CLAIM_DOP
    review_summary.json        chosen pair, band and dedup check per X
    review_load.csv            review/fail units, person-hours, days per container
    options.csv                do nothing, check everything, model alone, model + band
    (load and options at COST_FALSE_CLAIM_DOP)

Attempt 1 (no false-claim cost; fail cost the same either way) is kept as
outputs/review_attempt1_*. `--false-claim 0` reproduces it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.scoring import decide  # noqa: E402

OUT_DIR = ROOT / "outputs"
UNITS = 1000
STEP = 0.02
GRID = [float(round(i * STEP, 2)) for i in range(int(round(1 / STEP)) + 1)]
OUTCOMES = ("pass", "review", "fail")
REFERENCE_CONTAINER = "40ft_low"   # the 400-box container used in options.csv


def load_scores() -> list[dict]:
    with (OUT_DIR / "test_scores.csv").open(newline="", encoding="utf-8") as f:
        return [{"filename": r["filename"], "score": float(r["score"]), "label": int(r["label"]),
                 "dup": int(r["dup_in_trainval"])} for r in csv.DictReader(f)]


def shares(rows: list[dict], t_low: float, t_high: float) -> dict:
    """Share of positives and of negatives sent to each outcome."""
    counts = {"pos": dict.fromkeys(OUTCOMES, 0), "neg": dict.fromkeys(OUTCOMES, 0)}
    for r in rows:
        counts["pos" if r["label"] else "neg"][decide(r["score"], t_low, t_high)] += 1
    out = {}
    for grp, c in counts.items():
        n = sum(c.values())
        for o in OUTCOMES:
            out[f"{grp}_{o}"] = c[o] / n if n else 0.0
            out[f"{grp}_{o}_n"] = c[o]
    return out


def cost_per_1000(sh: dict, base_rate: float, false_claim: float) -> dict:
    pos = UNITS * base_rate
    neg = UNITS * (1 - base_rate)
    exp = {
        "fn": pos * sh["pos_pass"],
        "review": pos * sh["pos_review"] + neg * sh["neg_review"],
        "fail_damaged": pos * sh["pos_fail"],
        "fail_intact": neg * sh["neg_fail"],
        "pass_intact": neg * sh["neg_pass"],
    }
    exp["fail"] = exp["fail_damaged"] + exp["fail_intact"]
    cost = (exp["fn"] * config.COST_FN_DOP
            + exp["review"] * config.COST_REVIEW_DOP
            + exp["fail"] * config.COST_FAIL_CONFIRM_DOP
            + exp["fail_intact"] * false_claim)
    return {"exp_fn_per_1000": exp["fn"], "exp_review_per_1000": exp["review"],
            "exp_fail_per_1000": exp["fail"], "exp_false_claims_per_1000": exp["fail_intact"],
            "exp_pass_intact_per_1000": exp["pass_intact"],
            "cost_dop_per_1000": cost, "cost_eur_per_1000": cost / config.DOP_PER_EUR}


def sweep(rows: list[dict], base_rate: float, false_claim: float) -> list[dict]:
    grid = []
    for lo in GRID:
        for hi in GRID:
            if hi < lo:
                continue
            sh = shares(rows, lo, hi)
            grid.append({"t_low": lo, "t_high": hi, **sh, **cost_per_1000(sh, base_rate, false_claim)})
    return grid


def pick(grid: list[dict], tolerance: float) -> dict:
    cmin = min(g["cost_dop_per_1000"] for g in grid)
    at_min = [g for g in grid if g["cost_dop_per_1000"] == cmin]
    # Among exact ties prefer the narrowest band, then the lowest t_low: fewest
    # units to a person for the same cost.
    best = sorted(at_min, key=lambda g: (g["t_high"] - g["t_low"], g["t_low"]))[0]
    band = [g for g in grid if g["cost_dop_per_1000"] <= cmin + tolerance]
    for g in grid:
        g["within_band"] = int(g["cost_dop_per_1000"] <= cmin + tolerance)
    return {
        "t_low": best["t_low"], "t_high": best["t_high"],
        "min_cost_dop_per_1000": round(cmin, 2),
        "min_cost_eur_per_1000": round(cmin / config.DOP_PER_EUR, 2),
        "n_pairs_at_exact_min": len(at_min),
        "exact_min_pairs": [(g["t_low"], g["t_high"]) for g in at_min],
        "tolerance_dop": round(tolerance, 2),
        "n_pairs_within_band": len(band),
        "band_t_low_range": [min(g["t_low"] for g in band), max(g["t_low"] for g in band)],
        "band_t_high_range": [min(g["t_high"] for g in band), max(g["t_high"] for g in band)],
        "band_pairs": [(g["t_low"], g["t_high"]) for g in band],
        "chosen": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.items()},
    }


def review_load(sh: dict, base_rate: float) -> list[dict]:
    out = []
    for name, n_units in config.CONTAINER_SIZES.items():
        pos, neg = n_units * base_rate, n_units * (1 - base_rate)
        review = pos * sh["pos_review"] + neg * sh["neg_review"]
        fail = pos * sh["pos_fail"] + neg * sh["neg_fail"]
        hours = (review * config.MIN_PER_REVIEW + fail * config.MIN_PER_FAIL_CONFIRM) / 60
        days = hours / (config.STAFF * config.HOURS_PER_DAY)
        out.append({
            "container": name, "units": n_units,
            "units_to_review": round(review, 2), "units_failed": round(fail, 2),
            "units_passed": round(n_units - review - fail, 2),
            "expected_missed_damaged": round(pos * sh["pos_pass"], 3),
            "person_hours": round(hours, 2),
            "days_to_clear": round(days, 3),
            "staff": config.STAFF, "hours_per_day": config.HOURS_PER_DAY,
            "claim_window_days": config.CLAIM_WINDOW_DAYS,
            "fits_claim_window": int(days <= config.CLAIM_WINDOW_DAYS),
        })
    return out


def options(rows: list[dict], base_rate: float, chosen: dict, single_t: float, false_claim: float) -> list[dict]:
    n_ref = config.CONTAINER_SIZES[REFERENCE_CONTAINER]
    pos = UNITS * base_rate

    def row(option, lo, hi, c, note):
        return {"option": option, "t_low": lo, "t_high": hi,
                "exp_fn_per_1000": round(c["exp_fn_per_1000"], 4),
                "exp_review_per_1000": round(c["exp_review_per_1000"], 4),
                "exp_fail_per_1000": round(c["exp_fail_per_1000"], 4),
                "exp_false_claims_per_1000": round(c.get("exp_false_claims_per_1000", 0.0), 4),
                "cost_dop_per_1000": round(c["cost_dop_per_1000"], 2),
                "cost_eur_per_1000": round(c["cost_eur_per_1000"], 2),
                f"cost_dop_per_container_{n_ref}": round(c["cost_dop_per_1000"] * n_ref / UNITS, 2),
                f"cost_eur_per_container_{n_ref}": round(c["cost_eur_per_1000"] * n_ref / UNITS, 2),
                "note": note}

    nothing = {"exp_fn_per_1000": pos, "exp_review_per_1000": 0.0, "exp_fail_per_1000": 0.0,
               "cost_dop_per_1000": pos * config.COST_FN_DOP}
    nothing["cost_eur_per_1000"] = nothing["cost_dop_per_1000"] / config.DOP_PER_EUR
    everything = {"exp_fn_per_1000": 0.0, "exp_review_per_1000": float(UNITS), "exp_fail_per_1000": 0.0,
                  "cost_dop_per_1000": UNITS * config.COST_REVIEW_DOP}
    everything["cost_eur_per_1000"] = everything["cost_dop_per_1000"] / config.DOP_PER_EUR
    alone = cost_per_1000(shares(rows, single_t, single_t), base_rate, false_claim)
    band = cost_per_1000(shares(rows, chosen["t_low"], chosen["t_high"]), base_rate, false_claim)
    return [
        row("do_nothing", 1.0, 1.0, nothing, "every unit released"),
        row("person_checks_everything", 0.0, 0.0, everything,
            f"every unit reviewed at {config.COST_REVIEW_DOP} DOP; catches everything"),
        row("model_alone", single_t, single_t, alone,
            "t_low = t_high at the single-threshold optimum from sweep.py; no review band; "
            f"fail costs {config.COST_FAIL_CONFIRM_DOP} DOP, plus {false_claim} DOP false claim if intact"),
        row("model_plus_review_band", chosen["t_low"], chosen["t_high"], band,
            "pass / review / fail at the chosen pair"),
    ]


def heatmap(grid: list[dict], choice: dict, out: Path, false_claim: float) -> None:
    n = len(GRID)
    z = np.full((n, n), np.nan)
    idx = {t: i for i, t in enumerate(GRID)}
    for g in grid:
        z[idx[g["t_high"]], idx[g["t_low"]]] = g["cost_dop_per_1000"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    vmax = min(np.nanmax(z), choice["min_cost_dop_per_1000"] * 6)
    im = ax.imshow(z, origin="lower", cmap="viridis_r", vmin=np.nanmin(z), vmax=vmax,
                   extent=[-STEP / 2, 1 + STEP / 2, -STEP / 2, 1 + STEP / 2], aspect="equal")
    for lo, hi in choice["band_pairs"]:
        ax.add_patch(plt.Rectangle((lo - STEP / 2, hi - STEP / 2), STEP, STEP, fill=False,
                                   edgecolor="white", lw=0.8, alpha=0.9))
    ax.plot([choice["t_low"]], [choice["t_high"]], "o", color="#E45756", ms=10, mec="white",
            label=f"chosen ({choice['t_low']}, {choice['t_high']}): {choice['min_cost_dop_per_1000']:.0f} DOP")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("t_low (below: pass)")
    ax.set_ylabel("t_high (at or above: fail; between: review)")
    ax.set_title(f"Expected cost per {UNITS} units at base rate {config.BASE_RATE:.1%}, "
                 f"false claim {false_claim:.0f} DOP\n"
                 f"white cells: within one test positive ({choice['tolerance_dop']:.0f} DOP) of the minimum")
    fig.colorbar(im, ax=ax, label="DOP per 1,000 (clipped)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fc_tag(false_claim: float) -> str:
    return f"fc{int(false_claim) if float(false_claim).is_integer() else false_claim}"


def write_csv(path: Path, data: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
        w.writeheader()
        for g in data:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in g.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--false-claim", type=float, default=None,
                        help="run a single false-claim cost instead of COST_FALSE_CLAIM_SWEEP; "
                             "load/options/heatmap are written for it")
    args = parser.parse_args()
    if args.false_claim is None:
        sweep_values = list(dict.fromkeys([config.COST_FALSE_CLAIM_DOP] + list(config.COST_FALSE_CLAIM_SWEEP)))
        canonical = config.COST_FALSE_CLAIM_DOP
    else:
        sweep_values, canonical = [args.false_claim], args.false_claim

    rows = load_scores()
    rows_dedup = [r for r in rows if not r["dup"]]
    positives_in_test = sum(r["label"] for r in rows)
    tolerance = UNITS * config.BASE_RATE * (1 / positives_in_test) * config.COST_FN_DOP
    tolerance_dedup = UNITS * config.BASE_RATE * (1 / sum(r["label"] for r in rows_dedup)) * config.COST_FN_DOP
    single = json.loads((OUT_DIR / "sweep_summary.json").read_text(encoding="utf-8"))["all"]["threshold_at_min"]

    per_fc = {}
    for fc in sorted(sweep_values):
        grid = sweep(rows, config.BASE_RATE, fc)
        choice = pick(grid, tolerance)
        choice_dedup = pick(sweep(rows_dedup, config.BASE_RATE, fc), tolerance_dedup)
        write_csv(OUT_DIR / f"review_grid_{fc_tag(fc)}.csv", grid)
        per_fc[fc] = {"grid": grid, "choice": choice, "choice_dedup": choice_dedup}
        print(f"false claim {fc:>6.0f} DOP: chosen ({choice['t_low']}, {choice['t_high']}) "
              f"{choice['min_cost_dop_per_1000']} DOP/1000; {choice['n_pairs_at_exact_min']} exact ties; "
              f"{choice['n_pairs_within_band']} within {choice['tolerance_dop']} DOP, "
              f"t_low {choice['band_t_low_range']} t_high {choice['band_t_high_range']}; "
              f"dedup ({choice_dedup['t_low']}, {choice_dedup['t_high']}) {choice_dedup['min_cost_dop_per_1000']}")

    main_fc = per_fc[canonical]
    choice = main_fc["choice"]
    sh = shares(rows, choice["t_low"], choice["t_high"])
    load = review_load(sh, config.BASE_RATE)
    opts = options(rows, config.BASE_RATE, choice, single, canonical)
    write_csv(OUT_DIR / "review_load.csv", load)
    write_csv(OUT_DIR / "options.csv", opts)
    heatmap(main_fc["grid"], choice, OUT_DIR / "review_grid.png", canonical)

    summary = {
        "source": "outputs/test_scores.csv", "grid_step": STEP, "units": UNITS,
        "base_rate": config.BASE_RATE,
        "outcome_costs_dop": {"pass_damaged": config.COST_FN_DOP, "pass_intact": 0,
                              "review": config.COST_REVIEW_DOP,
                              "fail_damaged": config.COST_FAIL_CONFIRM_DOP,
                              "fail_intact": f"{config.COST_FAIL_CONFIRM_DOP} + false claim"},
        "false_claim_dop": canonical,
        "false_claim_sweep": sorted(sweep_values),
        "band_tolerance_rule": "cost <= min + 1000 * base_rate * (1 / positives_in_test) * COST_FN_DOP",
        "test_positives": positives_in_test, "test_negatives": len(rows) - positives_in_test,
        "single_threshold_optimum_from_sweep": single,
        "by_false_claim": {
            str(fc): {"all": v["choice"],
                      "dedup_check": {k: x for k, x in v["choice_dedup"].items() if k not in ("band_pairs", "chosen")}}
            for fc, v in per_fc.items()
        },
        "chosen": {"false_claim_dop": canonical, "t_low": choice["t_low"], "t_high": choice["t_high"],
                   "cost_dop_per_1000": choice["min_cost_dop_per_1000"]},
        "review_load": load,
        "options": opts,
    }
    (OUT_DIR / "review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote review_grid_fc*.csv, review_grid.png, review_summary.json, review_load.csv, options.csv "
          f"(load/options/heatmap at false claim {canonical})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
