"""Stage 3: Monte Carlo simulation of a budget-constrained retention campaign.

Each simulation run

1. resamples the test customers with replacement (sampling uncertainty),
2. draws the campaign's acceptance rate from a Beta prior (uncertainty about the offer),
3. contacts the top ``k`` share of customers by predicted churn probability (or a random
   ``k`` share as the baseline), and
4. draws, for every contacted churner, whether they accept the offer and stay.

Net savings = retained value - incentives paid - contact costs, reported per 1,000
customers. The economics are the illustrative assumptions in ``business`` (config); the
sensitivity grid shows how conclusions change when they change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import plotting as P
from .decision import customer_economics, expected_offer_value
from .runtime import artifact_path, figure_path, results_path, save_json, to_markdown


def simulate_campaign(
    y: np.ndarray,
    p: np.ndarray,
    value: np.ndarray,
    incentive: np.ndarray,
    contact_cost: float,
    fraction: float,
    strategy: str = "model",
    n_sim: int = 2000,
    prior: tuple[float, float] = (6.0, 14.0),
    seed: int = 0,
    ev_acceptance: float | None = None,
) -> np.ndarray:
    """Return ``n_sim`` simulated net savings per 1,000 customers.

    ``strategy`` is ``"model"`` (top-``fraction`` by ``p``), ``"random"`` or ``"ev"``
    (contact customers with positive expected value under ``ev_acceptance``; ``fraction``
    is ignored).
    """
    rng = np.random.default_rng(seed)
    y, p, value, incentive = map(np.asarray, (y, p, value, incentive))
    n = len(y)
    out = np.empty(n_sim)
    k = int(round(fraction * n))
    for s in range(n_sim):
        idx = rng.integers(0, n, n)
        yy, pp, vv, dd = y[idx], p[idx], value[idx], incentive[idx]
        if strategy == "model":
            target = np.zeros(n, bool)
            target[np.argsort(-pp, kind="stable")[:k]] = True
        elif strategy == "random":
            target = np.zeros(n, bool)
            target[rng.choice(n, k, replace=False)] = True
        elif strategy == "ev":
            target = expected_offer_value(pp, vv, dd, contact_cost, ev_acceptance) > 0
        else:
            raise ValueError(strategy)
        gamma = rng.beta(*prior)
        accepted = rng.random(n) < gamma
        churner = target & (yy == 1)
        saved = churner & accepted
        stayer = target & (yy == 0)
        net = vv[saved].sum() - dd[saved].sum() - dd[stayer].sum() - contact_cost * target.sum()
        out[s] = net * 1000 / n
    return out


def expected_best(
    y, p, value, incentive, contact_cost, acceptance
) -> tuple[float, float, float, float]:
    """Profit per 1,000 customers on an evaluation set with known outcomes.

    Returns (best profit over contact shares, that share, expected-value-rule profit,
    expected-value-rule share). The first two pick the share with hindsight and are an
    upper reference; the expected-value rule is a policy that can actually be deployed.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    per = y * acceptance * (value - incentive) - (1 - y) * incentive - contact_cost
    cum = np.concatenate([[0.0], np.cumsum(per[np.argsort(-p, kind="stable")])])
    i = int(np.argmax(cum))
    ev_target = expected_offer_value(p, value, incentive, contact_cost, acceptance) > 0
    return cum[i] * 1000 / n, i / n, per[ev_target].sum() * 1000 / n, float(ev_target.mean())


def run_campaign(cfg: dict) -> pd.DataFrame:
    camp, biz, seed = cfg["campaign"], cfg["business"], cfg["seed"]
    df = pd.read_csv(artifact_path(cfg, "test_calibrated.csv"))
    y, p = df.y.to_numpy(), df.p_calibrated.to_numpy()
    v, d = df.value.to_numpy(), df.incentive.to_numpy()
    f, prior, n_sim = biz["contact_cost"], tuple(camp["acceptance_prior"]), camp["n_simulations"]

    rows = []
    for frac in camp["budget_fractions"]:
        for strategy in ("model", "random"):
            sims = simulate_campaign(y, p, v, d, f, frac, strategy, n_sim, prior, seed)
            rows.append(
                {
                    "strategy": strategy,
                    "contact_share": frac,
                    "mean": sims.mean(),
                    "p05": np.quantile(sims, 0.05),
                    "p95": np.quantile(sims, 0.95),
                    "prob_loss": float((sims < 0).mean()),
                }
            )
    ev = simulate_campaign(
        y, p, v, d, f, 0.0, "ev", n_sim, prior, seed, ev_acceptance=biz["acceptance_rate"]
    )
    ev_share = float((expected_offer_value(p, v, d, f, biz["acceptance_rate"]) > 0).mean())
    rows.append(
        {
            "strategy": "ev_rule",
            "contact_share": ev_share,
            "mean": ev.mean(),
            "p05": np.quantile(ev, 0.05),
            "p95": np.quantile(ev, 0.95),
            "prob_loss": float((ev < 0).mean()),
        }
    )
    sim = pd.DataFrame(rows)
    sim.to_csv(results_path(cfg, "campaign_simulation.csv"), index=False)

    grid = []
    for g in camp["sensitivity_acceptance"]:
        for months in camp["sensitivity_incentive_months"]:
            vv, dd = customer_economics(
                df.value / biz["horizon_months"], {**biz, "incentive_months": months}
            )
            best, share, ev_profit, ev_sh = expected_best(y, p, vv, dd, f, g)
            grid.append(
                {
                    "acceptance_rate": g,
                    "incentive_months": months,
                    "hindsight_best_profit_per_1000": best,
                    "hindsight_best_share": share,
                    "ev_rule_profit_per_1000": ev_profit,
                    "ev_rule_share": ev_sh,
                }
            )
    sens = pd.DataFrame(grid)
    sens.to_csv(results_path(cfg, "campaign_sensitivity.csv"), index=False)

    best_row = sim[sim.strategy == "model"].sort_values("mean", ascending=False).iloc[0]
    rand_same = sim[
        (sim.strategy == "random") & (sim.contact_share == best_row.contact_share)
    ].iloc[0]
    summary = {
        "n_simulations": n_sim,
        "acceptance_prior_beta": prior,
        "best_budget_share_model": float(best_row.contact_share),
        "best_budget_mean_per_1000": float(best_row["mean"]),
        "best_budget_90pct_interval": [float(best_row.p05), float(best_row.p95)],
        "best_budget_prob_loss": float(best_row.prob_loss),
        "random_same_budget_mean_per_1000": float(rand_same["mean"]),
        "ev_rule_share": ev_share,
        "ev_rule_mean_per_1000": float(ev.mean()),
        "ev_rule_90pct_interval": [float(np.quantile(ev, 0.05)), float(np.quantile(ev, 0.95))],
    }
    save_json(summary, results_path(cfg, "campaign_summary.json"))
    tab = sim.assign(
        Strategy=sim.strategy.map(
            {"model": "Top-k by model", "random": "Random k", "ev_rule": "Expected-value rule"}
        ),
        **{
            "Contacted": sim.contact_share.map("{:.0%}".format),
            "Mean net savings per 1,000": sim["mean"].map("{:,.0f}".format),
            "90% interval": [f"[{a:,.0f}, {b:,.0f}]" for a, b in zip(sim.p05, sim.p95)],
            "P(loss)": sim.prob_loss.map("{:.1%}".format),
        },
    )
    results_path(cfg, "campaign_simulation.md").write_text(
        to_markdown(
            tab[["Strategy", "Contacted", "Mean net savings per 1,000", "90% interval", "P(loss)"]]
        )
    )
    _plot_simulation(sim, cfg)
    _plot_sensitivity(sens, biz, cfg)
    return sim


def _plot_simulation(sim: pd.DataFrame, cfg: dict) -> None:
    P.apply_style()
    fig, ax = P.plt.subplots(figsize=(8.5, 4.8))
    for strategy, color, label in (
        ("model", P.BLUE, "Top-k by predicted risk"),
        ("random", P.MUTED, "Random k (baseline)"),
    ):
        s = sim[sim.strategy == strategy].sort_values("contact_share")
        x = s.contact_share * 100
        ax.fill_between(x, s.p05, s.p95, color=color, alpha=0.15, lw=0)
        ax.plot(
            x,
            s["mean"],
            "-o",
            ms=5,
            color=color,
            mec="white",
            mew=1.2,
            label=f"{label}, mean and 90% band",
        )
    ev = sim[sim.strategy == "ev_rule"].iloc[0]
    ax.errorbar(
        ev.contact_share * 100,
        ev["mean"],
        yerr=[[ev["mean"] - ev.p05], [ev.p95 - ev["mean"]]],
        fmt="D",
        ms=8,
        color=P.ORANGE,
        mec="white",
        mew=1.2,
        elinewidth=2,
        label="Expected-value rule",
    )
    ax.axhline(0, color=P.AXIS, lw=1)
    ax.set(
        xlabel="Share of customers contacted (%)",
        ylabel="Net savings per 1,000 customers",
        title="Simulated retention campaign (2,000 runs per point)",
    )
    ax.yaxis.set_major_formatter(P.plt.FuncFormatter(lambda val, _: f"{val:,.0f}"))
    ax.legend(loc="lower left")
    P.savefig(fig, figure_path(cfg, "campaign_simulation.png"))


def _plot_sensitivity(sens: pd.DataFrame, biz: dict, cfg: dict) -> None:
    P.apply_style()
    piv = sens.pivot(
        index="acceptance_rate", columns="incentive_months", values="ev_rule_profit_per_1000"
    )
    share = sens.pivot(index="acceptance_rate", columns="incentive_months", values="ev_rule_share")
    fig, ax = P.plt.subplots(figsize=(7.5, 4.8))
    vals = piv.to_numpy()
    if vals.min() >= 0:  # magnitude only: single-hue sequential ramp
        cmap, vmin, vmax = P.SEQUENTIAL, 0.0, vals.max()
    else:  # signed: red = loss, blue = profit, grey = break-even
        vmax = np.abs(vals).max()
        cmap, vmin = P.DIVERGING.reversed(), -vmax
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iat[i, j]
            ink = "white" if abs(val) > 0.55 * vmax else P.INK
            ax.text(
                j,
                i,
                f"{val:,.0f}\n({share.iat[i, j]:.0%})",
                ha="center",
                va="center",
                fontsize=9,
                color=ink,
            )
    ax.set_xticks(range(piv.shape[1]), [f"{m:g}" for m in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{g:.0%}" for g in piv.index])
    ax.set(
        xlabel="Incentive (months of charges)",
        ylabel="Offer acceptance rate among churners",
        title="Expected-value rule: profit per 1,000 customers (share contacted)",
    )
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    cb.set_label("Expected profit per 1,000 customers")
    cb.ax.yaxis.set_major_formatter(P.plt.FuncFormatter(lambda val, _: f"{val:,.0f}"))
    P.savefig(fig, figure_path(cfg, "campaign_sensitivity.png"))
