from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower


def bootstrap_mean_ci(values, n_bootstrap: int = 10000, ci: float = 0.95, random_state: int = 42):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    rng = np.random.default_rng(random_state)
    boot_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(sample.mean())

    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def bootstrap_ci_by_internship(
    df: pd.DataFrame,
    internship_col: str = "has_internship",
    gpa_col: str = "gpa",
    n_bootstrap: int = 10000,
):
    results = []

    clean = df[[internship_col, gpa_col]].dropna()

    for status, group in clean.groupby(internship_col):
        result = bootstrap_mean_ci(group[gpa_col].values, n_bootstrap=n_bootstrap)
        results.append({
            "internship_status": status,
            "mean_gpa": result["mean"],
            "bootstrap_ci_lower": result["ci_lower"],
            "bootstrap_ci_upper": result["ci_upper"],
            "n": len(group),
        })

    return pd.DataFrame(results)


def welch_ttest_confidence_interval(group1, group2, confidence: float = 0.95):
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    mean_diff = g1.mean() - g2.mean()
    var1 = g1.var(ddof=1)
    var2 = g2.var(ddof=1)
    n1 = len(g1)
    n2 = len(g2)

    se = np.sqrt(var1 / n1 + var2 / n2)

    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
    df_welch = df_num / df_den

    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df_welch)
    margin = t_crit * se

    return {
        "mean_difference": float(mean_diff),
        "ci_lower": float(mean_diff - margin),
        "ci_upper": float(mean_diff + margin),
    }


def observed_effect_size(group1, group2):
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    pooled_sd = np.sqrt(((g1.std(ddof=1) ** 2) + (g2.std(ddof=1) ** 2)) / 2)
    d = (g1.mean() - g2.mean()) / pooled_sd
    return float(abs(d))


def required_sample_size_for_power(group1, group2, power: float = 0.80, alpha: float = 0.05):
    effect_size = observed_effect_size(group1, group2)
    analysis = TTestIndPower()
    n_required = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)
    return {
        "effect_size_cohens_d": effect_size,
        "required_n_per_group": float(n_required),
    }


def false_positive_rate_simulation(
    n_simulations: int = 5000,
    n_per_group: int = 30,
    mean: float = 3.0,
    std: float = 0.4,
    alpha: float = 0.05,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    significant_count = 0
    p_values = []

    for _ in range(n_simulations):
        group1 = rng.normal(loc=mean, scale=std, size=n_per_group)
        group2 = rng.normal(loc=mean, scale=std, size=n_per_group)

        _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        p_values.append(p_value)

        if p_value < alpha:
            significant_count += 1

    false_positive_rate = significant_count / n_simulations

    return {
        "alpha": float(alpha),
        "observed_false_positive_rate": float(false_positive_rate),
        "mean_p_value": float(np.mean(p_values)),
    }