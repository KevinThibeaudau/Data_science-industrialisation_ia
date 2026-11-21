"""Selection of the input variables by their p-value against the claim amount."""

from scipy import stats

from .config import (
    CATEGORICAL_CANDIDATES,
    NUMERIC_CANDIDATES,
    P_VALUE_THRESHOLD,
    TARGET,
)


def kruskal_p_value(claims, variable):
    """P-value of the Kruskal-Wallis test: does the amount differ between classes?"""
    groups = [
        group.to_numpy()
        for _, group in claims.groupby(variable)[TARGET]
        if len(group) >= 2
    ]
    if len(groups) < 2:
        return 1.0  # Not enough classes to show any link.
    return stats.kruskal(*groups).pvalue


def spearman_p_value(claims, variable):
    """P-value of the Spearman test: is the amount correlated with the variable?"""
    data = claims[[variable, TARGET]].dropna()
    return stats.spearmanr(data[variable], data[TARGET]).pvalue


def select_variables(claims, threshold=P_VALUE_THRESHOLD):
    """Return the variables whose p-value is below the threshold.

    `claims` holds claimants only, already cleaned. A variable with an
    undefined p-value (for instance a constant column) is not selected.
    """
    p_values = {v: kruskal_p_value(claims, v) for v in CATEGORICAL_CANDIDATES}
    p_values |= {v: spearman_p_value(claims, v) for v in NUMERIC_CANDIDATES}
    return [variable for variable, p in p_values.items() if p < threshold]
