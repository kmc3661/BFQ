"""Statistics-driven allocation rule for W3A16 and W4A8.

Consumes QE statistics without model-specific branches or external target files.
"""
import math

BOUNDS = {
    "w3a16": {"r": (0.016691189644144535, 0.20699458358498954),
              "C": (0.37533061927148087, 0.9154848482148373),
              "M": (0.06507958780479688, 1.3710523343192316)},
    "w4a8": {"A": (0., .8), "S": (0., .7), "C": (0., .6)},
}
KEYS = {
    "w3a16": {"r": "beneficial_ratio", "C": "top10_share_weight", "M": "harmful_ratio"},
    "w4a8": {"A": "activation_share", "S": "support90_frac_total",
              "C": "top10_share_total"},
}


def normalized_statistics(stats, mode):
    if mode not in BOUNDS:
        raise ValueError("COCO v3 covers W3A16/W4A8 only")
    normalized = {}
    for name, key in KEYS[mode].items():
        value = float(stats[key])
        if not math.isfinite(value) or value < 0 or (name != "M" and value > 1):
            raise ValueError(f"Invalid QE statistic {key}: {value}")
        lo, hi = BOUNDS[mode][name]
        normalized[name] = max(0., min(1., (value-lo)/(hi-lo)))
    return normalized


def continuous_prediction(stats, mode):
    z = normalized_statistics(stats, mode)
    if mode == "w3a16":
        r, c, m = z["r"], z["C"], z["M"]
        base = 20.-(2.+1.6*(1.-r))*r*(1.-c)
        mean = 22.+5.*c*m**3+r*(1.-c)
        gamma = 2.12+.45*c
    else:
        a, s, c = z["A"], z["S"], z["C"]
        base = 12.+2.*a*s+9.5*(1.-c)
        mean = 16.+9.*a*s+s*(1.-c)
        gamma = 1.+.9*c+.6*(1.-a)*(1.-c)
    return [base, mean, gamma], z


def predict(stats, mode):
    raw, normalized = continuous_prediction(stats, mode)
    # A profile with no measured effect needs no adaptive redistribution.
    if float(stats["harmful_ratio"]) == 0 and float(stats.get("beneficial_ratio", 0)) == 0:
        raw = [20., 20., 1.]
    base = max(2, min(48, round(raw[0])))
    mean = max(base, min(48, round(raw[1])))
    gamma = round(round(raw[2]/.05)*.05, 2)
    result = {
        "rule_name": f"{mode}_coco_v3",
        "g_ref": 20.,
        "BFQ_BUDGET_BASE_GRID": str(base),
        "BFQ_N_GRID_TARGET_MEAN": str(mean),
        "BFQ_BUDGET_BONUS_GAMMA": str(gamma),
        "BFQ_BUDGET_BONUS_HIGH_PERCENTILE": "90",
        "explicit_rule_unrounded": raw,
    }
    for name, value in normalized.items():
        result[name.lower()+"_bar"] = value
    return result
