"""Freeze all diagnostic wrapper defaults and job-specific explicit overrides."""
import copy
import json
from recovery_engine import ENGINE

def defaults():
    return json.loads((ENGINE / "bcf_defaults.json").read_text())

def actual_parameters(kwargs):
    result = copy.deepcopy(defaults()["defaults"])
    for source, target in (("general_params", "general_params_default"),
                           ("prognostic_forest_params", "prognostic_forest_params_default"),
                           ("treatment_effect_forest_params", "treatment_effect_forest_params_default"),
                           ("random_effects_params", "rfx_params_default")):
        for key, value in kwargs.get(source, {}).items():
            result[target][key] = value.tolist() if hasattr(value, "tolist") else value
    return result
