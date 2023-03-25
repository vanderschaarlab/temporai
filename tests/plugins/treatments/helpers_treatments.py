import numpy as np


def simulate_horizons(data):
    return [tc.time_indexes()[0][len(tc.time_indexes()[0]) // 2 :] for tc in data.time_series]


def simulate_treatments_scenarios(data, n_counterfactuals_per_sample: int = 2):
    horizons = simulate_horizons(data)

    treatment_scenarios = []
    for idx, sample_idx in enumerate(data.time_series.sample_index()):
        sample_scenarios = []
        treat = data.predictive.treatments[sample_idx].dataframe()
        horizon_counterfactuals_sample = horizons[idx]

        for treat_sc_idx in range(n_counterfactuals_per_sample):
            np.random.seed(12345 + treat_sc_idx)
            treat_sc = np.random.randint(low=0, high=1 + 1, size=(len(horizon_counterfactuals_sample), treat.shape[1]))
            sample_scenarios.append(treat_sc)
        treatment_scenarios.append(sample_scenarios)

    return horizons, treatment_scenarios
