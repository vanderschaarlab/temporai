import math

import torch

from tempor.models.ts_cde import NeuralCDE


def get_data():
    seq_len = 100
    t = torch.linspace(0.0, 4 * math.pi, seq_len)

    start = torch.rand(128) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(128)
    y[:64] = 1

    perm = torch.randperm(128)

    temporal = X[perm]
    outcome = y[perm]
    observation_times = t.unsqueeze(0).repeat(len(X), 1)

    static = torch.randn(len(X), 3)
    return static, temporal, observation_times, outcome


def test_sanity():
    static, temporal, observation_times, outcome = get_data()

    model = NeuralCDE(
        task_type="classification",
        n_static_units_in=static.shape[-1],
        n_temporal_units_in=temporal.shape[-1],
        output_shape=[2],
        n_units_hidden=8,
        n_iter=100,
    )
    model.fit(static, temporal, observation_times, outcome)

    # testing
    static, temporal, observation_times, outcome = get_data()
    score = model.score(static, temporal, observation_times, outcome)
    print(score)
    assert score > 0
