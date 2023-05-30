from tempor.models.samplers import ImbalancedDatasetSampler


def test_basics():
    s = ImbalancedDatasetSampler(labels=[0, 1, 2, 3, 4], train_size=0.8)
    assert len(s) == 4
    tr, ts = s.train_test()
    assert len(tr) == 4
    assert len(ts) == 1
