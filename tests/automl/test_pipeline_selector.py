import pytest

from tempor.automl import pipeline_selector
from tempor.plugins.pipeline import PipelineBase

TEST_PREDICTOR_CASES = [
    ("prediction.one_off.classification", "nn_classifier"),
    ("prediction.one_off.regression", "nn_regressor"),
    ("time_to_event", "dynamic_deephit"),
    ("treatments.one_off.regression", "synctwin_regressor"),
    ("treatments.temporal.classification", "crn_classifier"),
    ("treatments.temporal.regression", "crn_regressor"),
]
TEST_STATIC_IMPUTERS_CASES = [
    [],
    pipeline_selector.DEFAULT_STATIC_IMPUTERS,
]
TEST_STATIC_SCALERS_CASES = [
    [],
    pipeline_selector.DEFAULT_STATIC_SCALERS,
]
TEST_TEMPORAL_IMPUTERS_CASES = [
    [],
    pipeline_selector.DEFAULT_TEMPORAL_IMPUTERS,
]
TEST_TEMPORAL_SCALERS_CASES = [
    [],
    pipeline_selector.DEFAULT_TEMPORAL_SCALERS,
]


@pytest.mark.parametrize("task_type,predictor", TEST_PREDICTOR_CASES)
@pytest.mark.parametrize("static_imputers", TEST_STATIC_IMPUTERS_CASES)
@pytest.mark.parametrize("static_scalers", TEST_STATIC_SCALERS_CASES)
@pytest.mark.parametrize("temporal_imputers", TEST_TEMPORAL_IMPUTERS_CASES)
@pytest.mark.parametrize("temporal_scalers", TEST_TEMPORAL_SCALERS_CASES)
def test_init(task_type, predictor, static_imputers, static_scalers, temporal_imputers, temporal_scalers):
    ps = pipeline_selector.PipelineSelector(
        task_type=task_type,
        predictor=predictor,
        static_imputers=static_imputers,
        static_scalers=static_scalers,
        temporal_imputers=temporal_imputers,
        temporal_scalers=temporal_scalers,
    )

    assert len(ps.static_imputers) == len(static_imputers)
    assert len(ps.static_scalers) == len(static_scalers)
    assert len(ps.temporal_imputers) == len(temporal_imputers)
    assert len(ps.temporal_scalers) == len(temporal_scalers)
    assert ps.task_type == task_type
    assert ps.predictor.name == predictor


@pytest.mark.parametrize("task_type,predictor", TEST_PREDICTOR_CASES)
@pytest.mark.parametrize("static_imputers", TEST_STATIC_IMPUTERS_CASES)
@pytest.mark.parametrize("static_scalers", TEST_STATIC_SCALERS_CASES)
@pytest.mark.parametrize("temporal_imputers", TEST_TEMPORAL_IMPUTERS_CASES)
@pytest.mark.parametrize("temporal_scalers", TEST_TEMPORAL_SCALERS_CASES)
def test_hyperparameter_space(
    task_type, predictor, static_imputers, static_scalers, temporal_imputers, temporal_scalers
):
    ps = pipeline_selector.PipelineSelector(
        task_type=task_type,
        predictor=predictor,
        static_imputers=static_imputers,
        static_scalers=static_scalers,
        temporal_imputers=temporal_imputers,
        temporal_scalers=temporal_scalers,
    )

    hps = ps.hyperparameter_space()
    hps_names = [hp.name for hp in hps]

    for hp in ps.predictor.hyperparameter_space():
        assert f"[{predictor}]({hp.name})" in hps_names

    if static_imputers:
        assert f"<candidates>({pipeline_selector.PREFIX_STATIC_IMPUTERS})" in hps_names
        for candidate in ps.static_imputers:
            for hp in candidate.hyperparameter_space():
                assert f"[{candidate.name}]({hp.name})" in hps_names
    else:
        assert f"<candidates>({pipeline_selector.PREFIX_STATIC_IMPUTERS})" not in hps_names
    if static_scalers:
        assert f"<candidates>({pipeline_selector.PREFIX_STATIC_SCALERS})" in hps_names
        for candidate in ps.static_scalers:
            for hp in candidate.hyperparameter_space():
                assert f"[{candidate.name}]({hp.name})" in hps_names
    else:
        assert f"<candidates>({pipeline_selector.PREFIX_STATIC_SCALERS})" not in hps_names
    if temporal_imputers:
        assert f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_IMPUTERS})" in hps_names
        for candidate in ps.temporal_imputers:
            for hp in candidate.hyperparameter_space():
                assert f"[{candidate.name}]({hp.name})" in hps_names
    else:
        assert f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_IMPUTERS})" not in hps_names
    if temporal_scalers:
        assert f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_SCALERS})" in hps_names
        for candidate in ps.temporal_scalers:
            for hp in candidate.hyperparameter_space():
                assert f"[{candidate.name}]({hp.name})" in hps_names
    else:
        assert f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_SCALERS})" not in hps_names


@pytest.mark.parametrize("task_type,predictor", TEST_PREDICTOR_CASES)
@pytest.mark.parametrize("static_imputers", TEST_STATIC_IMPUTERS_CASES)
@pytest.mark.parametrize("static_scalers", TEST_STATIC_SCALERS_CASES)
@pytest.mark.parametrize("temporal_imputers", TEST_TEMPORAL_IMPUTERS_CASES)
@pytest.mark.parametrize("temporal_scalers", TEST_TEMPORAL_SCALERS_CASES)
def test_sample_hyperparameters(
    task_type, predictor, static_imputers, static_scalers, temporal_imputers, temporal_scalers
):
    ps = pipeline_selector.PipelineSelector(
        task_type=task_type,
        predictor=predictor,
        static_imputers=static_imputers,
        static_scalers=static_scalers,
        temporal_imputers=temporal_imputers,
        temporal_scalers=temporal_scalers,
    )

    sample = ps.sample_hyperparameters()

    for hp in ps.predictor.hyperparameter_space():
        assert f"[{predictor}]({hp.name})" in sample

    if static_imputers:
        assert f"<candidates>({pipeline_selector.PREFIX_STATIC_IMPUTERS})" in sample
        for candidate in ps.static_imputers:
            for hp in candidate.hyperparameter_space():
                assert f"[{candidate.name}]({hp.name})" in sample
    else:
        assert f"<candidates>({pipeline_selector.PREFIX_STATIC_IMPUTERS})" not in sample
    if static_scalers:
        assert f"<candidates>({pipeline_selector.PREFIX_STATIC_SCALERS})" in sample
        for candidate in ps.static_scalers:
            for hp in candidate.hyperparameter_space():
                assert f"[{candidate.name}]({hp.name})" in sample
    else:
        assert f"<candidates>({pipeline_selector.PREFIX_STATIC_SCALERS})" not in sample
    if temporal_imputers:
        assert f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_IMPUTERS})" in sample
        for candidate in ps.temporal_imputers:
            for hp in candidate.hyperparameter_space():
                assert f"[{candidate.name}]({hp.name})" in sample
    else:
        assert f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_IMPUTERS})" not in sample
    if temporal_scalers:
        assert f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_SCALERS})" in sample
        for candidate in ps.temporal_scalers:
            for hp in candidate.hyperparameter_space():
                assert f"[{candidate.name}]({hp.name})" in sample
    else:
        assert f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_SCALERS})" not in sample


def assert_actual_params(pipe, sample, chosen):
    estimator = [p for p in pipe.stages if p.name == chosen][0]  # pylint: disable=no-member
    relevant = [(k.split("(")[-1][:-1], v) for k, v in sample.items() if f"[{chosen}]" in k]
    for k, v in relevant:
        assert getattr(estimator.params, k) == v


@pytest.mark.filterwarnings("ignore:.*bool8.*:DeprecationWarning")  # Expected.
@pytest.mark.parametrize("task_type,predictor", TEST_PREDICTOR_CASES)
@pytest.mark.parametrize("static_imputers", TEST_STATIC_IMPUTERS_CASES)
@pytest.mark.parametrize("static_scalers", TEST_STATIC_SCALERS_CASES)
@pytest.mark.parametrize("temporal_imputers", TEST_TEMPORAL_IMPUTERS_CASES)
@pytest.mark.parametrize("temporal_scalers", TEST_TEMPORAL_SCALERS_CASES)
def test_pipeline_from_hps(task_type, predictor, static_imputers, static_scalers, temporal_imputers, temporal_scalers):
    ps = pipeline_selector.PipelineSelector(
        task_type=task_type,
        predictor=predictor,
        static_imputers=static_imputers,
        static_scalers=static_scalers,
        temporal_imputers=temporal_imputers,
        temporal_scalers=temporal_scalers,
    )

    sample = ps.sample_hyperparameters()
    pipe = ps.pipeline_from_hps(sample)
    actual_estimator_names = [p.name for p in pipe.stages]  # pylint: disable=no-member

    assert isinstance(pipe, PipelineBase)

    if static_imputers:
        chosen = sample[f"<candidates>({pipeline_selector.PREFIX_STATIC_IMPUTERS})"]
        assert chosen in actual_estimator_names
        assert_actual_params(pipe, sample, chosen)
    if static_scalers:
        chosen = sample[f"<candidates>({pipeline_selector.PREFIX_STATIC_SCALERS})"]
        assert chosen in actual_estimator_names
        assert_actual_params(pipe, sample, chosen)
    if temporal_imputers:
        chosen = sample[f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_IMPUTERS})"]
        assert chosen in actual_estimator_names
        assert_actual_params(pipe, sample, chosen)
    if temporal_scalers:
        chosen = sample[f"<candidates>({pipeline_selector.PREFIX_TEMPORAL_SCALERS})"]
        assert chosen in actual_estimator_names
        assert_actual_params(pipe, sample, chosen)

    assert_actual_params(pipe, sample, predictor)
