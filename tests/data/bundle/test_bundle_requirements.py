# pylint: disable=redefined-outer-name, unused-argument

from unittest.mock import Mock

import pytest

import tempor.exc
from tempor.data import bundle
from tempor.data.bundle import requirements as r


@pytest.fixture
def patch_data_validation(monkeypatch):
    import tempor.data.container._validator as v

    monkeypatch.setattr(v.DataValidator, "_validate", Mock())


@pytest.fixture
def patch_time_series_samples_init(monkeypatch):
    import tempor.data.samples as s

    monkeypatch.setattr(s.TimeSeriesSamples, "__init__", Mock(return_value=None))


@pytest.fixture
def patch_all(patch_data_validation, patch_time_series_samples_init):
    pass


def test_bundle_requirement_data_present_success(patch_all):
    data_container = Mock()

    req = r.DataPresent(["Xt", "Yt"])
    data_bundle = bundle.DataBundle.from_data_containers(Xt=data_container, Yt=data_container)
    validator = r.DataBundleValidator()

    validator.validate(data_bundle, requirements=[req])


@pytest.mark.parametrize(
    "samples_required",
    (
        ["Yt"],
        ["Xt", "Yt"],
        ["Ys", "At", "Ae"],
    ),
)
def test_bundle_requirement_data_present_fail(samples_required, patch_all):
    data_container = Mock()

    req = r.DataPresent(samples_required)
    data_bundle = bundle.DataBundle.from_data_containers(Xt=data_container)
    validator = r.DataBundleValidator()

    with pytest.raises(tempor.exc.DataValidationFailedException) as excinfo:
        validator.validate(data_bundle, requirements=[req])
    assert "defined" in str(excinfo.getrepr())
