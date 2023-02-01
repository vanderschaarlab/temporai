import re

import pytest

import tempor.model


class TestFitConfig:
    def test_fit_passes(self, df_time_series_num_nonan, df_event_num_nonan):
        class MyModel(tempor.model.TemporBaseModel):
            CONFIG = {
                "fit_config": {
                    "data_present": ["Xt", "Ae"],
                }
            }

            def _fit(self, data, **kwargs):
                pass

        my_model = MyModel()
        my_model.fit(Xt=df_time_series_num_nonan, Ae=df_event_num_nonan)

    def test_fit_fails_data_bundle_missing_item(self, df_time_series_num_nonan):
        class MyModel(tempor.model.TemporBaseModel):
            CONFIG = {
                "fit_config": {
                    "data_present": ["Xt", "Ae"],
                }
            }

            def _fit(self, data, **kwargs):
                pass

        my_model = MyModel()

        with pytest.raises(ValueError) as excinfo:
            my_model.fit(Xt=df_time_series_num_nonan)  # No Ae
        assert re.search(r".*[Ee]xpect.*Ae.*", str(excinfo.getrepr()))

    def test_fit_fails_data_container_validation_missing_data(self, df_time_series_num_nonan, df_static_cat_num_hasnan):
        class MyModel(tempor.model.TemporBaseModel):
            CONFIG = {"fit_config": {"data_present": ["Xt"], "Xs_config": {"allow_missing": False}}}

            def _fit(self, data, **kwargs):
                pass

        my_model = MyModel()

        with pytest.raises(ValueError) as excinfo:
            my_model.fit(Xt=df_time_series_num_nonan, Xs=df_static_cat_num_hasnan)
            # ^ df for Xs has nans, but Xs_config doesn't allow missing values.
        assert re.search(r".*contains.*null.*", str(excinfo.getrepr()))
