from tempor.plugins import plugin_loader


def test_tempor_plugin_loader_contents():
    all_plugins = plugin_loader.list()

    # Do some checks that expected plugins have been registered.
    # Update as plugins get added / reorganized.

    # Check categories:
    assert "prediction" in all_plugins
    assert "preprocessing" in all_plugins
    assert "time_to_event" in all_plugins
    assert "treatments" in all_plugins

    # Check subcategories:
    assert "imputation" in all_plugins["preprocessing"]
    assert "scaling" in all_plugins["preprocessing"]
    assert "one_off" in all_plugins["prediction"]
    assert "temporal" in all_plugins["prediction"]

    # Check sub-subcategories:
    assert "classification" in all_plugins["prediction"]["one_off"]
    assert "regression" in all_plugins["prediction"]["one_off"]
    assert "classification" in all_plugins["prediction"]["temporal"]
    assert "regression" in all_plugins["prediction"]["temporal"]

    # Check plugins:
    assert "cde_classifier" in all_plugins["prediction"]["one_off"]["classification"]
    assert "ode_classifier" in all_plugins["prediction"]["one_off"]["classification"]
    assert "nn_classifier" in all_plugins["prediction"]["one_off"]["classification"]
    assert "laplace_ode_classifier" in all_plugins["prediction"]["one_off"]["classification"]
    # ---
    assert "seq2seq_classifier" in all_plugins["prediction"]["temporal"]["classification"]
    # ---
    assert "laplace_ode_regressor" in all_plugins["prediction"]["one_off"]["regression"]
    assert "nn_regressor" in all_plugins["prediction"]["one_off"]["regression"]
    assert "ode_regressor" in all_plugins["prediction"]["one_off"]["regression"]
    assert "cde_regressor" in all_plugins["prediction"]["one_off"]["regression"]
    # ---
    assert "seq2seq_regressor" in all_plugins["prediction"]["temporal"]["regression"]
    # ---
    assert "nop_imputer" in all_plugins["preprocessing"]["imputation"]
    assert "ffill" in all_plugins["preprocessing"]["imputation"]
    assert "bfill" in all_plugins["preprocessing"]["imputation"]
    assert "static_imputation" in all_plugins["preprocessing"]["imputation"]
    assert "ts_minmax_scaler" in all_plugins["preprocessing"]["scaling"]
    assert "static_minmax_scaler" in all_plugins["preprocessing"]["scaling"]
    assert "static_standard_scaler" in all_plugins["preprocessing"]["scaling"]
    assert "ts_standard_scaler" in all_plugins["preprocessing"]["scaling"]
    # ---
    assert "ts_coxph" in all_plugins["time_to_event"]
    assert "ts_xgb" in all_plugins["time_to_event"]
    assert "dynamic_deephit" in all_plugins["time_to_event"]
    # ---
    assert "crn_classifier" in all_plugins["treatments"]
    assert "crn_regressor" in all_plugins["treatments"]
    assert "synctwin_regressor" in all_plugins["treatments"]
