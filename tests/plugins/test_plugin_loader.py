from tempor.plugins import plugin_loader


def test_tempor_plugin_loader_contents():
    all_plugins = plugin_loader.list()

    # Do some checks that expected plugins have been registered.
    # Update as plugins get added / reorganized.

    # Check categories:
    assert "classification" in all_plugins
    assert "preprocessing" in all_plugins
    assert "regression" in all_plugins
    assert "time_to_event" in all_plugins
    assert "treatments" in all_plugins

    # Check subcategories:
    assert "imputation" in all_plugins["preprocessing"]
    assert "scaling" in all_plugins["preprocessing"]

    # Check plugins:
    assert "cde_classifier" in all_plugins["classification"]
    assert "ode_classifier" in all_plugins["classification"]
    assert "nn_classifier" in all_plugins["classification"]
    assert "seq2seq_classifier" in all_plugins["classification"]
    assert "laplace_ode_classifier" in all_plugins["classification"]
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
    assert "laplace_ode_regressor" in all_plugins["regression"]
    assert "nn_regressor" in all_plugins["regression"]
    assert "ode_regressor" in all_plugins["regression"]
    assert "seq2seq_regressor" in all_plugins["regression"]
    assert "cde_regressor" in all_plugins["regression"]
    # ---
    assert "ts_coxph" in all_plugins["time_to_event"]
    assert "ts_xgb" in all_plugins["time_to_event"]
    assert "dynamic_deephit" in all_plugins["time_to_event"]
    # ---
    assert "crn_classifier" in all_plugins["treatments"]
    assert "crn_regressor" in all_plugins["treatments"]
    assert "synctwin_regressor" in all_plugins["treatments"]
