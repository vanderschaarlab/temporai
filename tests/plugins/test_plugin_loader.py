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
    assert "encoding" in all_plugins["preprocessing"]
    assert "imputation" in all_plugins["preprocessing"]
    assert "scaling" in all_plugins["preprocessing"]
    assert "nop" in all_plugins["preprocessing"]
    assert "one_off" in all_plugins["prediction"]
    assert "temporal" in all_plugins["prediction"]
    assert "one_off" in all_plugins["treatments"]
    assert "temporal" in all_plugins["treatments"]

    # Check sub-subcategories:
    assert "classification" in all_plugins["prediction"]["one_off"]
    assert "regression" in all_plugins["prediction"]["one_off"]
    assert "classification" in all_plugins["prediction"]["temporal"]
    assert "regression" in all_plugins["prediction"]["temporal"]
    assert "static" in all_plugins["preprocessing"]["encoding"]
    # assert "temporal" in all_plugins["preprocessing"]["encoding"]
    assert "static" in all_plugins["preprocessing"]["imputation"]
    assert "temporal" in all_plugins["preprocessing"]["imputation"]
    assert "static" in all_plugins["preprocessing"]["scaling"]
    assert "temporal" in all_plugins["preprocessing"]["scaling"]
    # assert "classification" in all_plugins["treatments"]["one_off"]
    assert "regression" in all_plugins["treatments"]["one_off"]
    assert "classification" in all_plugins["treatments"]["temporal"]
    assert "regression" in all_plugins["treatments"]["temporal"]

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
    assert "nop_transformer" in all_plugins["preprocessing"]["nop"]
    # ---
    assert "static_onehot_encoder" in all_plugins["preprocessing"]["encoding"]["static"]
    # assert "ts_onehot_encoder" in all_plugins["preprocessing"]["encoding"]["temporal"]
    # ---
    assert "static_tabular_imputer" in all_plugins["preprocessing"]["imputation"]["static"]
    assert "ts_tabular_imputer" in all_plugins["preprocessing"]["imputation"]["temporal"]
    assert "ffill" in all_plugins["preprocessing"]["imputation"]["temporal"]
    assert "bfill" in all_plugins["preprocessing"]["imputation"]["temporal"]
    # ---
    assert "static_minmax_scaler" in all_plugins["preprocessing"]["scaling"]["static"]
    assert "static_standard_scaler" in all_plugins["preprocessing"]["scaling"]["static"]
    assert "ts_minmax_scaler" in all_plugins["preprocessing"]["scaling"]["temporal"]
    assert "ts_standard_scaler" in all_plugins["preprocessing"]["scaling"]["temporal"]
    # ---
    assert "ts_coxph" in all_plugins["time_to_event"]
    assert "ts_xgb" in all_plugins["time_to_event"]
    assert "dynamic_deephit" in all_plugins["time_to_event"]
    # ---
    assert "crn_classifier" in all_plugins["treatments"]["temporal"]["classification"]
    assert "crn_regressor" in all_plugins["treatments"]["temporal"]["regression"]
    assert "synctwin_regressor" in all_plugins["treatments"]["one_off"]["regression"]
