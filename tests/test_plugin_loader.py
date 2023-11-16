import pytest

from tempor import plugin_loader


def test_methods_contents():
    all_plugins = plugin_loader.list(plugin_type="method")

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
    assert "temporal" in all_plugins["preprocessing"]["encoding"]
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
    assert "ts_onehot_encoder" in all_plugins["preprocessing"]["encoding"]["temporal"]
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


def test_datasource_contents():
    all_plugins = plugin_loader.list(plugin_type="datasource")

    # Check categories:
    assert "prediction" in all_plugins
    assert "time_to_event" in all_plugins
    assert "treatments" in all_plugins

    # Check subcategories:
    assert "one_off" in all_plugins["prediction"]
    assert "temporal" in all_plugins["prediction"]
    assert "one_off" in all_plugins["treatments"]
    assert "temporal" in all_plugins["treatments"]

    # Check plugins:
    assert "sine" in all_plugins["prediction"]["one_off"]
    assert "google_stocks" in all_plugins["prediction"]["one_off"]
    assert "dummy_prediction" in all_plugins["prediction"]["temporal"]
    assert "uci_diabetes" in all_plugins["prediction"]["temporal"]
    assert "pbc" in all_plugins["time_to_event"]
    assert "pkpd" in all_plugins["treatments"]["one_off"]
    assert "dummy_treatments" in all_plugins["treatments"]["temporal"]


def test_metrics_contents():
    all_plugins = plugin_loader.list(plugin_type="metric")

    # Check categories:
    assert "prediction" in all_plugins
    assert "time_to_event" in all_plugins
    # assert "treatments" in all_plugins

    # Check subcategories:
    assert "one_off" in all_plugins["prediction"]
    # assert "temporal" in all_plugins["prediction"]
    # assert "one_off" in all_plugins["treatments"]
    # assert "temporal" in all_plugins["treatments"]

    # Check sub-subcategories:
    assert "classification" in all_plugins["prediction"]["one_off"]
    assert "regression" in all_plugins["prediction"]["one_off"]
    # assert "classification" in all_plugins["prediction"]["temporal"]
    # assert "regression" in all_plugins["prediction"]["temporal"]
    # assert "classification" in all_plugins["treatments"]["one_off"]
    # assert "regression" in all_plugins["treatments"]["one_off"]
    # assert "classification" in all_plugins["treatments"]["temporal"]
    # assert "regression" in all_plugins["treatments"]["temporal"]

    # Check plugins:
    assert "accuracy" in all_plugins["prediction"]["one_off"]["classification"]
    assert "f1_score_micro" in all_plugins["prediction"]["one_off"]["classification"]
    assert "f1_score_macro" in all_plugins["prediction"]["one_off"]["classification"]
    assert "f1_score_weighted" in all_plugins["prediction"]["one_off"]["classification"]
    assert "kappa" in all_plugins["prediction"]["one_off"]["classification"]
    assert "kappa_quadratic" in all_plugins["prediction"]["one_off"]["classification"]
    assert "recall_micro" in all_plugins["prediction"]["one_off"]["classification"]
    assert "recall_macro" in all_plugins["prediction"]["one_off"]["classification"]
    assert "recall_weighted" in all_plugins["prediction"]["one_off"]["classification"]
    assert "precision_micro" in all_plugins["prediction"]["one_off"]["classification"]
    assert "precision_macro" in all_plugins["prediction"]["one_off"]["classification"]
    assert "precision_weighted" in all_plugins["prediction"]["one_off"]["classification"]
    assert "mcc" in all_plugins["prediction"]["one_off"]["classification"]
    assert "aucprc" in all_plugins["prediction"]["one_off"]["classification"]
    assert "aucroc" in all_plugins["prediction"]["one_off"]["classification"]
    # ---
    assert "mse" in all_plugins["prediction"]["one_off"]["regression"]
    assert "mae" in all_plugins["prediction"]["one_off"]["regression"]
    assert "r2" in all_plugins["prediction"]["one_off"]["regression"]
    # ---
    assert "c_index" in all_plugins["time_to_event"]
    assert "brier_score" in all_plugins["time_to_event"]


def test_plugin_types():
    plugin_types = plugin_loader.list_plugin_types()

    assert "method" in plugin_types
    assert "datasource" in plugin_types
    assert "metric" in plugin_types


PLUGIN_FQNS = plugin_loader.list_full_names()


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.parametrize("plugin_fqn", PLUGIN_FQNS)
def test_init_success(plugin_fqn):
    PluginCls = plugin_loader.get_class(plugin_fqn)
    PluginCls()  # Should successfully initialize with all default params.
