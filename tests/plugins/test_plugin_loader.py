from tempor.plugins import plugin_loader


def test_tempor_plugin_loader_contents():
    all_plugins = plugin_loader.list()

    # Do some checks that expected plugins have been registered.
    # Update as plugins get added / reorganized.
    assert "preprocessing" in all_plugins
    assert "survival" in all_plugins
    assert "imputation" in all_plugins["preprocessing"]
    assert "scaling" in all_plugins["preprocessing"]
    assert "nop_imputer" in all_plugins["preprocessing"]["imputation"]
    assert "nop_scaler" in all_plugins["preprocessing"]["scaling"]
    assert "dynamic_deephit" in all_plugins["survival"]
