from tempor.plugins.preprocessing.scaling.plugin_static_minmax_scaler import (
    StaticMinMaxScaler as plugin,
)


def test_hyperparam_sample():
    for repeat in range(10000):  # pylint: disable=unused-variable
        args = plugin.sample_hyperparameters()
        plugin(**args)
