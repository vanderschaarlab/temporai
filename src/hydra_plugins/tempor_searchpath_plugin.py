import hydra.core.config_search_path
import hydra.plugins.search_path_plugin


class TemporSearchPathPlugin(hydra.plugins.search_path_plugin.SearchPathPlugin):
    def manipulate_search_path(self, search_path: hydra.core.config_search_path.ConfigSearchPath) -> None:
        search_path.append(provider="tempor", path="pkg://tempor.config.conf")
