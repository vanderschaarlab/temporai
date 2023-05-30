# pylint: disable=unused-argument

from unittest.mock import Mock

import pytest

import tempor.plugins.core._plugin as plugin_core
from tempor.plugins import plugin_loader


class TestPluginClass:
    def test_plugin_init_fails_no_name(self):
        class NamelessPlugin(plugin_core.Plugin):
            category = "category_was_set"

        with pytest.raises(ValueError, match=".*name.*not set.*"):
            NamelessPlugin()

    def test_plugin_init_fails_no_category(self):
        class CategorylessPlugin(plugin_core.Plugin):
            name = "name_was_set"

        with pytest.raises(ValueError, match=".*category.*not set.*"):
            CategorylessPlugin()


class TestRegistration:
    @pytest.fixture(autouse=True)
    def always_patch_module_in_this_test_class(self, patch_plugins_core_module):
        pass

    def test_register_plugin_and_category_success(self):
        plugin_core.register_plugin_category("dummy_category", expected_class=plugin_core.Plugin)

        @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category")
        class DummyPlugin(plugin_core.Plugin):
            pass

        plugin = DummyPlugin()

        assert isinstance(plugin, plugin_core.Plugin)
        assert plugin.name == "dummy_plugin"
        assert plugin.category == "dummy_category"

        fqn = plugin.fqn()

        assert fqn == "dummy_category.dummy_plugin"

        assert "dummy_category" in plugin_core.PLUGIN_CATEGORY_REGISTRY
        assert fqn in plugin_core.PLUGIN_REGISTRY

        assert plugin_core.PLUGIN_CATEGORY_REGISTRY["dummy_category"] == plugin_core.Plugin
        assert plugin_core.PLUGIN_REGISTRY[fqn] == DummyPlugin

    def test_register_multiple_plugins_and_categories_success(self):
        plugin_core.register_plugin_category("dummy_category_A", expected_class=plugin_core.Plugin)

        @plugin_core.register_plugin(name="dummy_plugin_A1", category="dummy_category_A")
        class DummyPluginA1(plugin_core.Plugin):
            pass

        @plugin_core.register_plugin(name="dummy_plugin_A2", category="dummy_category_A")
        class DummyPluginA2(plugin_core.Plugin):
            pass

        plugin_core.register_plugin_category("dummy_category_B", expected_class=plugin_core.Plugin)

        @plugin_core.register_plugin(name="dummy_plugin_B1", category="dummy_category_B")
        class DummyPluginB1(plugin_core.Plugin):
            pass

        p_a1 = DummyPluginA1()
        p_a2 = DummyPluginA2()
        p_b1 = DummyPluginB1()

        assert plugin_core.PLUGIN_CATEGORY_REGISTRY["dummy_category_A"] == plugin_core.Plugin
        assert plugin_core.PLUGIN_CATEGORY_REGISTRY["dummy_category_B"] == plugin_core.Plugin
        assert plugin_core.PLUGIN_REGISTRY[p_a1.fqn()] == DummyPluginA1
        assert plugin_core.PLUGIN_REGISTRY[p_a2.fqn()] == DummyPluginA2
        assert plugin_core.PLUGIN_REGISTRY[p_b1.fqn()] == DummyPluginB1

    def test_category_registration_reimport_allowed(self):
        plugin_core.register_plugin_category("dummy_category", expected_class=plugin_core.Plugin)

        @plugin_core.register_plugin(name="dummy_plugin_A", category="dummy_category")
        class DummyPluginA(plugin_core.Plugin):  # type: ignore  # pylint: disable=unused-variable
            pass

        @plugin_core.register_plugin(name="dummy_plugin_A", category="dummy_category")
        # pylint: disable-next=function-redefined
        class DummyPluginA(plugin_core.Plugin):  # type: ignore  # pyright: ignore  # noqa: F811
            pass

    def test_category_registration_fails_duplicate(self):
        plugin_core.register_plugin_category("same_category", expected_class=plugin_core.Plugin)

        with pytest.raises(TypeError, match=".*category.*already registered.*"):
            plugin_core.register_plugin_category("same_category", expected_class=plugin_core.Plugin)

    def test_category_registration_fails_not_plugin(self):
        class DoesNotInheritPlugin:
            pass

        with pytest.raises(TypeError, match=".*subclass.*Plugin.*"):
            plugin_core.register_plugin_category("dummy_category", expected_class=DoesNotInheritPlugin)

    def test_plugin_registration_fails_unknown_category(self):
        with pytest.raises(TypeError, match=".*category.*not.*registered.*"):

            @plugin_core.register_plugin(name="dummy_plugin", category="category_not_registered")
            class DummyPlugin(plugin_core.Plugin):  # pylint: disable=unused-variable
                pass

    def test_plugin_registration_fails_not_plugin(self):
        plugin_core.register_plugin_category("dummy_category", expected_class=plugin_core.Plugin)

        with pytest.raises(TypeError, match=".*[Ee]xpected.*subclass.*Plugin.*"):

            @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category")  # type: ignore
            class DoesNotInheritPlugin:  # pylint: disable=unused-variable
                pass

    def test_plugin_registration_fails_wrong_class_for_category(self):
        class ExpectedClassForDummyCategory(plugin_core.Plugin):
            pass

        plugin_core.register_plugin_category("dummy_category", expected_class=ExpectedClassForDummyCategory)

        with pytest.raises(TypeError, match=".*[Ee]xpected.*subclass.*ExpectedClassForDummyCategory.*"):

            @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category")
            class UnexpectedClassPlugin(plugin_core.Plugin):  # pylint: disable=unused-variable
                pass

    def test_plugin_registration_fails_duplicate_fqn(
        self,
    ):
        plugin_core.register_plugin_category("dummy_category", expected_class=plugin_core.Plugin)

        @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category")
        class DummyPlugin(plugin_core.Plugin):  # pylint: disable=unused-variable
            pass

        with pytest.raises(TypeError, match=".*[Pp]lugin with fully-qualified name.*already registered.*"):

            @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category")
            class DummyPluginDuplicate(plugin_core.Plugin):  # pylint: disable=unused-variable
                pass

    def test_plugin_registration_fails_duplicate_name(
        self,
    ):
        plugin_core.register_plugin_category("dummy_category", expected_class=plugin_core.Plugin)
        plugin_core.register_plugin_category("dummy_category_2", expected_class=plugin_core.Plugin)

        @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category")
        class DummyPlugin(plugin_core.Plugin):  # pylint: disable=unused-variable
            pass

        with pytest.raises(TypeError, match=".*[Pp]lugin with name.*already registered.*"):

            @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category_2")
            class DummyPluginDuplicate(plugin_core.Plugin):  # pylint: disable=unused-variable
                pass


category_a = "category_a"
category_b_x = "category_b.x"
category_b_y = "category_b.y"
CategoryAExpectedClass = Mock()
CategoryBXExpectedClass = Mock()
CategoryBYExpectedClass = Mock()
PluginA1 = Mock(category=category_a)
PluginA1.name = "plugin_a1"
PluginA2 = Mock(category=category_a)
PluginA2.name = "plugin_a2"
PluginBX1 = Mock(category=category_b_x)
PluginBX1.name = "plugin_bx1"
PluginBX2 = Mock(category=category_b_x)
PluginBX2.name = "plugin_bx2"
PluginBY1 = Mock(category=category_b_y)
PluginBY1.name = "plugin_by1"
PluginBY2 = Mock(category=category_b_y)
PluginBY2.name = "plugin_by2"


class TestPluginLoader:
    @pytest.fixture
    def always_patch_module_in_this_test_class(self, patch_plugins_core_module):
        pass

    @pytest.fixture(autouse=True)
    def pre_fill_registry(self, always_patch_module_in_this_test_class):
        plugin_core.PLUGIN_CATEGORY_REGISTRY = {
            category_a: CategoryAExpectedClass,
            category_b_x: CategoryBXExpectedClass,
            category_b_y: CategoryBYExpectedClass,
        }
        plugin_core.PLUGIN_REGISTRY = {
            f"{PluginA1.category}.{PluginA1.name}": PluginA1,
            f"{PluginA2.category}.{PluginA2.name}": PluginA2,
            f"{PluginBX1.category}.{PluginBX1.name}": PluginBX1,
            f"{PluginBX2.category}.{PluginBX2.name}": PluginBX2,
            f"{PluginBY1.category}.{PluginBY1.name}": PluginBY1,
            f"{PluginBY2.category}.{PluginBY2.name}": PluginBY2,
        }

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_list(self, loader):
        listed = loader.list()

        assert listed == {
            "category_a": ["plugin_a1", "plugin_a2"],
            "category_b": {
                "x": ["plugin_bx1", "plugin_bx2"],
                "y": ["plugin_by1", "plugin_by2"],
            },
        }

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_list_classes(self, loader):
        listed = loader.list_classes()

        assert listed == {
            "category_a": [PluginA1, PluginA2],
            "category_b": {
                "x": [PluginBX1, PluginBX2],
                "y": [PluginBY1, PluginBY2],
            },
        }

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_list_fqns(self, loader):
        listed = loader.list_fqns()

        assert listed == [
            "category_a.plugin_a1",
            "category_a.plugin_a2",
            "category_b.x.plugin_bx1",
            "category_b.x.plugin_bx2",
            "category_b.y.plugin_by1",
            "category_b.y.plugin_by2",
        ]

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_list_categories(self, loader):
        listed = loader.list_categories()

        assert listed == {
            "category_a": CategoryAExpectedClass,
            "category_b.x": CategoryBXExpectedClass,
            "category_b.y": CategoryBYExpectedClass,
        }

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_get(self, loader):
        plugin_a2_instance = loader.get(  # pylint: disable=unused-variable  # noqa: F841
            "category_a.plugin_a2", "arg", kwarg="kwarg"
        )
        PluginA2.assert_called_once_with("arg", kwarg="kwarg")
        PluginA2.reset_mock()

        plugin_by2_instance = loader.get(  # pylint: disable=unused-variable  # noqa: F841
            name="category_b.y.plugin_by2", kwarg="kwarg"
        )
        PluginBY2.assert_called_once_with(kwarg="kwarg")
        PluginBY2.reset_mock()

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_get_class(self, loader):
        plugin_a2_class = loader.get_class("category_a.plugin_a2")
        assert plugin_a2_class == PluginA2

        plugin_bx1_class = loader.get_class("category_b.x.plugin_bx1")
        assert plugin_bx1_class == PluginBX1

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_get_fails_no_such_plugin(self, loader):
        with pytest.raises(ValueError, match=".*[Pp]lugin.*not.*exist.*"):
            loader.get_class("category_a.no_such_plugin")

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_plugin_added_live(self, loader: plugin_core.PluginLoader):
        plugin_core.register_plugin_category("category_c", expected_class=plugin_core.Plugin)

        @plugin_core.register_plugin(name="plugin_c1", category="category_c")
        class PluginC1(plugin_core.Plugin):
            pass

        listed = loader.list()
        assert "category_c" in listed
        assert "plugin_c1" in listed["category_c"]

        listed_classes = loader.list_classes()
        assert "category_c" in listed
        assert PluginC1 in listed_classes["category_c"]

        plugin_c1_class = loader.get_class("category_c.plugin_c1")
        assert plugin_c1_class == PluginC1

        plugin_c1_instance = loader.get(
            "category_c.plugin_c1",
        )
        assert isinstance(plugin_c1_instance, PluginC1)  # type: ignore
