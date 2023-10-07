# pylint: disable=unused-argument,protected-access

from unittest.mock import Mock

import pytest

import tempor.plugins.core._plugin as plugin_core
from tempor.plugins import plugin_loader


class TestHelpers:
    def test_parse_plugin_type(self):
        plugin_type_normal = plugin_core.parse_plugin_type("my_plugin_type", must_be_one_of=None)
        assert plugin_type_normal == "my_plugin_type"

        plugin_type_default = plugin_core.parse_plugin_type(None, must_be_one_of=None)
        assert plugin_type_default == plugin_core.DEFAULT_PLUGIN_TYPE

        with pytest.raises(ValueError, match=".*all.*reserved.*"):
            plugin_core.parse_plugin_type("all", must_be_one_of=None)

        plugin_type_must_be_one_of_ok = plugin_core.parse_plugin_type("a", must_be_one_of=["a", "b", "c"])
        assert plugin_type_must_be_one_of_ok == "a"

        with pytest.raises(ValueError, match=".*one of.*"):
            plugin_core.parse_plugin_type("x", must_be_one_of=["a", "b", "c"])

    def test_create_fqn(self):
        fqn = plugin_core.create_fqn(suffix="my_category.my_subcategory.my_plugin", plugin_type="my_plugin_type")
        assert fqn == "[my_plugin_type].my_category.my_subcategory.my_plugin"

    def test_create_fqn_fails_plugin_type_none(self):
        with pytest.raises(ValueError, match=".*[Pp]lugin type.*None.*"):
            plugin_core.create_fqn(suffix="my_category.my_subcategory.my_plugin", plugin_type=None)

    def test_filter_list_by_plugin_type(self):
        lst = [
            "[my_plugin_type_A].my_category.my_subcategory.my_plugin_A1",
            "[my_plugin_type_A].my_category.my_subcategory.my_plugin_A2",
            "[my_plugin_type_B].my_category.my_subcategory.my_plugin_B1",
            "random.stuff",
        ]
        filtered_A = plugin_core.filter_list_by_plugin_type(lst, plugin_type="my_plugin_type_A")
        filtered_B = plugin_core.filter_list_by_plugin_type(lst, plugin_type="my_plugin_type_B")
        assert filtered_A == [
            "[my_plugin_type_A].my_category.my_subcategory.my_plugin_A1",
            "[my_plugin_type_A].my_category.my_subcategory.my_plugin_A2",
        ]
        assert filtered_B == ["[my_plugin_type_B].my_category.my_subcategory.my_plugin_B1"]

    def test_filter_dict_by_plugin_type(self):
        d = {
            "[my_plugin_type_A].my_category.my_subcategory.my_plugin_A1": "foo",
            "[my_plugin_type_A].my_category.my_subcategory.my_plugin_A2": "bar",
            "[my_plugin_type_B].my_category.my_subcategory.my_plugin_B1": "baz",
            "random.stuff": "qux",
        }
        filtered_A = plugin_core.filter_dict_by_plugin_type(d, plugin_type="my_plugin_type_A")
        filtered_B = plugin_core.filter_dict_by_plugin_type(d, plugin_type="my_plugin_type_B")
        assert filtered_A == {
            "[my_plugin_type_A].my_category.my_subcategory.my_plugin_A1": "foo",
            "[my_plugin_type_A].my_category.my_subcategory.my_plugin_A2": "bar",
        }
        assert filtered_B == {"[my_plugin_type_B].my_category.my_subcategory.my_plugin_B1": "baz"}

    def test_remove_plugin_type_from_fqn_success(self):
        plugin_fqn = "[my_plugin_type].my_category.my_subcategory.my_plugin"
        plugin_fqn_without_plugin_type = plugin_core.remove_plugin_type_from_fqn(plugin_fqn)
        assert plugin_fqn_without_plugin_type == "my_category.my_subcategory.my_plugin"

        category_fqn = "[my_plugin_type].my_category"
        category_fqn_without_plugin_type = plugin_core.remove_plugin_type_from_fqn(category_fqn)
        assert category_fqn_without_plugin_type == "my_category"

    def test_remove_plugin_type_from_fqn_fails_incorrect_format(self):
        fqn_incorrect_format = "/my_plugin_type>.my_category.my_subcategory.my_plugin"
        with pytest.raises(ValueError, match=".*FQN.*incorrect.*"):
            plugin_core.remove_plugin_type_from_fqn(fqn_incorrect_format)

    def test_get_plugin_type_from_fqn(self):
        plugin_fqn = "[my_plugin_type].my_category.my_subcategory.my_plugin"
        plugin_fqn_without_plugin_type = plugin_core.get_plugin_type_from_fqn(plugin_fqn)
        assert plugin_fqn_without_plugin_type == "my_plugin_type"

        category_fqn = "[my_plugin_type].my_category"
        category_fqn_without_plugin_type = plugin_core.get_plugin_type_from_fqn(category_fqn)
        assert category_fqn_without_plugin_type == "my_plugin_type"


class TestPluginClass:
    def test_init_success(self):
        class NamelessPlugin(plugin_core.Plugin):
            name = "my_name"
            category = "my_category"
            plugin_type = "my_plugin_type"

        p = NamelessPlugin()
        assert p.name == "my_name"
        assert p.category == "my_category"
        assert p.plugin_type == "my_plugin_type"

    def test_name_helpers(self):
        class NamelessPlugin(plugin_core.Plugin):
            name = "my_name"
            category = "my_category"
            plugin_type = "my_plugin_type"

        p = NamelessPlugin()

        # Full name of plugin as user sees it:
        assert p.full_name() == "my_category.my_name"
        # FQN of plugin internally:
        assert p._fqn() == "[my_plugin_type].my_category.my_name"
        # FQN of plugin's category internally:
        assert p._category_fqn() == "[my_plugin_type].my_category"

    def test_init_fails_no_name(self):
        class NamelessPlugin(plugin_core.Plugin):
            category = "category_was_set"
            plugin_type = "plugin_type_was_set"

        with pytest.raises(ValueError, match=".*name.*not set.*"):
            NamelessPlugin()

    def test_init_fails_no_category(self):
        class CategorylessPlugin(plugin_core.Plugin):
            name = "name_was_set"
            plugin_type = "plugin_type_was_set"

        with pytest.raises(ValueError, match=".*category.*not set.*"):
            CategorylessPlugin()

    def test_init_fails_no_plugin_type(self):
        class PluginTypelessPlugin(plugin_core.Plugin):
            name = "name_was_set"
            category = "category_was_set"

        with pytest.raises(ValueError, match=".*plugin_type.*not set.*"):
            PluginTypelessPlugin()


class TestRegistration:
    @pytest.fixture(autouse=True)
    def always_patch_module_in_this_test_class(self, patch_plugins_core_module):
        pass

    def test_default_plugin_type(self):
        assert plugin_core.DEFAULT_PLUGIN_TYPE == "method"

    def test_register_plugin_and_category_success(self):
        plugin_core.register_plugin_category(
            "dummy_category", expected_class=plugin_core.Plugin, plugin_type="dummy_type"
        )

        @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category", plugin_type="dummy_type")
        class DummyPlugin(plugin_core.Plugin):
            pass

        plugin = DummyPlugin()

        assert isinstance(plugin, plugin_core.Plugin)
        assert plugin.name == "dummy_plugin"
        assert plugin.category == "dummy_category"
        assert plugin.plugin_type == "dummy_type"

        full_name = plugin.full_name()
        fqn = plugin._fqn()

        assert full_name == "dummy_category.dummy_plugin"

        assert "[dummy_type].dummy_category" in plugin_core.PLUGIN_CATEGORY_REGISTRY
        assert fqn in plugin_core.PLUGIN_REGISTRY

        assert plugin_core.PLUGIN_CATEGORY_REGISTRY["[dummy_type].dummy_category"] == plugin_core.Plugin
        assert plugin_core.PLUGIN_REGISTRY[fqn] == DummyPlugin

    def test_register_multiple_plugins_and_categories_and_types_success(self):
        plugin_core.register_plugin_category(
            "dummy_category_A", expected_class=plugin_core.Plugin, plugin_type="dummy_type_X"
        )

        @plugin_core.register_plugin(name="dummy_plugin_A1", category="dummy_category_A", plugin_type="dummy_type_X")
        class DummyXPluginA1(plugin_core.Plugin):
            pass

        @plugin_core.register_plugin(name="dummy_plugin_A2", category="dummy_category_A", plugin_type="dummy_type_X")
        class DummyXPluginA2(plugin_core.Plugin):
            pass

        plugin_core.register_plugin_category(
            "dummy_category_B", expected_class=plugin_core.Plugin, plugin_type="dummy_type_X"
        )

        @plugin_core.register_plugin(name="dummy_plugin_B1", category="dummy_category_B", plugin_type="dummy_type_X")
        class DummyXPluginB1(plugin_core.Plugin):
            pass

        plugin_core.register_plugin_category(
            "dummy_category_C", expected_class=plugin_core.Plugin, plugin_type="dummy_type_Y"
        )

        @plugin_core.register_plugin(name="dummy_plugin_C1", category="dummy_category_C", plugin_type="dummy_type_Y")
        class DummyYPluginC1(plugin_core.Plugin):
            pass

        pX_a1 = DummyXPluginA1()
        pX_a2 = DummyXPluginA2()
        pX_b1 = DummyXPluginB1()
        pY_c1 = DummyYPluginC1()

        assert plugin_core.PLUGIN_CATEGORY_REGISTRY["[dummy_type_X].dummy_category_A"] == plugin_core.Plugin
        assert plugin_core.PLUGIN_CATEGORY_REGISTRY["[dummy_type_X].dummy_category_B"] == plugin_core.Plugin
        assert plugin_core.PLUGIN_CATEGORY_REGISTRY["[dummy_type_Y].dummy_category_C"] == plugin_core.Plugin
        assert plugin_core.PLUGIN_REGISTRY[pX_a1._fqn()] == DummyXPluginA1
        assert plugin_core.PLUGIN_REGISTRY[pX_a2._fqn()] == DummyXPluginA2
        assert plugin_core.PLUGIN_REGISTRY[pX_b1._fqn()] == DummyXPluginB1
        assert plugin_core.PLUGIN_REGISTRY[pY_c1._fqn()] == DummyYPluginC1

    def test_category_registration_reimport_allowed(self):
        plugin_core.register_plugin_category(
            "dummy_category", expected_class=plugin_core.Plugin, plugin_type="dummy_type"
        )

        @plugin_core.register_plugin(name="dummy_plugin_A", category="dummy_category", plugin_type="dummy_type")
        class DummyPluginA(plugin_core.Plugin):  # type: ignore  # pylint: disable=unused-variable
            pass

        @plugin_core.register_plugin(name="dummy_plugin_A", category="dummy_category", plugin_type="dummy_type")
        # pylint: disable-next=function-redefined
        class DummyPluginA(plugin_core.Plugin):  # type: ignore  # pyright: ignore  # noqa: F811
            pass

    def test_category_registration_fails_duplicate(self):
        plugin_core.register_plugin_category(
            "same_category", expected_class=plugin_core.Plugin, plugin_type="same_type"
        )

        with pytest.raises(TypeError, match=".*category.*already registered.*"):
            plugin_core.register_plugin_category(
                "same_category", expected_class=plugin_core.Plugin, plugin_type="same_type"
            )

    def test_diff_plugin_types_same_category_registration_ok(self):
        plugin_core.register_plugin_category(
            "same_category_name", expected_class=plugin_core.Plugin, plugin_type="dummy_type_X"
        )

        @plugin_core.register_plugin(name="dummy_plugin_A", category="same_category_name", plugin_type="dummy_type_X")
        class DummyPluginA(plugin_core.Plugin):  # type: ignore  # pylint: disable=unused-variable
            pass

        plugin_core.register_plugin_category(
            "same_category_name", expected_class=plugin_core.Plugin, plugin_type="dummy_type_Y"
        )

        @plugin_core.register_plugin(name="dummy_plugin_B", category="same_category_name", plugin_type="dummy_type_Y")
        # pylint: disable-next=function-redefined
        class DummyPluginA(plugin_core.Plugin):  # type: ignore  # pyright: ignore  # noqa: F811
            pass

    def test_category_registration_fails_not_plugin(self):
        class DoesNotInheritPlugin:
            pass

        with pytest.raises(TypeError, match=".*subclass.*Plugin.*"):
            plugin_core.register_plugin_category(
                "dummy_category", expected_class=DoesNotInheritPlugin, plugin_type="dummy_type"
            )

    def test_plugin_registration_fails_unknown_category(self):
        with pytest.raises(TypeError, match=".*category.*not.*registered.*"):

            @plugin_core.register_plugin(
                name="dummy_plugin", category="category_not_registered", plugin_type="dummy_type"
            )
            class DummyPlugin(plugin_core.Plugin):  # pylint: disable=unused-variable
                pass

        with pytest.raises(TypeError, match=".*category.*not.*registered.*"):
            plugin_core.register_plugin_category(
                "category_registered", expected_class=plugin_core.Plugin, plugin_type="but_different_type"
            )

            @plugin_core.register_plugin(
                name="dummy_plugin", category="category_registered", plugin_type="but_not_for_this_type"
            )
            class AnotherDummyPlugin(plugin_core.Plugin):  # pylint: disable=unused-variable
                pass

    def test_plugin_registration_fails_not_plugin(self):
        plugin_core.register_plugin_category(
            "dummy_category", expected_class=plugin_core.Plugin, plugin_type="dummy_type"
        )

        with pytest.raises(TypeError, match=".*[Ee]xpected.*subclass.*Plugin.*"):

            @plugin_core.register_plugin(
                name="dummy_plugin",
                category="dummy_category",
                plugin_type="dummy_type",
            )
            class DoesNotInheritPlugin:  # pylint: disable=unused-variable
                pass

    def test_plugin_registration_fails_wrong_class_for_category(self):
        class ExpectedClassForDummyCategory(plugin_core.Plugin):
            pass

        plugin_core.register_plugin_category(
            "dummy_category", expected_class=ExpectedClassForDummyCategory, plugin_type="dummy_type"
        )

        with pytest.raises(TypeError, match=".*[Ee]xpected.*subclass.*ExpectedClassForDummyCategory.*"):

            @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category", plugin_type="dummy_type")
            class UnexpectedClassPlugin(plugin_core.Plugin):  # pylint: disable=unused-variable
                pass

    def test_plugin_registration_fails_duplicate_fqn(
        self,
    ):
        plugin_core.register_plugin_category(
            "dummy_category", expected_class=plugin_core.Plugin, plugin_type="dummy_type"
        )

        @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category", plugin_type="dummy_type")
        class DummyPlugin(plugin_core.Plugin):  # pylint: disable=unused-variable
            pass

        with pytest.raises(TypeError, match=".*[Pp]lugin.*with full name.*already been registered.*"):

            @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category", plugin_type="dummy_type")
            class DummyPluginDuplicate(plugin_core.Plugin):  # pylint: disable=unused-variable
                pass

    def test_plugin_registration_fails_duplicate_name(
        self,
    ):
        # NOTE: At some point this restriction might be lifted, at which point remove this test.

        plugin_core.register_plugin_category(
            "dummy_category", expected_class=plugin_core.Plugin, plugin_type="dummy_type"
        )
        plugin_core.register_plugin_category(
            "dummy_category_2", expected_class=plugin_core.Plugin, plugin_type="dummy_type"
        )

        @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category", plugin_type="dummy_type")
        class DummyPlugin(plugin_core.Plugin):  # pylint: disable=unused-variable
            pass

        with pytest.raises(TypeError, match=".*[Pp]lugin.*with name.*already been registered.*"):

            @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category_2", plugin_type="dummy_type")
            class DummyPluginDuplicate(plugin_core.Plugin):  # pylint: disable=unused-variable
                pass

    def test_default_plugin_type_value(self):
        plugin_core.register_plugin_category("dummy_category", expected_class=plugin_core.Plugin)

        @plugin_core.register_plugin(name="dummy_plugin", category="dummy_category")
        class DummyPlugin(plugin_core.Plugin):
            pass

        plugin = DummyPlugin()

        assert plugin.plugin_type == plugin_core.DEFAULT_PLUGIN_TYPE
        assert plugin.full_name() == "dummy_category.dummy_plugin"
        assert plugin._fqn() == f"[{plugin_core.DEFAULT_PLUGIN_TYPE}].dummy_category.dummy_plugin"
        assert f"[{plugin_core.DEFAULT_PLUGIN_TYPE}].dummy_category" in plugin_core.PLUGIN_CATEGORY_REGISTRY


# Prepare things for TestPluginLoader. ---

plugin_type_foo = "plugin_type_foo"
plugin_type_bar = "plugin_type_bar"
category_a = "category_a"  # plugin_type_foo
category_b_x = "category_b.x"  # plugin_type_foo
category_b_y = "category_b.y"  # plugin_type_foo
category_c = "category_c"  # plugin_type_bar
PluginTypeFooCategoryAExpectedClass = Mock()
PluginTypeFooCategoryBXExpectedClass = Mock()
PluginTypeFooCategoryBYExpectedClass = Mock()
PluginTypeBarCategoryCExpectedClass = Mock()


class PluginFooA1(plugin_core.Plugin):
    name = "plugin_a1"
    category = category_a
    plugin_type = plugin_type_foo


MockPluginFooA1 = Mock(wraps=PluginFooA1)
MockPluginFooA1.name = PluginFooA1.name


class PluginFooA2(plugin_core.Plugin):
    name = "plugin_a2"
    category = category_a
    plugin_type = plugin_type_foo

    # For test_get():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


MockPluginFooA2 = Mock(wraps=PluginFooA2)
MockPluginFooA2.name = PluginFooA2.name


class PluginFooBX1(plugin_core.Plugin):
    name = "plugin_bx1"
    category = category_b_x
    plugin_type = plugin_type_foo


MockPluginFooBX1 = Mock(wraps=PluginFooBX1)
MockPluginFooBX1.name = PluginFooBX1.name


class PluginFooBX2(plugin_core.Plugin):
    name = "plugin_bx2"
    category = category_b_x
    plugin_type = plugin_type_foo


MockPluginFooBX2 = Mock(wraps=PluginFooBX2)
MockPluginFooBX2.name = PluginFooBX2.name


class PluginFooBY1(plugin_core.Plugin):
    name = "plugin_by1"
    category = category_b_y
    plugin_type = plugin_type_foo


MockPluginFooBY1 = Mock(wraps=PluginFooBY1)
MockPluginFooBY1.name = PluginFooBY1.name


class PluginFooBY2(plugin_core.Plugin):
    name = "plugin_by2"
    category = category_b_y
    plugin_type = plugin_type_foo

    # For test_get():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


MockPluginFooBY2 = Mock(wraps=PluginFooBY2)
MockPluginFooBY2.name = PluginFooBY2.name


class PluginBarC1(plugin_core.Plugin):
    name = "plugin_c1"
    category = category_c
    plugin_type = plugin_type_bar


MockPluginBarC1 = Mock(wraps=PluginBarC1)
MockPluginBarC1.name = PluginBarC1.name


class PluginBarC2(plugin_core.Plugin):
    name = "plugin_c2"
    category = category_c
    plugin_type = plugin_type_bar


MockPluginBarC2 = Mock(wraps=PluginBarC2)
MockPluginBarC2.name = PluginBarC2.name

# Prepare things for TestPluginLoader (end). ---


class TestPluginLoader:
    @pytest.fixture
    def always_patch_module_in_this_test_class(self, patch_plugins_core_module):
        pass

    @pytest.fixture(autouse=True)
    def pre_fill_registry(self, always_patch_module_in_this_test_class):
        plugin_core.PLUGIN_CATEGORY_REGISTRY = {
            f"[{plugin_type_foo}].{category_a}": PluginTypeFooCategoryAExpectedClass,
            f"[{plugin_type_foo}].{category_b_x}": PluginTypeFooCategoryBXExpectedClass,
            f"[{plugin_type_foo}].{category_b_y}": PluginTypeFooCategoryBYExpectedClass,
            f"[{plugin_type_bar}].{category_c}": PluginTypeBarCategoryCExpectedClass,
        }
        plugin_core.PLUGIN_REGISTRY = {
            f"[{PluginFooA1.plugin_type}].{PluginFooA1.category}.{PluginFooA1.name}": MockPluginFooA1,
            f"[{PluginFooA2.plugin_type}].{PluginFooA2.category}.{PluginFooA2.name}": MockPluginFooA2,
            f"[{PluginFooBX1.plugin_type}].{PluginFooBX1.category}.{PluginFooBX1.name}": MockPluginFooBX1,
            f"[{PluginFooBX2.plugin_type}].{PluginFooBX2.category}.{PluginFooBX2.name}": MockPluginFooBX2,
            f"[{PluginFooBY1.plugin_type}].{PluginFooBY1.category}.{PluginFooBY1.name}": MockPluginFooBY1,
            f"[{PluginFooBY2.plugin_type}].{PluginFooBY2.category}.{PluginFooBY2.name}": MockPluginFooBY2,
            f"[{PluginBarC1.plugin_type}].{PluginBarC1.category}.{PluginBarC1.name}": MockPluginBarC1,
            f"[{PluginBarC2.plugin_type}].{PluginBarC2.category}.{PluginBarC2.name}": MockPluginBarC2,
        }

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_list(self, loader):
        listed_foo = loader.list(plugin_type=plugin_type_foo)

        assert listed_foo == {
            "category_a": ["plugin_a1", "plugin_a2"],
            "category_b": {
                "x": ["plugin_bx1", "plugin_bx2"],
                "y": ["plugin_by1", "plugin_by2"],
            },
        }

        listed_bar = loader.list(plugin_type=plugin_type_bar)

        assert listed_bar == {
            "category_c": ["plugin_c1", "plugin_c2"],
        }

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_list_classes(self, loader):
        listed_foo = loader.list_classes(plugin_type=plugin_type_foo)

        assert listed_foo == {
            "category_a": [MockPluginFooA1, MockPluginFooA2],
            "category_b": {
                "x": [MockPluginFooBX1, MockPluginFooBX2],
                "y": [MockPluginFooBY1, MockPluginFooBY2],
            },
        }

        listed_bar = loader.list_classes(plugin_type=plugin_type_bar)

        assert listed_bar == {
            "category_c": [MockPluginBarC1, MockPluginBarC2],
        }

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_list_full_names(self, loader):
        listed_foo = loader.list_full_names(plugin_type=plugin_type_foo)

        assert listed_foo == [
            "category_a.plugin_a1",
            "category_a.plugin_a2",
            "category_b.x.plugin_bx1",
            "category_b.x.plugin_bx2",
            "category_b.y.plugin_by1",
            "category_b.y.plugin_by2",
        ]

        listed_bar = loader.list_full_names(plugin_type=plugin_type_bar)

        assert listed_bar == [
            "category_c.plugin_c1",
            "category_c.plugin_c2",
        ]

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_list_categories(self, loader):
        listed_foo = loader.list_categories(plugin_type=plugin_type_foo)

        assert listed_foo == {
            "category_a": PluginTypeFooCategoryAExpectedClass,
            "category_b.x": PluginTypeFooCategoryBXExpectedClass,
            "category_b.y": PluginTypeFooCategoryBYExpectedClass,
        }

        listed_bar = loader.list_categories(plugin_type=plugin_type_bar)

        assert listed_bar == {
            "category_c": PluginTypeBarCategoryCExpectedClass,
        }

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_list_plugin_types(self, loader):
        listed_types = loader.list_plugin_types()
        assert listed_types == [plugin_type_foo, plugin_type_bar]

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_get_plugin_type_provided_as_kwarg(self, loader):
        # Cases: plugin_type provided as kwarg:

        plugin_a2_instance = loader.get(  # pylint: disable=unused-variable  # noqa: F841
            "category_a.plugin_a2", "arg", 123, plugin_type="plugin_type_foo", kwarg1="kwarg1", kwarg2="kwarg2"
        )
        MockPluginFooA2.assert_called_once_with("arg", 123, kwarg1="kwarg1", kwarg2="kwarg2")
        MockPluginFooA2.reset_mock()

        plugin_by2_instance = loader.get(  # pylint: disable=unused-variable  # noqa: F841
            name="category_b.y.plugin_by2", plugin_type="plugin_type_foo", kwarg1="kwarg1", kwarg2="kwarg2"
        )
        MockPluginFooBY2.assert_called_once_with(kwarg1="kwarg1", kwarg2="kwarg2")
        MockPluginFooBY2.reset_mock()

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_get_plugin_type_provided_as_arg(self, loader):
        # Cases: plugin_type provided as positional arg:
        plugin_a2_instance = loader.get(  # pylint: disable=unused-variable  # noqa: F841
            "category_a.plugin_a2", "plugin_type_foo", "arg", 123, kwarg1="kwarg1", kwarg2="kwarg2"
        )
        MockPluginFooA2.assert_called_once_with("arg", 123, kwarg1="kwarg1", kwarg2="kwarg2")
        MockPluginFooA2.reset_mock()

        plugin_by2_instance = loader.get(  # pylint: disable=unused-variable  # noqa: F841
            "category_b.y.plugin_by2", "plugin_type_foo", "arg", 123, kwarg1="kwarg1", kwarg2="kwarg2"
        )
        MockPluginFooBY2.assert_called_once_with("arg", 123, kwarg1="kwarg1", kwarg2="kwarg2")
        MockPluginFooBY2.reset_mock()

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_get_class(self, loader):
        plugin_a2_class = loader.get_class("category_a.plugin_a2", "plugin_type_foo")
        assert plugin_a2_class == MockPluginFooA2

        plugin_bx1_class = loader.get_class("category_b.x.plugin_bx1", plugin_type="plugin_type_foo")
        assert plugin_bx1_class == MockPluginFooBX1

        plugin_a2_class = loader.get_class("category_c.plugin_c2", plugin_type="plugin_type_bar")
        assert plugin_a2_class == MockPluginBarC2

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_get_fails_no_such_plugin(self, loader):
        with pytest.raises(ValueError, match=".*[Pp]lugin.*not.*exist.*"):
            loader.get_class("category_a.no_such_plugin")

    @pytest.mark.parametrize("loader", [plugin_core.PluginLoader(), plugin_loader])
    def test_plugin_added_live(self, loader: plugin_core.PluginLoader):
        # Case: add plugin live in a new category.

        plugin_core.register_plugin_category(
            "category_d", expected_class=plugin_core.Plugin, plugin_type="plugin_type_bar"
        )

        @plugin_core.register_plugin(name="plugin_d1", category="category_d", plugin_type="plugin_type_bar")
        class PluginD1(plugin_core.Plugin):
            pass

        listed = loader.list(plugin_type="plugin_type_bar")
        assert "category_d" in listed
        assert "plugin_d1" in listed["category_d"]

        listed_classes = loader.list_classes(plugin_type="plugin_type_bar")
        assert "category_d" in listed
        assert PluginD1 in listed_classes["category_d"]

        plugin_d1_class = loader.get_class("category_d.plugin_d1", plugin_type="plugin_type_bar")
        assert plugin_d1_class == PluginD1

        plugin_d1_instance = loader.get("category_d.plugin_d1", "plugin_type_bar")
        assert isinstance(plugin_d1_instance, PluginD1)

        # Case: add plugin live in an existing category.

        @plugin_core.register_plugin(name="plugin_d2", category="category_d", plugin_type="plugin_type_bar")
        class PluginD2(plugin_core.Plugin):
            pass

        listed = loader.list(plugin_type="plugin_type_bar")
        assert "category_d" in listed
        assert "plugin_d2" in listed["category_d"]

        listed_classes = loader.list_classes(plugin_type="plugin_type_bar")
        assert "category_d" in listed
        assert PluginD2 in listed_classes["category_d"]

        plugin_d2_class = loader.get_class("category_d.plugin_d2", plugin_type="plugin_type_bar")
        assert plugin_d2_class == PluginD2

        plugin_d2_instance = loader.get("category_d.plugin_d2", "plugin_type_bar")
        assert isinstance(plugin_d2_instance, PluginD2)

        # Case: add plugin live in a new category which has a new plugin type.

        plugin_core.register_plugin_category(
            "category_e", expected_class=plugin_core.Plugin, plugin_type="plugin_type_baz"
        )

        @plugin_core.register_plugin(name="plugin_e1", category="category_e", plugin_type="plugin_type_baz")
        class PluginE1(plugin_core.Plugin):
            pass

        listed = loader.list(plugin_type="plugin_type_baz")
        assert "category_e" in listed
        assert "plugin_e1" in listed["category_e"]

        listed_classes = loader.list_classes(plugin_type="plugin_type_baz")
        assert "category_e" in listed
        assert PluginE1 in listed_classes["category_e"]

        plugin_e1_class = loader.get_class("category_e.plugin_e1", plugin_type="plugin_type_baz")
        assert plugin_e1_class == PluginE1

        plugin_e1_instance = loader.get("category_e.plugin_e1", "plugin_type_baz")
        assert isinstance(plugin_e1_instance, PluginE1)
