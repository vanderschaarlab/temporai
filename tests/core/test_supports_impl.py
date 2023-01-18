from typing import Dict, Tuple

import pytest

from tempor.core import supports_impl


class DummySupported:
    pass


dummy_supported_a = DummySupported()
dummy_supported_b = DummySupported()


class DummyImpl:
    pass


dummy_impl = DummyImpl()


def test_init_success():
    class TestSupportsImplementations(supports_impl.SupportsImplementations[DummySupported, DummyImpl]):
        @property
        def supports_implementations_for(self) -> Tuple[DummySupported, ...]:
            return (dummy_supported_a, dummy_supported_b)

        def _register_implementations(self) -> Dict[DummySupported, DummyImpl]:
            return {
                dummy_supported_a: dummy_impl,
                dummy_supported_b: dummy_impl,
            }

    _ = TestSupportsImplementations()


@pytest.mark.parametrize(
    "registered_implementations_for", [[dummy_supported_a], [dummy_supported_a, dummy_supported_b]]
)
def test_init_fails_wrong_implementations_registered(registered_implementations_for):
    class TestSupportsImplementations(supports_impl.SupportsImplementations[DummySupported, DummyImpl]):
        @property
        def supports_implementations_for(self) -> Tuple[DummySupported, ...]:
            return (dummy_supported_b,)

        def _register_implementations(self) -> Dict[DummySupported, DummyImpl]:
            return {k: dummy_impl for k in registered_implementations_for}

    with pytest.raises(TypeError, match=".*supported.*"):
        _ = TestSupportsImplementations()


def test_dispatch_success():
    class TestSupportsImplementations(supports_impl.SupportsImplementations[DummySupported, DummyImpl]):
        @property
        def supports_implementations_for(self) -> Tuple[DummySupported, ...]:
            return (dummy_supported_a,)

        def _register_implementations(self) -> Dict[DummySupported, DummyImpl]:
            return {
                dummy_supported_a: dummy_impl,
            }

    to_test = TestSupportsImplementations()
    impl = to_test.dispatch_to_implementation(dummy_supported_a)
    assert impl == dummy_impl


def test_dispatch_fails_not_registered():
    class TestSupportsImplementations(supports_impl.SupportsImplementations[DummySupported, DummyImpl]):
        @property
        def supports_implementations_for(self) -> Tuple[DummySupported, ...]:
            return (dummy_supported_a,)

        def _register_implementations(self) -> Dict[DummySupported, DummyImpl]:
            return {
                dummy_supported_a: dummy_impl,
            }

    to_test = TestSupportsImplementations()

    with pytest.raises(TypeError, match=".*implementation.*") as excinfo:
        to_test.dispatch_to_implementation(dummy_supported_b)
    assert "supported" not in str(excinfo.value)  # To differentiate with exception form the other test.


def test_str():
    impl_dict = {
        dummy_supported_a: dummy_impl,
        dummy_supported_b: dummy_impl,
    }

    class TestSupportsImplementations(supports_impl.SupportsImplementations[DummySupported, DummyImpl]):
        @property
        def supports_implementations_for(self) -> Tuple[DummySupported, ...]:
            return (dummy_supported_a, dummy_supported_b)

        def _register_implementations(self) -> Dict[DummySupported, DummyImpl]:
            return impl_dict

    to_test = TestSupportsImplementations()

    assert "_implementations" in str(to_test)
    assert f"{impl_dict}" in str(to_test)
