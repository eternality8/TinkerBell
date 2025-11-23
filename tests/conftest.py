"""Shared pytest fixtures."""

import importlib
import importlib.util


def _load_pytest():  # pragma: no cover - helper for dynamic import
    spec = importlib.util.find_spec("pytest")
    if spec is None:
        class _PytestShim:  # type: ignore
            def fixture(self, func=None, **_kwargs):
                if func is None:
                    def decorator(inner):
                        return inner

                    return decorator
                return func

        return _PytestShim()  # type: ignore
    return importlib.import_module("pytest")


pytest = _load_pytest()


@pytest.fixture
def sample_snapshot() -> dict:
    return {
        "text": "hello",
        "text_range": {"start": 0, "end": 5},
        "window": {"start": 0, "end": 5},
    }
