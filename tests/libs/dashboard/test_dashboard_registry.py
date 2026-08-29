import pytest

from libs.dashboard import BaseDashboard, DashboardRegistry


def test_create_returns_named_dashboard():
    registry = DashboardRegistry.with_builtin_dashboards()
    dashboard = registry.create("generic")
    assert isinstance(dashboard, BaseDashboard)
    assert dashboard.name == "generic"


def test_all_builtin_dashboards_share_common_components():
    registry = DashboardRegistry.with_builtin_dashboards()
    for name in ("generic", "training", "inference", "evaluation"):
        assert {"summary", "values", "log"} <= set(registry.create(name).components())


def test_unknown_dashboard_raises():
    with pytest.raises(KeyError):
        DashboardRegistry.with_builtin_dashboards().create("nope")
