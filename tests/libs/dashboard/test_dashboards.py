from libs.dashboard import (
    BaseDashboard,
    EvaluationDashboard,
    GenericDashboard,
    GraphTile,
    InferenceDashboard,
    TrainingDashboard,
)


def test_base_dashboard_has_one_basic_tile():
    dashboard = BaseDashboard()
    assert list(dashboard.components()) == ["summary"]


def test_subclasses_inherit_summary_and_add_their_own():
    assert issubclass(GenericDashboard, BaseDashboard)
    generic = GenericDashboard()
    assert set(generic.components()) == {"summary", "values", "log"}

    training = TrainingDashboard()
    assert {"summary", "values", "learning_rate", "log"} == set(training.components())


def test_all_dashboards_share_the_common_components():
    for dashboard in (
        GenericDashboard(),
        TrainingDashboard(),
        InferenceDashboard(),
        EvaluationDashboard(),
    ):
        assert {"summary", "values", "log"} <= set(dashboard.components())


def test_custom_subclass_defines_its_own_layout():
    class CustomDashboard(BaseDashboard):
        name = "custom"

        def _build(self) -> None:
            self._add("extra", GraphTile("t", "v"))

    dashboard = CustomDashboard()
    assert set(dashboard.components()) == {"summary", "extra"}
    dashboard.tile("extra").update(0, 1)
    dashboard.tile("extra").update(1, 2)
    assert "polyline" in dashboard.render(0)


def test_tile_access_and_render_contains_run_id():
    dashboard = GenericDashboard()
    dashboard.set_run_id("run-9")
    dashboard.tile("summary").update("k", "v")
    assert "run-9" in dashboard.render(0)
