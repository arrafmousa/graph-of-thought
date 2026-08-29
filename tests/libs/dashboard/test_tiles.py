import pytest

from libs.dashboard import GraphTile, TableTile


def test_table_tile_appends_rows_and_validates_width():
    tile = TableTile(["a", "b"])
    tile.update("1", "2")
    tile.update("3", "4")
    html = tile.render()
    assert html.count("<tr>") == 3  # header + two rows
    with pytest.raises(ValueError):
        tile.update("only-one")


def test_graph_tile_collects_points():
    tile = GraphTile("step", "score")
    tile.update(0, 1.0)
    tile.update(1, 2.0)
    assert "polyline" in tile.render()
    with pytest.raises(ValueError):
        tile.update(1)


def test_tile_notifies_listener_on_update():
    tile = GraphTile("x", "y")
    calls: list[int] = []
    tile.set_listener(lambda: calls.append(1))
    tile.update(0, 0)
    tile.update(1, 1)
    assert len(calls) == 2
