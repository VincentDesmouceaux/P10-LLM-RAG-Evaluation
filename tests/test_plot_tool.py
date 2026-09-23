from pathlib import Path

import pytest

from plot_tool import PlotTool


@pytest.fixture
def tool():
    return PlotTool()


@pytest.fixture
def sample_data():
    return [
        {"player": "Player A", "points": 2100},
        {"player": "Player B", "points": 1850},
        {"player": "Player C", "points": 1600},
    ]


@pytest.mark.parametrize(
    "chart_type",
    ["bar", "line", "pie"],
)
def test_plot_tool_generates_png(
    tool,
    sample_data,
    chart_type,
):
    path = Path(
        tool.invoke(
            {
                "data": sample_data,
                "chart_type": chart_type,
                "x": "player",
                "y": "points",
                "title": f"Test {chart_type}",
            }
        )
    )

    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0
    assert path.parent.name == "generated_plots"


def test_plot_tool_rejects_empty_data(tool):
    with pytest.raises(ValueError):
        tool.invoke(
            {
                "data": [],
                "chart_type": "bar",
                "x": "player",
                "y": "points",
            }
        )


def test_plot_tool_rejects_missing_key(tool):
    with pytest.raises(ValueError):
        tool.invoke(
            {
                "data": [
                    {"player": "Player A"},
                ],
                "chart_type": "bar",
                "x": "player",
                "y": "points",
            }
        )


def test_plot_tool_rejects_non_numeric_values(tool):
    with pytest.raises(ValueError):
        tool.invoke(
            {
                "data": [
                    {
                        "player": "Player A",
                        "points": "not-a-number",
                    }
                ],
                "chart_type": "bar",
                "x": "player",
                "y": "points",
            }
        )
