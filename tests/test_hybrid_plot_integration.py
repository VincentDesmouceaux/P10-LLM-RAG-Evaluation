from pathlib import Path

from sportsee.agent.hybrid_agent import HybridNBAAgent
from plot_tool import PlotTool
from sportsee.observability.logfire_config import configure_observability


def test_hybrid_agent_generates_plot_for_sql_request(
    monkeypatch,
):
    configure_observability()

    agent = object.__new__(
        HybridNBAAgent
    )

    agent.plot_tool = PlotTool()

    monkeypatch.setattr(
        agent,
        "_has_numeric_intent",
        lambda question: True,
    )

    monkeypatch.setattr(
        agent,
        "_has_textual_intent",
        lambda question: False,
    )

    monkeypatch.setattr(
        agent,
        "_has_plot_intent",
        lambda question: True,
    )

    monkeypatch.setattr(
        agent,
        "_answer_from_sql",
        lambda question: {
            "route": "sql",
            "question": question,
            "answer": "Top joueurs NBA.",
            "sql": (
                "SELECT name, pts "
                "FROM stats "
                "ORDER BY pts DESC "
                "LIMIT 2"
            ),
            "data": [
                {
                    "name": "Player A",
                    "pts": 2100,
                },
                {
                    "name": "Player B",
                    "pts": 1900,
                },
            ],
            "sources": [
                "SQLite NBA",
            ],
        },
    )

    result = agent.ask(
        (
            "Montre les meilleurs joueurs "
            "sous forme de graphique."
        )
    )

    plot_path = Path(
        result["plot_path"]
    )

    try:
        assert result["route"] == "sql"
        assert (
            result["visualization_requested"]
            is True
        )
        assert result["plot_type"] == "bar"
        assert result["plot_x"] == "name"
        assert result["plot_y"] == "pts"

        assert plot_path.exists()
        assert plot_path.suffix == ".png"
        assert plot_path.stat().st_size > 0
        assert (
            plot_path.parent.name
            == "generated_plots"
        )

    finally:
        plot_path.unlink(
            missing_ok=True
        )
