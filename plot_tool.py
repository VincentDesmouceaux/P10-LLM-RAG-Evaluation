from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Type

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class PlotInput(BaseModel):
    data: list[dict[str, Any]] = Field(
        ...,
        description="Données structurées à représenter.",
    )
    chart_type: Literal["bar", "line", "pie"] = Field(
        ...,
        description="Type de graphique : bar, line ou pie.",
    )
    x: str = Field(
        ...,
        description="Clé utilisée pour l'axe X ou les labels.",
    )
    y: str = Field(
        ...,
        description="Clé numérique utilisée pour l'axe Y.",
    )
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None


class PlotTool(BaseTool):
    name: str = "plot_data"
    description: str = (
        "Génère un graphique matplotlib à partir de données structurées. "
        "Supporte les graphiques bar, line et pie."
    )
    args_schema: Type[BaseModel] = PlotInput

    def _run(
        self,
        data: list[dict[str, Any]],
        chart_type: str,
        x: str,
        y: str,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> str:
        if not data:
            raise ValueError(
                "Les données du graphique sont vides."
            )

        if any(x not in row or y not in row for row in data):
            raise ValueError(
                f"Les clés '{x}' et '{y}' doivent exister "
                "dans toutes les lignes."
            )

        labels = [str(row[x]) for row in data]

        try:
            values = [float(row[y]) for row in data]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"La colonne '{y}' doit contenir "
                "des valeurs numériques."
            ) from exc

        output_dir = (
            Path(__file__).resolve().parent
            / "generated_plots"
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        try:
            if chart_type == "bar":
                ax.bar(
                    labels,
                    values,
                )

            elif chart_type == "line":
                ax.plot(
                    labels,
                    values,
                    marker="o",
                )

            elif chart_type == "pie":
                ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                )

            else:
                raise ValueError(
                    "Type de graphique non supporté : "
                    f"{chart_type}"
                )

            if title:
                ax.set_title(title)

            if chart_type != "pie":
                ax.set_xlabel(
                    xlabel or x
                )
                ax.set_ylabel(
                    ylabel or y
                )
                ax.tick_params(
                    axis="x",
                    rotation=30,
                )

            fig.tight_layout()

            with NamedTemporaryFile(
                prefix="nba_plot_",
                suffix=".png",
                dir=output_dir,
                delete=False,
            ) as tmp:
                output_path = Path(
                    tmp.name
                )

            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
            )

        finally:
            plt.close(fig)

        return str(output_path)