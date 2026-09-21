from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("evaluation_results")
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INITIAL_RAGAS = {
    "context_precision": 0.8375,
    "context_recall": 1.0000,
    "faithfulness": 0.6071,
    "answer_relevancy": 0.2804,
}


def load_final_ragas():
    path = (
        OUTPUT_DIR
        / "reddit_ragas_final.csv"
    )

    df = pd.read_csv(path)

    metrics = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]

    return {
        metric: float(
            df[metric].mean()
        )
        for metric in metrics
    }


def load_sql_results():
    path = (
        OUTPUT_DIR
        / "before_after_hybrid_comparison.csv"
    )

    df = pd.read_csv(path)

    total = len(df)

    before = (
        df["before_correct"].sum()
        / total
    )

    after = (
        df["after_correct"].sum()
        / total
    )

    return before, after


def main():
    final_ragas = load_final_ragas()

    sql_before, sql_after = (
        load_sql_results()
    )

    rag_rows = []

    for metric in INITIAL_RAGAS:
        before = INITIAL_RAGAS[
            metric
        ]

        after = final_ragas[
            metric
        ]

        rag_rows.append(
            {
                "evaluation": "RAG",
                "metric": metric,
                "before": before,
                "after": after,
                "delta": after - before,
            }
        )

    rag_df = pd.DataFrame(
        rag_rows
    )

    sql_df = pd.DataFrame(
        [
            {
                "evaluation": "SQL",
                "metric": "accuracy",
                "before": sql_before,
                "after": sql_after,
                "delta": (
                    sql_after
                    - sql_before
                ),
            }
        ]
    )

    summary = pd.concat(
        [
            rag_df,
            sql_df,
        ],
        ignore_index=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "final_evaluation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("=" * 90)
    print("SYNTHESE FINALE")
    print("=" * 90)

    print(
        summary.to_string(
            index=False
        )
    )

    metrics = rag_df[
        "metric"
    ].tolist()

    x = range(
        len(metrics)
    )

    width = 0.35

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        [
            value - width / 2
            for value in x
        ],
        rag_df["before"],
        width=width,
        label="Avant",
    )

    plt.bar(
        [
            value + width / 2
            for value in x
        ],
        rag_df["after"],
        width=width,
        label="Après",
    )

    plt.xticks(
        list(x),
        metrics,
        rotation=20,
        ha="right",
    )

    plt.ylim(
        0,
        1.1,
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Evaluation RAGAS avant / après"
    )

    plt.legend()

    plt.tight_layout()

    rag_figure = (
        FIGURE_DIR
        / "ragas_before_after.png"
    )

    fig.savefig(
        rag_figure,
        dpi=150,
    )

    plt.close(fig)

    fig = plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        [
            "RAG seul",
            "RAG + SQL Tool",
        ],
        [
            sql_before * 100,
            sql_after * 100,
        ],
    )

    plt.ylim(
        0,
        110,
    )

    plt.ylabel(
        "Réponses correctes (%)"
    )

    plt.title(
        "Questions numériques avant / après"
    )

    for index, value in enumerate(
        [
            sql_before * 100,
            sql_after * 100,
        ]
    ):
        plt.text(
            index,
            value + 2,
            f"{value:.0f}%",
            ha="center",
        )

    plt.tight_layout()

    sql_figure = (
        FIGURE_DIR
        / "sql_before_after.png"
    )

    fig.savefig(
        sql_figure,
        dpi=150,
    )

    plt.close(fig)

    print()
    print(
        "CSV :",
        summary_path,
    )

    print(
        "RAGAS figure :",
        rag_figure,
    )

    print(
        "SQL figure   :",
        sql_figure,
    )


if __name__ == "__main__":
    main()
