from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "evaluation_results/"
    "before_after_hybrid_comparison.csv"
)

OUTPUT_DIR = Path(
    "evaluation_results/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():
    df = pd.read_csv(INPUT_PATH)

    total = len(df)

    before_score = (
        df["before_correct"].sum()
        / total
        * 100
    )

    after_score = (
        df["after_correct"].sum()
        / total
        * 100
    )

    fig = plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        [
            "RAG seul",
            "RAG + SQL + Hybrid",
        ],
        [
            before_score,
            after_score,
        ],
    )

    plt.ylim(
        0,
        110,
    )

    plt.ylabel(
        "Taux de réponses correctes (%)"
    )

    plt.title(
        "Performance avant / après amélioration"
    )

    for index, value in enumerate(
        [
            before_score,
            after_score,
        ]
    ):
        plt.text(
            index,
            value + 2,
            f"{value:.0f}%",
            ha="center",
        )

    plt.tight_layout()

    accuracy_path = (
        OUTPUT_DIR
        / "before_after_accuracy.png"
    )

    fig.savefig(
        accuracy_path,
        dpi=150,
    )

    plt.close(fig)

    category_df = (
        df.groupby("category")[
            [
                "before_correct",
                "after_correct",
            ]
        ]
        .mean()
        .mul(100)
    )

    fig = plt.figure(
        figsize=(10, 6)
    )

    x = range(
        len(category_df)
    )

    width = 0.35

    plt.bar(
        [
            value - width / 2
            for value in x
        ],
        category_df[
            "before_correct"
        ],
        width=width,
        label="RAG seul",
    )

    plt.bar(
        [
            value + width / 2
            for value in x
        ],
        category_df[
            "after_correct"
        ],
        width=width,
        label="RAG + SQL + Hybrid",
    )

    plt.xticks(
        list(x),
        category_df.index,
        rotation=25,
        ha="right",
    )

    plt.ylim(
        0,
        110,
    )

    plt.ylabel(
        "Taux de réussite (%)"
    )

    plt.title(
        "Performance par catégorie de question"
    )

    plt.legend()

    plt.tight_layout()

    category_path = (
        OUTPUT_DIR
        / "before_after_by_category.png"
    )

    fig.savefig(
        category_path,
        dpi=150,
    )

    plt.close(fig)

    print(
        "Graphique global :",
        accuracy_path,
    )

    print(
        "Graphique catégories :",
        category_path,
    )

    print()
    print(
        f"RAG seul       : "
        f"{before_score:.1f}%"
    )

    print(
        f"RAG + SQL + Hybrid : "
        f"{after_score:.1f}%"
    )


if __name__ == "__main__":
    main()
