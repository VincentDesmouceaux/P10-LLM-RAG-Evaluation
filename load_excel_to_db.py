import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from sportsee.sql.schemas import (
    PlayerRecord,
    ReportRecord,
    StatRecord,
)


EXCEL_PATH = Path("inputs/regular NBA.xlsx")
DATABASE_PATH = Path("data/nba_rag.db")
SCHEMA_PATH = Path("db/schema.sql")


STAT_COLUMN_MAPPING = {
    "GP": "gp",
    "W": "wins",
    "L": "losses",
    "Min": "minutes",
    "PTS": "pts",
    "FGM": "fgm",
    "FGA": "fga",
    "FG%": "fg_pct",
    "3PM": "three_pm",
    "3PA": "three_pa",
    "3P%": "three_p_pct",
    "FTM": "ftm",
    "FTA": "fta",
    "FT%": "ft_pct",
    "OREB": "oreb",
    "DREB": "dreb",
    "REB": "reb",
    "AST": "ast",
    "TOV": "tov",
    "STL": "stl",
    "BLK": "blk",
    "PF": "pf",
    "FP": "fp",
    "DD2": "dd2",
    "TD3": "td3",
    "+ / -": "plus_minus",
    "OFFRTG": "offrtg",
    "DEFRTG": "defrtg",
    "NETRTG": "netrtg",
    "AST%": "ast_pct",
    "AST/TO": "ast_to",
    "AST RATIO": "ast_ratio",
    "OREB%": "oreb_pct",
    "DREB%": "dreb_pct",
    "REB%": "reb_pct",
    "TO RATIO": "to_ratio",
    "EFG%": "efg_pct",
    "TS%": "ts_pct",
    "USG%": "usg_pct",
    "PACE": "pace",
    "PIE": "pie",
    "POSS": "poss",
}


def find_header_row(
    raw_df: pd.DataFrame,
    required_values: set[str],
) -> int:
    for index, row in raw_df.iterrows():
        values = {
            str(value).strip()
            for value in row.dropna().tolist()
        }

        if required_values.issubset(values):
            return int(index)

    raise ValueError(
        f"En-tête introuvable : {required_values}"
    )


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    return value


def load_nba_dataframe() -> pd.DataFrame:
    raw = pd.read_excel(
        EXCEL_PATH,
        sheet_name="Données NBA",
        header=None,
    )

    header_row = find_header_row(
        raw,
        {"Player", "Team", "PTS"},
    )

    headers = [
        str(value).strip()
        if not pd.isna(value)
        else ""
        for value in raw.iloc[header_row]
    ]

    df = raw.iloc[
        header_row + 1:
    ].copy()

    df.columns = headers

    df = df.loc[
        :,
        [
            column
            for column in df.columns
            if column
        ],
    ]

    df = df.dropna(
        subset=["Player", "Team"]
    ).reset_index(drop=True)

    columns = list(df.columns)

    three_pa_index = columns.index("3PA")

    if three_pa_index > 0:
        possible_three_pm = columns[
            three_pa_index - 1
        ]

        if possible_three_pm != "3PM":
            columns[
                three_pa_index - 1
            ] = "3PM"

    df.columns = columns

    return df


def load_reports_dataframe() -> pd.DataFrame:
    raw = pd.read_excel(
        EXCEL_PATH,
        sheet_name="Analyse",
        header=None,
    )

    header_row = find_header_row(
        raw,
        {
            "Code",
            "Nom complet de l'équipe",
        },
    )

    headers = [
        str(value).strip()
        if not pd.isna(value)
        else ""
        for value in raw.iloc[header_row]
    ]

    df = raw.iloc[
        header_row + 1:
    ].copy()

    df.columns = headers

    useful_columns = [
        "Code",
        "Nom complet de l'équipe",
        "Nombre de joueur par équipe",
        "Nombre de point total par équipe",
    ]

    df = df[
        useful_columns
    ].copy()

    df = df.dropna(
        subset=[
            "Code",
            "Nom complet de l'équipe",
            "Nombre de joueur par équipe",
            "Nombre de point total par équipe",
        ]
    ).reset_index(drop=True)

    df["Code"] = (
        df["Code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[
        df["Code"].str.fullmatch(
            r"[A-Z]{3}"
        )
    ].reset_index(drop=True)

    return df


def create_database(
    connection: sqlite3.Connection,
) -> None:
    connection.executescript(
        SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )


def reset_tables(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        "DELETE FROM stats"
    )
    connection.execute(
        "DELETE FROM reports"
    )
    connection.execute(
        "DELETE FROM matches"
    )
    connection.execute(
        "DELETE FROM players"
    )


def insert_players_and_stats(
    connection: sqlite3.Connection,
    df: pd.DataFrame,
) -> tuple[int, int]:
    players_inserted = 0
    stats_inserted = 0

    for row_index, row in df.iterrows():
        try:
            player = PlayerRecord(
                name=str(
                    row["Player"]
                ),
                team_code=str(
                    row["Team"]
                ),
                age=int(
                    row["Age"]
                ),
            )

            stat_payload = {}

            for excel_column, field_name in (
                STAT_COLUMN_MAPPING.items()
            ):
                stat_payload[
                    field_name
                ] = clean_value(
                    row.get(
                        excel_column
                    )
                )

            stats = StatRecord(
                **stat_payload
            )

        except (
            ValidationError,
            ValueError,
            TypeError,
        ) as error:
            raise RuntimeError(
                "Validation Pydantic échouée "
                f"à la ligne NBA {row_index}: "
                f"{error}"
            ) from error

        cursor = connection.execute(
            """
            INSERT INTO players (
                name,
                team_code,
                age
            )
            VALUES (?, ?, ?)
            """,
            (
                player.name,
                player.team_code,
                player.age,
            ),
        )

        player_id = cursor.lastrowid

        stat_data = stats.model_dump()

        columns = [
            "player_id",
            "match_id",
            "stat_scope",
            *stat_data.keys(),
        ]

        values = [
            player_id,
            None,
            "season",
            *stat_data.values(),
        ]

        placeholders = ", ".join(
            ["?"] * len(columns)
        )

        connection.execute(
            f"""
            INSERT INTO stats (
                {", ".join(columns)}
            )
            VALUES (
                {placeholders}
            )
            """,
            values,
        )

        players_inserted += 1
        stats_inserted += 1

    return (
        players_inserted,
        stats_inserted,
    )


def insert_reports(
    connection: sqlite3.Connection,
    df: pd.DataFrame,
) -> int:
    reports_inserted = 0

    for row_index, row in df.iterrows():
        try:
            report = ReportRecord(
                team_code=str(
                    row["Code"]
                ),
                team_name=str(
                    row[
                        "Nom complet de l'équipe"
                    ]
                ),
                player_count=int(
                    row[
                        "Nombre de joueur par équipe"
                    ]
                ),
                total_points=int(
                    row[
                        "Nombre de point total par équipe"
                    ]
                ),
                source_sheet="Analyse",
            )

        except (
            ValidationError,
            ValueError,
            TypeError,
        ) as error:
            raise RuntimeError(
                "Validation Pydantic échouée "
                f"à la ligne Analyse {row_index}: "
                f"{error}"
            ) from error

        connection.execute(
            """
            INSERT INTO reports (
                team_code,
                team_name,
                player_count,
                total_points,
                source_sheet
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                report.team_code,
                report.team_name,
                report.player_count,
                report.total_points,
                report.source_sheet,
            ),
        )

        reports_inserted += 1

    return reports_inserted


def main() -> None:
    print("Chargement du fichier Excel...")

    nba_df = load_nba_dataframe()
    reports_df = load_reports_dataframe()

    print(
        f"Joueurs détectés : {len(nba_df)}"
    )
    print(
        f"Rapports détectés : {len(reports_df)}"
    )

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    try:
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        create_database(
            connection
        )

        reset_tables(
            connection
        )

        players_count, stats_count = (
            insert_players_and_stats(
                connection,
                nba_df,
            )
        )

        reports_count = insert_reports(
            connection,
            reports_df,
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    print()
    print(
        f"players : {players_count}"
    )
    print(
        f"stats   : {stats_count}"
    )
    print(
        f"reports : {reports_count}"
    )
    print(
        "matches : 0 "
        "(aucune donnée match-par-match)"
    )

    print()
    print("Ingestion SQLite : OK")


if __name__ == "__main__":
    main()
