import json
import re
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from sqlalchemy.exc import SQLAlchemyError

from sportsee.core.config import DATABASE_URL


SQL_TABLES = ["players", "matches", "stats", "reports"]
OLLAMA_MODEL = "qwen2.5:7b-instruct"


FEW_SHOT_EXAMPLES = """
QUESTION:
Quel joueur a marqué le plus de points ?

SQL:
SELECT
    p.name,
    p.team_code,
    s.pts
FROM players AS p
JOIN stats AS s
    ON s.player_id = p.player_id
ORDER BY s.pts DESC
LIMIT 1;


QUESTION:
Quelle équipe a marqué le plus de points ?

SQL:
SELECT
    team_code,
    team_name,
    total_points
FROM reports
ORDER BY total_points DESC
LIMIT 1;


QUESTION:
Quels sont les 5 meilleurs pourcentages à 3 points
parmi les joueurs ayant tenté au moins 100 tirs à 3 points ?

SQL:
SELECT
    p.name,
    p.team_code,
    s.three_p_pct,
    s.three_pa
FROM players AS p
JOIN stats AS s
    ON s.player_id = p.player_id
WHERE s.three_pa >= 100
ORDER BY s.three_p_pct DESC
LIMIT 5;


QUESTION:
Combien de joueurs jouent pour OKC ?

SQL:
SELECT
    COUNT(*) AS player_count
FROM players
WHERE team_code = 'OKC';


QUESTION:
Parmi MIA, OKC, LAC, BKN et ATL, quelle équipe
a marqué le plus de points, laquelle en a marqué
le moins et quelle est la différence ?

SQL:
WITH filtered AS (
    SELECT
        team_code,
        team_name,
        total_points
    FROM reports
    WHERE team_code IN (
        'MIA',
        'OKC',
        'LAC',
        'BKN',
        'ATL'
    )
),
max_team AS (
    SELECT
        team_code,
        team_name,
        total_points
    FROM filtered
    ORDER BY total_points DESC
    LIMIT 1
),
min_team AS (
    SELECT
        team_code,
        team_name,
        total_points
    FROM filtered
    ORDER BY total_points ASC
    LIMIT 1
)
SELECT
    max_team.team_code AS max_team_code,
    max_team.team_name AS max_team_name,
    max_team.total_points AS max_total_points,
    min_team.team_code AS min_team_code,
    min_team.team_name AS min_team_name,
    min_team.total_points AS min_total_points,
    (
        max_team.total_points
        - min_team.total_points
    ) AS difference
FROM max_team
CROSS JOIN min_team;
"""


FORBIDDEN_SQL_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "REPLACE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "VACUUM",
    "PRAGMA",
}


def get_sql_database() -> SQLDatabase:
    return SQLDatabase.from_uri(
        DATABASE_URL,
        include_tables=SQL_TABLES,
    )


class StructuredQuerySQLDatabaseTool(QuerySQLDatabaseTool):
    def _run(
        self,
        query: str,
        run_manager=None,
    ):
        return self.db._execute(
            query,
            fetch="all",
        )


def get_sql_query_tool() -> QuerySQLDatabaseTool:
    return StructuredQuerySQLDatabaseTool(
        db=get_sql_database()
    )


def get_database_schema() -> str:
    return get_sql_database().get_table_info()


def clean_generated_sql(
    content: str,
) -> str:
    content = content.strip()

    fenced_match = re.search(
        r"```(?:sql)?\s*(.*?)```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced_match:
        content = fenced_match.group(1).strip()
    else:
        content = re.sub(
            r"^```sql\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"^```\s*",
            "",
            content,
        )

        if "```" in content:
            content = content.split(
                "```",
                1,
            )[0].strip()

    match = re.search(
        r"(?is)\b(SELECT|WITH)\b.*",
        content,
    )

    if not match:
        raise ValueError(
            "Le modèle n'a pas généré "
            "de requête SQL exploitable."
        )

    sql = match.group(0).strip()

    if ";" in sql:
        sql = sql.split(
            ";",
            1,
        )[0].strip()

    return sql



def validate_readonly_sql(
    sql: str,
) -> None:
    normalized = re.sub(
        r"\s+",
        " ",
        sql,
    ).strip()

    upper_sql = normalized.upper()

    if not (
        upper_sql.startswith("SELECT ")
        or upper_sql.startswith("WITH ")
    ):
        raise ValueError(
            "Seules les requêtes SELECT "
            "sont autorisées."
        )

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(
            rf"\b{keyword}\b",
            upper_sql,
        ):
            raise ValueError(
                f"Mot-clé SQL interdit : "
                f"{keyword}"
            )

    if "--" in sql or "/*" in sql:
        raise ValueError(
            "Les commentaires SQL "
            "ne sont pas autorisés."
        )


def generate_sql(
    question: str,
) -> str:
    schema = get_database_schema()

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
    )

    prompt = f"""
Tu es un expert PostgreSQL spécialisé dans les données NBA.

Ta mission est de convertir la question utilisateur
en UNE requête SQL PostgreSQL en lecture seule.

RÈGLES:
- Retourne uniquement la requête SQL.
- Utilise uniquement SELECT ou WITH ... SELECT.
- N'invente aucune table ni colonne.
- N'utilise jamais INSERT, UPDATE, DELETE,
  DROP, ALTER, CREATE ou PRAGMA.
- Utilise LIMIT lorsque la question demande
  quelques meilleurs résultats.
- Les statistiques de joueurs sont dans stats.
- Le nom et l'équipe du joueur sont dans players.
- Les agrégats d'équipe sont disponibles dans reports.
- matches est actuellement vide car la source Excel
  ne contient pas de données match-par-match.
- Ne prétends donc pas calculer des statistiques
  sur les 5 derniers matchs.

SCHÉMA:
{schema}

EXEMPLES:
{FEW_SHOT_EXAMPLES}

QUESTION:
{question}

SQL:
"""

    response = llm.invoke(prompt)

    sql = clean_generated_sql(
        response.content
    )

    validate_readonly_sql(sql)

    return sql


def repair_sql(
    question: str,
    invalid_sql: str,
    error: str,
) -> str:
    schema = get_database_schema()

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
    )

    prompt = f"""
Tu dois corriger une requête PostgreSQL invalide.

QUESTION UTILISATEUR:
{question}

SCHÉMA:
{schema}

REQUÊTE INVALIDE:
{invalid_sql}

ERREUR POSTGRESQL:
{error}

RÈGLES:
- retourne UNE SEULE requête SQL ;
- retourne uniquement du SQL ;
- utilise uniquement SELECT ou WITH ... SELECT ;
- n'invente aucune table ni colonne ;
- conserve exactement l'intention de la question ;
- n'utilise aucun SQL de modification ;
- retourne un résultat tabulaire plat ;
- une sous-requête scalaire ne doit retourner
  qu'une seule colonne ;
- n'utilise jamais la syntaxe
  (SELECT ...).nom_de_colonne ;
- pour comparer un maximum et un minimum,
  préfère des CTE puis un CROSS JOIN.
"""

    response = llm.invoke(prompt)

    repaired_sql = clean_generated_sql(
        response.content
    )

    validate_readonly_sql(
        repaired_sql
    )

    return repaired_sql


def execute_sql(
    sql: str,
) -> dict:
    validate_readonly_sql(sql)

    query_tool = get_sql_query_tool()
    rows = query_tool.invoke(sql)

    if not isinstance(rows, list):
        raise RuntimeError(
            "Le SQL Tool LangChain n a pas retourné "
            "des lignes structurées."
        )

    rows = rows[:100]

    columns = (
        list(rows[0].keys())
        if rows
        else []
    )

    return {
        "columns": columns,
        "rows": [
            dict(row)
            for row in rows
        ],
    }


@tool
def nba_sql_tool(
    question: str,
) -> str:
    """
    Répond aux questions quantitatives sur les données NBA
    stockées dans PostgreSQL.

    Le tool convertit une question en SQL,
    vérifie que la requête est en lecture seule,
    l'exécute puis retourne les résultats.
    """

    sql = generate_sql(question)

    try:
        result = execute_sql(sql)

    except SQLAlchemyError as error:
        sql = repair_sql(
            question=question,
            invalid_sql=sql,
            error=str(error),
        )

        result = execute_sql(sql)

    payload = {
        "question": question,
        "sql": sql,
        "columns": result["columns"],
        "rows": result["rows"],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


if __name__ == "__main__":
    test_question = (
        "Quel joueur a marqué "
        "le plus de points ?"
    )

    print(
        nba_sql_tool.invoke(
            {
                "question": test_question,
            }
        )
    )
