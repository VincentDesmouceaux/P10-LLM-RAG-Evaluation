import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from sql_tool import nba_sql_tool
from utils.structured_answer import generate_structured_answer
from utils.vector_store import VectorStoreManager


OLLAMA_MODEL = "qwen2.5:7b-instruct"


ROUTER_SYSTEM_PROMPT = """
Tu es le routeur d'un assistant NBA hybride.

Tu disposes d'un outil SQL nommé nba_sql_tool.

UTILISE nba_sql_tool lorsque la question demande une information
quantitative, statistique, numérique, comparative ou agrégée
présente dans la base NBA.

Exemples SQL :
- meilleur scoreur
- nombre de points
- pourcentage à 3 points
- moyenne ou classement
- combien de joueurs
- comparaison entre équipes
- statistiques d'un joueur

N'UTILISE PAS le tool SQL lorsque la question concerne :
- commentaires Reddit
- opinions de fans
- discussions
- perceptions
- analyse textuelle
- contenu provenant des PDF Reddit

Dans ce cas, réponds sans appeler de tool.
"""


class HybridNBAAgent:
    def __init__(self) -> None:
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.0,
        )

        self.router = self.llm.bind_tools(
            [nba_sql_tool]
        )

        self.vector_store = VectorStoreManager()

    def _route(
        self,
        question: str,
    ):
        return self.router.invoke(
            [
                SystemMessage(
                    content=ROUTER_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=question
                ),
            ]
        )

    def _has_numeric_intent(
        self,
        question: str,
    ) -> bool:
        normalized = question.lower().strip()

        explicit_text_only = [
            "sans calculer",
            "sans faire de calcul",
            "sans statistiques",
            "sans stats",
            "juste savoir ce que racontent",
            "je veux juste savoir ce que racontent",
        ]

        if any(
            phrase in normalized
            for phrase in explicit_text_only
        ):
            return False

        numeric_patterns = [
            r"\bcombien\b",
            r"\bplus de points\b",
            r"\bmoins de points\b",
            r"\btotalis\w*\b",
            r"\bpourcentage\b",
            r"\bmoyenne\b",
            r"\bclassement\b",
            r"\btop\s*\d+\b",
            r"\bpoints\b",
            r"\brebonds?\b",
            r"\bpasses?\b",
            r"\bassists?\b",
            r"\btentatives?\b",
            r"\b3 points\b",
        ]

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in numeric_patterns
        )

    def _answer_from_sql(
        self,
        question: str,
    ) -> dict:
        tool_result = nba_sql_tool.invoke(
            {
                "question": question,
            }
        )

        payload = json.loads(
            tool_result
        )

        synthesis_prompt = f"""
Tu es un analyste NBA.

Réponds à la question uniquement à partir
du résultat SQL fourni.

QUESTION:
{question}

REQUÊTE SQL:
{payload["sql"]}

RÉSULTAT:
{json.dumps(
    payload["rows"],
    ensure_ascii=False,
    indent=2,
)}

RÈGLES:
- sois précis et concis ;
- n'invente aucune valeur ;
- ne cite que les données présentes ;
- si aucune ligne n'est retournée, indique-le clairement.
"""

        response = self.llm.invoke(
            synthesis_prompt
        )

        return {
            "route": "sql",
            "question": question,
            "answer": response.content.strip(),
            "sql": payload["sql"],
            "data": payload["rows"],
            "sources": [
                "SQLite NBA"
            ],
        }

    def _answer_from_rag(
        self,
        question: str,
    ) -> dict:
        results = self.vector_store.search(
            question,
            k=5,
        )

        if not results:
            return {
                "route": "rag",
                "question": question,
                "answer": (
                    "Aucun contexte pertinent "
                    "n'a été retrouvé."
                ),
                "sources": [],
                "retrieval": [],
            }

        context_parts = []
        sources = []

        for result in results:
            metadata = result.get(
                "metadata",
                {},
            )

            source = (
                metadata.get("filename")
                or metadata.get("source")
                or metadata.get("file_name")
                or "source inconnue"
            )

            sources.append(source)

            context_parts.append(
                (
                    f"SOURCE: {source}\n"
                    f"SCORE: {result['score']:.2f}%\n"
                    f"TEXTE:\n{result['text']}"
                )
            )

        context = "\n\n---\n\n".join(
            context_parts
        )

        structured_answer = (
            generate_structured_answer(
                question=question,
                context=context,
            )
        )

        return {
            "route": "rag",
            "question": question,
            "answer": structured_answer.answer,
            "sources": list(
                dict.fromkeys(sources)
            ),
            "retrieval": [
                {
                    "source": (
                        result.get(
                            "metadata",
                            {},
                        ).get("filename")
                        or result.get(
                            "metadata",
                            {},
                        ).get("source")
                        or "source inconnue"
                    ),
                    "score": result["score"],
                }
                for result in results
            ],
        }

    def ask(
        self,
        question: str,
    ) -> dict:
        if self._has_numeric_intent(
            question
        ):
            return self._answer_from_sql(
                question
            )

        routed_response = self._route(
            question
        )

        if routed_response.tool_calls:
            tool_call = (
                routed_response.tool_calls[0]
            )

            if (
                tool_call["name"]
                == "nba_sql_tool"
            ):
                return self._answer_from_sql(
                    question
                )

        return self._answer_from_rag(
            question
        )


if __name__ == "__main__":
    agent = HybridNBAAgent()

    questions = [
        (
            "Quel joueur a marqué "
            "le plus de points ?"
        ),
        (
            "Quels sont les commentaires "
            "des fans sur Reddit ?"
        ),
    ]

    for question in questions:
        print()
        print("=" * 100)
        print("QUESTION :", question)
        print("=" * 100)

        result = agent.ask(
            question
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
