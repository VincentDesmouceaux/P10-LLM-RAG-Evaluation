import json
import re

import logfire
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from plot_tool import PlotTool
from sql_tool import nba_sql_tool
from utils.observability import configure_observability
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
        configure_observability()

        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.0,
        )

        self.router = self.llm.bind_tools(
            [nba_sql_tool]
        )

        self.vector_store = VectorStoreManager()
        self.plot_tool = PlotTool()

    def _route(
        self,
        question: str,
    ):
        with logfire.span(
            "llm_router",
            question=question,
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
        """
        Détecte les questions nécessitant
        des données quantitatives ou statistiques.
        """
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

    def _has_textual_intent(
        self,
        question: str,
    ) -> bool:
        """
        Détecte une demande explicite d'analyse
        documentaire ou d'opinion.

        La simple présence des mots Reddit,
        commentaires ou fans ne suffit pas.
        """
        normalized = question.lower().strip()

        textual_patterns = [
            r"\bque disent\b",
            r"\bqu['’]en pensent\b",
            r"\bquels? sont les commentaires?\b",
            r"\bquels? commentaires?\b",
            r"\bcommentaires? (?:sur|à propos de|concernant)\b",
            r"\bavis (?:sur|à propos de|concernant)\b",
            r"\bopinions? (?:sur|à propos de|concernant)\b",
            r"\bperceptions? (?:sur|à propos de|concernant)\b",
            r"\bdiscussions? (?:sur|à propos de|concernant)\b",
            r"\bà son sujet\b",
            r"\ba son sujet\b",
            r"\bà leur sujet\b",
            r"\ba leur sujet\b",
            r"\best[- ]il décrit\b",
            r"\best[- ]elle décrite\b",
            r"\bcomment est[- ]il décrit\b",
            r"\bcomment est[- ]elle décrite\b",
        ]

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in textual_patterns
        )

    def _has_plot_intent(
        self,
        question: str,
    ) -> bool:
        """
        Détecte une demande explicite
        de visualisation graphique.
        """
        normalized = question.lower().strip()

        plot_patterns = [
            r"\bgraphique\b",
            r"\bgraphe\b",
            r"\bvisualis\w*\b",
            r"\bcourbe\b",
            r"\bhistogramme\b",
            r"\bdiagramme\b",
            r"\bcamembert\b",
            r"\bbarres?\b",
            r"\bplot\b",
            r"\bchart\b",
        ]

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern in plot_patterns
        )

    def _attach_plot(
        self,
        question: str,
        result: dict,
    ) -> dict:
        """
        Génère un graphique à partir des
        données structurées retournées par SQL.
        """
        data = result.get("data") or []

        if not data:
            result["visualization_requested"] = True
            result["plot_path"] = None
            return result

        first_row = data[0]

        categorical_keys = [
            key
            for key, value in first_row.items()
            if not isinstance(value, (int, float))
        ]

        numeric_keys = [
            key
            for key, value in first_row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]

        if not categorical_keys or not numeric_keys:
            result["visualization_requested"] = True
            result["plot_path"] = None
            return result

        x_key = categorical_keys[0]
        y_key = numeric_keys[0]

        normalized = question.lower()

        if "camembert" in normalized or "pie" in normalized:
            chart_type = "pie"
        elif "courbe" in normalized or "line" in normalized:
            chart_type = "line"
        else:
            chart_type = "bar"

        with logfire.span(
            "plot_generation",
            chart_type=chart_type,
            x=x_key,
            y=y_key,
        ):
            plot_path = self.plot_tool.invoke(
                {
                    "data": data,
                    "chart_type": chart_type,
                    "x": x_key,
                    "y": y_key,
                    "title": question,
                    "xlabel": x_key,
                    "ylabel": y_key,
                }
            )

        result["visualization_requested"] = True
        result["plot_path"] = plot_path
        result["plot_type"] = chart_type
        result["plot_x"] = x_key
        result["plot_y"] = y_key

        return result

    def _answer_from_sql(
        self,
        question: str,
    ) -> dict:
        """
        Exécute la partie SQL puis génère
        une réponse naturelle à partir des résultats.
        """
        with logfire.span(
            "sql_pipeline",
            question=question,
        ):
            with logfire.span(
                "sql_tool_execution"
            ):
                tool_result = nba_sql_tool.invoke(
                    {
                        "question": question,
                    }
                )

            payload = json.loads(
                tool_result
            )

            logfire.info(
                "sql_result",
                row_count=len(
                    payload["rows"]
                ),
                sql=payload["sql"],
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
- si aucune ligne n'est retournée, indique-le clairement ;
- ne génère jamais de lien, URL ou image Markdown ;
- si un graphique est demandé, fournis seulement la synthèse textuelle :
  le graphique réel est généré localement par PlotTool.
"""

            with logfire.span(
                "sql_answer_synthesis"
            ):
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
        """
        Recherche le contexte documentaire dans FAISS
        puis génère une réponse structurée.
        """
        with logfire.span(
            "rag_pipeline",
            question=question,
        ):
            with logfire.span(
                "rag_retrieval",
                top_k=5,
            ):
                results = self.vector_store.search(
                    question,
                    k=5,
                )

            logfire.info(
                "rag_retrieval_result",
                retrieval_count=len(results),
            )

            if not results:
                logfire.warn(
                    "rag_no_context",
                    question=question,
                )

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

            unique_sources = list(
                dict.fromkeys(sources)
            )

            logfire.info(
                "rag_context_ready",
                source_count=len(
                    unique_sources
                ),
                retrieval_count=len(results),
            )

            with logfire.span(
                "rag_answer_generation",
                source_count=len(
                    unique_sources
                ),
            ):
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
                "sources": unique_sources,
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
                            or result.get(
                                "metadata",
                                {},
                            ).get("file_name")
                            or "source inconnue"
                        ),
                        "score": result["score"],
                    }
                    for result in results
                ],
            }

    def _build_sql_subquestion(
        self,
        question: str,
    ) -> str:
        """
        Extrait uniquement la composante statistique
        d'une question nécessitant SQL + RAG.
        """
        prompt = f"""
Tu dois extraire uniquement la partie statistique
ou numérique de la question suivante.

QUESTION ORIGINALE:
{question}

RÈGLES:
- conserve les noms de joueurs ou d'équipes utiles ;
- conserve les comparaisons et agrégations numériques ;
- supprime toute demande concernant Reddit,
  les commentaires, les fans, les discussions
  ou les opinions ;
- ne réponds pas à la question ;
- retourne uniquement une question destinée
  à une base SQL ;
- n'ajoute aucune explication.

EXEMPLE:

Question originale :
Quel joueur a marqué le plus de points et que disent
les discussions Reddit à son sujet ?

Réponse :
Quel joueur a marqué le plus de points ?
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content.strip()

    def _build_rag_subquestion(
        self,
        original_question: str,
        sql_answer: str,
    ) -> str:
        """
        Construit la question documentaire après
        l'exécution SQL afin d'utiliser explicitement
        l'entité trouvée par la base de données.
        """
        prompt = f"""
Tu dois construire une question destinée
à une recherche documentaire RAG.

QUESTION ORIGINALE:
{original_question}

RÉSULTAT STATISTIQUE SQL:
{sql_answer}

RÈGLES:
- identifie dans le résultat SQL le joueur
  ou l'équipe concerné ;
- utilise explicitement son nom dans
  la question documentaire ;
- conserve uniquement la demande portant
  sur Reddit, les commentaires, les discussions,
  les perceptions ou les opinions ;
- ne demande aucun calcul ;
- ne réponds pas à la question ;
- retourne uniquement la question documentaire ;
- n'ajoute aucune explication.

EXEMPLE:

Question originale :
Quel joueur a marqué le plus de points et que disent
les discussions Reddit à son sujet ?

Résultat SQL :
Shai Gilgeous-Alexander a marqué le plus de points
avec 2485 points.

Réponse :
Que disent les discussions Reddit sur
Shai Gilgeous-Alexander ?
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content.strip()

    def _answer_hybrid(
        self,
        question: str,
    ) -> dict:
        """
        Traite une question qui nécessite à la fois
        les données structurées SQL et le corpus RAG.
        """
        with logfire.span(
            "hybrid_pipeline",
            question=question,
        ):
            with logfire.span(
                "hybrid_sql_question_generation"
            ):
                sql_question = (
                    self._build_sql_subquestion(
                        question
                    )
                )

            logfire.info(
                "hybrid_sql_question_ready",
                sql_question=sql_question,
            )

            with logfire.span(
                "hybrid_sql_execution"
            ):
                sql_result = (
                    self._answer_from_sql(
                        sql_question
                    )
                )

            with logfire.span(
                "hybrid_rag_question_generation"
            ):
                rag_question = (
                    self._build_rag_subquestion(
                        original_question=question,
                        sql_answer=sql_result[
                            "answer"
                        ],
                    )
                )

            logfire.info(
                "hybrid_rag_question_ready",
                rag_question=rag_question,
            )

            with logfire.span(
                "hybrid_rag_execution"
            ):
                rag_result = (
                    self._answer_from_rag(
                        rag_question
                    )
                )

            synthesis_prompt = f"""
Tu es un analyste NBA.

Tu dois répondre à la question originale
en combinant deux sources distinctes :

1. des statistiques structurées issues de SQL ;
2. des informations documentaires issues
   du corpus Reddit via RAG.

QUESTION ORIGINALE:
{question}

QUESTION SQL:
{sql_question}

RÉSULTAT SQL:
{sql_result["answer"]}

QUESTION RAG:
{rag_question}

RÉSULTAT RAG:
{rag_result["answer"]}

RÈGLES:
- réponds à toutes les parties de la question ;
- distingue clairement les faits statistiques
  des commentaires et opinions ;
- toute valeur numérique doit provenir
  du résultat SQL ;
- toute opinion ou perception doit provenir
  du résultat RAG ;
- n'invente aucune information ;
- ne génère jamais de lien, URL ou image Markdown ;
- si un graphique est demandé, fournis seulement la synthèse textuelle :
  le graphique réel est généré localement par PlotTool ;
- ne transforme pas une opinion Reddit
  en fait objectif ;
- si le corpus ne permet pas de répondre
  à la partie documentaire, indique-le ;
- sois précis, synthétique et lisible.
"""

            with logfire.span(
                "hybrid_final_synthesis"
            ):
                response = self.llm.invoke(
                    synthesis_prompt
                )

            sources = list(
                dict.fromkeys(
                    sql_result.get(
                        "sources",
                        [],
                    )
                    + rag_result.get(
                        "sources",
                        [],
                    )
                )
            )

            logfire.info(
                "hybrid_pipeline_complete",
                source_count=len(sources),
                retrieval_count=len(
                    rag_result.get(
                        "retrieval",
                        [],
                    )
                ),
            )

            return {
                "route": "hybrid",
                "question": question,
                "answer": response.content.strip(),
                "sql_question": sql_question,
                "rag_question": rag_question,
                "sql": sql_result.get("sql"),
                "data": sql_result.get("data"),
                "sources": sources,
                "retrieval": rag_result.get(
                    "retrieval",
                    [],
                ),
            }

    def ask(
        self,
        question: str,
    ) -> dict:
        """
        Route la question vers :
        - SQL ;
        - RAG ;
        - HYBRID SQL + RAG.
        """
        with logfire.span(
            "hybrid_agent.ask",
            question=question,
        ):
            numeric_intent = (
                self._has_numeric_intent(
                    question
                )
            )

            textual_intent = (
                self._has_textual_intent(
                    question
                )
            )

            plot_intent = (
                self._has_plot_intent(
                    question
                )
            )

            logfire.info(
                "routing_detection",
                numeric_intent=numeric_intent,
                textual_intent=textual_intent,
                plot_intent=plot_intent,
            )

            if (
                numeric_intent
                and textual_intent
            ):
                logfire.info(
                    "routing_decision",
                    route="hybrid",
                    decision_source=(
                        "deterministic_rules"
                    ),
                )

                result = self._answer_hybrid(
                    question
                )

                if plot_intent:
                    result = self._attach_plot(
                        question,
                        result,
                    )

                return result

            if numeric_intent:
                logfire.info(
                    "routing_decision",
                    route="sql",
                    decision_source=(
                        "deterministic_rules"
                    ),
                )

                result = self._answer_from_sql(
                    question
                )

                if plot_intent:
                    result = self._attach_plot(
                        question,
                        result,
                    )

                return result

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
                    logfire.info(
                        "routing_decision",
                        route="sql",
                        decision_source=(
                            "llm_router"
                        ),
                    )

                    return self._answer_from_sql(
                        question
                    )

            logfire.info(
                "routing_decision",
                route="rag",
                decision_source="llm_router",
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
        (
            "J'ai vu plein de commentaires "
            "Reddit contradictoires mais bref "
            "kel joueur a marker "
            "le + de points ???"
        ),
        (
            "Quel joueur a marqué le plus "
            "de points et que disent les "
            "discussions Reddit à son sujet ?"
        ),
    ]

    for question in questions:
        print()
        print("=" * 100)
        print(
            "QUESTION :",
            question,
        )
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