from pathlib import Path

import streamlit as st

from sportsee.agent.hybrid_agent import HybridNBAAgent


st.set_page_config(
    page_title="NBA Hybrid AI",
    page_icon="🏀",
    layout="wide",
)


def initial_messages() -> list[dict]:
    """
    Retourne l'historique initial de la conversation.
    """
    return [
        {
            "role": "assistant",
            "content": (
                "Bonjour. Je suis l'assistant NBA hybride de SportSee. "
                "Je peux analyser les discussions Reddit avec le RAG, "
                "interroger les statistiques structurées avec SQL "
                "ou combiner les deux sources."
            ),
            "metadata": None,
        }
    ]


@st.cache_resource
def get_agent() -> HybridNBAAgent:
    """
    Initialise une seule instance de HybridNBAAgent.

    Streamlit conserve cette instance entre les reruns afin
    d'éviter de recharger FAISS et les modèles à chaque question.
    """
    return HybridNBAAgent()


def display_sql_metadata(
    result: dict,
) -> None:
    """
    Affiche les informations techniques associées
    à une réponse SQL.
    """
    st.caption(
        "Route utilisée : SQL"
    )

    sql = result.get("sql")
    data = result.get("data")
    sources = result.get(
        "sources",
        [],
    )

    if sql:
        with st.expander(
            "Voir la requête SQL"
        ):
            st.code(
                sql,
                language="sql",
            )

    if data is not None:
        with st.expander(
            "Voir les données SQL"
        ):
            st.json(data)

    if sources:
        with st.expander(
            "Voir les sources"
        ):
            for source in sources:
                st.write(
                    f"- {source}"
                )


def display_rag_metadata(
    result: dict,
) -> None:
    """
    Affiche les sources et scores de retrieval
    associés à une réponse RAG.
    """
    st.caption(
        "Route utilisée : RAG"
    )

    sources = result.get(
        "sources",
        [],
    )

    retrieval = result.get(
        "retrieval",
        [],
    )

    if sources:
        with st.expander(
            "Voir les sources"
        ):
            for source in sources:
                st.write(
                    f"- {source}"
                )

    if retrieval:
        with st.expander(
            "Voir le retrieval"
        ):
            for item in retrieval:
                source = item.get(
                    "source",
                    "source inconnue",
                )

                score = item.get(
                    "score"
                )

                if score is None:
                    st.write(
                        f"- {source}"
                    )
                else:
                    st.write(
                        f"- {source} "
                        f"— score : {score:.2f}%"
                    )


def display_hybrid_metadata(
    result: dict,
) -> None:
    """
    Affiche les informations techniques d'une réponse
    combinant SQL et RAG.
    """
    st.caption(
        "Route utilisée : HYBRID — SQL + RAG"
    )

    sql_question = result.get(
        "sql_question"
    )

    rag_question = result.get(
        "rag_question"
    )

    sql = result.get("sql")
    data = result.get("data")

    sources = result.get(
        "sources",
        [],
    )

    retrieval = result.get(
        "retrieval",
        [],
    )

    if (
        sql_question
        or rag_question
    ):
        with st.expander(
            "Voir la décomposition de la question"
        ):
            if sql_question:
                st.markdown(
                    "**Sous-question SQL**"
                )
                st.write(
                    sql_question
                )

            if rag_question:
                st.markdown(
                    "**Sous-question RAG**"
                )
                st.write(
                    rag_question
                )

    if sql:
        with st.expander(
            "Voir la requête SQL"
        ):
            st.code(
                sql,
                language="sql",
            )

    if data is not None:
        with st.expander(
            "Voir les données SQL"
        ):
            st.json(data)

    if sources:
        with st.expander(
            "Voir les sources"
        ):
            for source in sources:
                st.write(
                    f"- {source}"
                )

    if retrieval:
        with st.expander(
            "Voir le retrieval RAG"
        ):
            for item in retrieval:
                source = item.get(
                    "source",
                    "source inconnue",
                )

                score = item.get(
                    "score"
                )

                if score is None:
                    st.write(
                        f"- {source}"
                    )
                else:
                    st.write(
                        f"- {source} "
                        f"— score : {score:.2f}%"
                    )


def display_plot(
    result: dict,
) -> None:
    """
    Affiche le graphique généré par PlotTool
    lorsqu'un chemin PNG est disponible.
    """
    plot_path = result.get(
        "plot_path"
    )

    if not plot_path:
        return

    path = Path(
        plot_path
    )

    if not path.exists():
        st.warning(
            "Le graphique généré "
            "n'est plus disponible."
        )
        return

    st.image(
        str(path),
        caption=(
            "Visualisation générée "
            "par PlotTool"
        ),
        use_container_width=True,
    )


def display_metadata(
    result: dict,
) -> None:
    """
    Affiche les métadonnées adaptées à la route
    sélectionnée par l'agent hybride.
    """
    display_plot(
        result
    )

    route = result.get(
        "route",
        "unknown",
    ).lower()

    if route == "sql":
        display_sql_metadata(
            result
        )
        return

    if route == "rag":
        display_rag_metadata(
            result
        )
        return

    if route == "hybrid":
        display_hybrid_metadata(
            result
        )
        return

    if route == "error":
        st.caption(
            "Route utilisée : ERROR"
        )
        return

    st.caption(
        f"Route utilisée : "
        f"{route.upper()}"
    )


st.title(
    "🏀 NBA Hybrid AI"
)

st.caption(
    "Assistant SportSee — "
    "RAG documentaire + SQL analytique + "
    "routage hybride"
)


with st.sidebar:
    st.header(
        "Architecture"
    )

    st.markdown(
        """
### Questions textuelles

**FAISS → RAG → LLM**

Utilisé pour :

- commentaires Reddit ;
- opinions de fans ;
- analyses textuelles ;
- contenu documentaire.

### Questions numériques

**LangChain SQL Tool → SQLite → LLM**

Utilisé pour :

- points ;
- rebonds ;
- moyennes ;
- classements ;
- comparaisons ;
- agrégations.

### Questions mixtes

**SQL + RAG → LLM**

Utilisé lorsqu'une question nécessite à la fois :

- une donnée statistique ;
- une analyse documentaire ;
- des commentaires ou opinions Reddit.

### Modèle local

`Ollama / qwen2.5:7b-instruct`

### Validation

`Pydantic + Pydantic AI`
"""
    )

    st.divider()

    st.subheader(
        "Questions de démonstration"
    )

    st.markdown(
        """
**SQL**

`Quel joueur a marqué le plus de points ?`

`Quelle équipe totalise le plus de points ?`

`Parmi MIA, OKC, LAC, BKN et ATL, quelle équipe a marqué le plus de points ?`

**RAG**

`Que disent les discussions Reddit sur Reggie Miller ?`

`Hali est-il décrit comme très vocal avec ses coéquipiers ?`

**HYBRID**

`Quel joueur a marqué le plus de points et que disent les discussions Reddit à son sujet ?`
"""
    )

    st.divider()

    if st.button(
        "Effacer la conversation",
        use_container_width=True,
    ):
        st.session_state.messages = (
            initial_messages()
        )

        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = (
        initial_messages()
    )


try:
    agent = get_agent()

except Exception as exc:
    st.error(
        "Impossible d'initialiser "
        "l'agent hybride."
    )

    st.exception(exc)

    st.stop()


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        metadata = message.get(
            "metadata"
        )

        if (
            message["role"] == "assistant"
            and metadata
        ):
            display_metadata(
                metadata
            )


question = st.chat_input(
    "Posez une question sur la NBA..."
)


if question:
    user_message = {
        "role": "user",
        "content": question,
        "metadata": None,
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            question
        )

    with st.chat_message(
        "assistant"
    ):
        with st.spinner(
            "Analyse de la question..."
        ):
            try:
                result = agent.ask(
                    question
                )

                answer = result.get(
                    "answer",
                    (
                        "Aucune réponse "
                        "n'a été générée."
                    ),
                )

                st.markdown(
                    answer
                )

                display_metadata(
                    result
                )

            except Exception as exc:
                answer = (
                    "Une erreur est survenue "
                    "pendant le traitement "
                    "de la question."
                )

                result = {
                    "route": "error",
                    "error": str(exc),
                }

                st.error(
                    answer
                )

                with st.expander(
                    "Voir le détail de l'erreur"
                ):
                    st.exception(
                        exc
                    )

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "metadata": result,
    }

    st.session_state.messages.append(
        assistant_message
    )


st.divider()

st.caption(
    "P10 OpenClassrooms — "
    "RAG + SQL Tool + Hybrid Routing + "
    "FAISS + SQLite + Ollama + "
    "Pydantic AI + RAGAS"
)