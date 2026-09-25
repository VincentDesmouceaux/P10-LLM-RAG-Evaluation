import json

from ollama import chat
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelResponse,
    ToolCallPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    FunctionModel,
)

from sportsee.observability.logfire_config import (
    configure_observability,
)
from sportsee.rag.schemas import RAGAnswer


OLLAMA_MODEL = "qwen2.5:7b-instruct"


SYSTEM_PROMPT = """
Tu es un assistant expert NBA utilisant exclusivement
le contexte fourni.

RÈGLES OBLIGATOIRES :

1. Réponds uniquement avec les informations présentes
   dans le contexte.

2. N'invente aucune information et ne complète jamais
   avec tes connaissances générales.

3. Réponds explicitement à CHAQUE élément demandé
   dans la question.

4. Si la question contient plusieurs parties,
   traite chacune d'elles dans la réponse.

5. Ne remplace jamais un élément demandé par une
   généralité différente, même si cette généralité
   est vraie dans le contexte.

6. Pour une comparaison ou une opposition,
   expose clairement les deux côtés demandés.

7. Lorsque le contexte contient plusieurs opinions,
   présente-les comme des opinions ou commentaires
   et non comme des faits absolus.

8. La réponse doit être autosuffisante, précise
   et complète. Elle ne doit jamais s'arrêter
   après une introduction ou se terminer par ":".

9. Si le contexte ne permet pas de répondre à une
   partie de la question, indique explicitement
   quelle partie n'est pas documentée.

10. Le champ "question" doit reproduire exactement
    la question utilisateur.

11. Le champ "sources" doit contenir uniquement
    des sources effectivement présentes dans
    le contexte.

Retourne uniquement l'objet JSON demandé.
""".strip()


def generate_structured_answer(
    question: str,
    context: str,
) -> RAGAnswer:
    configure_observability()

    attempt_number = 0

    def ollama_model_function(
        messages,
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal attempt_number

        attempt_number += 1

        retry_instruction = ""

        if attempt_number > 1:
            retry_instruction = """
La réponse précédente a été rejetée car elle était
invalide ou incomplète.

Réécris entièrement la réponse.
Elle doit répondre à toutes les parties de la
question et ne doit pas se terminer par ":".
""".strip()

        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXTE:\n{context}\n\n"
        )

        if retry_instruction:
            user_prompt += (
                f"{retry_instruction}\n\n"
            )

        user_prompt += (
            "Produis maintenant une réponse complète "
            "au format JSON demandé."
        )

        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            format=RAGAnswer.model_json_schema(),
            options={
                "temperature": 0,
                "num_predict": 700,
            },
        )

        payload = json.loads(
            response.message.content
        )

        if not info.output_tools:
            raise RuntimeError(
                "Aucun outil de sortie "
                "Pydantic AI disponible."
            )

        output_tool_name = (
            info.output_tools[0].name
        )

        return ModelResponse(
            parts=[
                ToolCallPart(
                    output_tool_name,
                    payload,
                )
            ]
        )

    model = FunctionModel(
        ollama_model_function
    )

    agent = Agent(
        model=model,
        output_type=RAGAnswer,
        retries=2,
    )

    result = agent.run_sync(
        "Produis la réponse structurée."
    )

    return result.output
