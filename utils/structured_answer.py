import json

from ollama import chat
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from utils.schemas import RAGAnswer
from utils.observability import configure_observability


OLLAMA_MODEL = "qwen2.5:7b-instruct"


def generate_structured_answer(
    question: str,
    context: str,
) -> RAGAnswer:
    configure_observability()
    def ollama_model_function(
        messages,
        info: AgentInfo,
    ) -> ModelResponse:
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant expert NBA. "
                        "Réponds uniquement à partir du contexte fourni. "
                        "N'invente aucune information. "
                        "La liste sources doit contenir uniquement "
                        "les sources présentes dans le contexte."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{question}\n\n"
                        f"CONTEXTE:\n{context}\n\n"
                        "Retourne la réponse au format JSON demandé."
                    ),
                },
            ],
            format=RAGAnswer.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        payload = json.loads(
            response.message.content
        )

        if not info.output_tools:
            raise RuntimeError(
                "Aucun outil de sortie Pydantic AI disponible."
            )

        output_tool_name = info.output_tools[0].name

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
    )

    result = agent.run_sync(
        "Produis la réponse structurée."
    )

    return result.output
