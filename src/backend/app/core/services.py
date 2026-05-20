from google import genai
from google.genai import types
from django.conf import settings
from .prompts import SYSTEM_PROMPT


def get_gm_response(messages: list[dict]) -> str:
    """
    messages: [{"role": "user"|"assistant", "content": "..."}]
    returns: str response from Gemini
    """
    client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    # Конвертуємо формат React → формат Gemini
    history = []
    for msg in messages[:-1]:
        history.append(
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part(text=msg["content"])],
            )
        )

    last_message = messages[-1]["content"]

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
        contents=[
            *history,
            types.Content(
                role="user",
                parts=[types.Part(text=last_message)],
            ),
        ],
    )

    return response.text