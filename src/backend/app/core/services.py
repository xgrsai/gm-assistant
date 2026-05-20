import google.generativeai as genai
from django.conf import settings
from .prompts import SYSTEM_PROMPT

genai.configure(api_key=settings.GOOGLE_API_KEY)


def get_gm_response(messages: list[dict]) -> str:
    """
    messages: [{"role": "user"|"assistant", "content": "..."}]
    returns: str response from Gemini
    """
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    # Конвертуємо формат React → формат Gemini
    history = []
    for msg in messages[:-1]:  # все крім останнього
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]],
        })

    chat = model.start_chat(history=history)

    last_message = messages[-1]["content"]
    response = chat.send_message(last_message)

    return response.text