from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .services import get_gm_response


@api_view(["POST"])
def chat(request):
    messages = request.data.get("messages", [])

    if not messages:
        return Response(
            {"error": "messages is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if messages[-1].get("role") != "user":
        return Response(
            {"error": "last message must be from user"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        reply = get_gm_response(messages)
        return Response({"content": reply})
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )