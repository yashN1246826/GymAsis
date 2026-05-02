from google import genai

PROJECT_ID = "project-ddff7863-446f-4384-b6b"
LOCATION = "global"
MODEL_NAME = "gemini-2.5-flash"

def cloud_answer_text(user_text: str) -> str:
    """
    Use Google Gemini on Vertex AI to answer only gym-related questions.
    If the API is unavailable, return a safe fallback instead of crashing.
    """
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )

        prompt = f"""
You are GymBot, a gym training assistant.

Answer ONLY gym-related questions about:
- exercises
- workout technique
- muscle groups
- recovery
- gym equipment
- training frequency
- beginner guidance

If the question is not gym-related, reply exactly with:
Please ask me a gym-related question.

User question:
{user_text}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()

    except Exception as e:
        print(f"[cloud_text_helper] Google text API error: {e}")

    return "Please ask me a gym-related question."