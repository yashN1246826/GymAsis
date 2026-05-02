"""
main_chatbot.py
===============
GymBot — Gym Training Chatbot
NTU ISYS30221 Artificial Intelligence Coursework
"""

import os
import sys

_QUIET_STARTUP = True
_ORIGINAL_STDERR = sys.stderr

if _QUIET_STARTUP:
    sys.stderr = open(os.devnull, "w")

import re
import io
import contextlib
import logging
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyttsx3
from cloud_text_helper import cloud_answer_text
warnings.filterwarnings("ignore")
logging.getLogger("absl").setLevel(logging.ERROR)

try:
    import aiml
except ImportError:
    print("ERROR: 'aiml' package not installed.")
    print("Run:  pip install python-aiml")
    sys.exit(1)

from similarity_matcher import SimilarityMatcher
from logic_engine import LogicEngine
from predict_image import load_model_once, predict_image, get_friendly_response
from cloud_vision_helper import cloud_classify_image
from cloud_speech_helper import transcribe_voice_with_google_cloud

AIML_FILE      = 'gym_chatbot.aiml'
QA_CSV_FILE    = 'gym_qa.csv'
KB_FILE        = 'gym_kb.csv'
CNN_MODEL_FILE = 'gym_cnn_model.h5'

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)


def load_aiml_kernel(aiml_file: str) -> aiml.Kernel:
    kernel = aiml.Kernel()
    kernel.verbose(False)
    if os.path.exists(aiml_file):
        kernel.learn(aiml_file)
    return kernel


def startup():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        kernel = load_aiml_kernel(AIML_FILE)
        matcher = SimilarityMatcher(QA_CSV_FILE)
        engine = LogicEngine(KB_FILE)
        cnn_model = load_model_once(CNN_MODEL_FILE)

    return kernel, matcher, engine, cnn_model


_IKNOW_RE = re.compile(
    r'^i\s+know\s+that\s+(.+?)\s+is\s+(.+)$', re.IGNORECASE
)

_CHECK_RE = re.compile(
    r'^check\s+that\s+(.+?)\s+is\s+(.+)$', re.IGNORECASE
)

_IKNOW_TRAINS_RE = re.compile(
    r'^i\s+know\s+that\s+(.+?)\s+trains\s+(.+)$', re.IGNORECASE
)

_CHECK_TRAINS_RE = re.compile(
    r'^check\s+that\s+(.+?)\s+trains\s+(.+)$', re.IGNORECASE
)

_IMAGE_KEYWORDS = [
    'classify image', 'classify this image',
    'what is in this image', 'what is in the image',
    'what is this image', 'identify this image',
    'identify image', 'image classification',
    'what do you see in this image', 'analyse this image',
    'analyze this image', 'what is in this picture',
    'classify picture', 'what equipment is this',
    'what exercise is this',
]

_CLOUD_IMAGE_KEYWORDS = [
    "cloud classify image",
    "cloud image recognition",
    "google classify image",
    "google vision image",
]


def _detect_i_know(text: str):
    m = _IKNOW_RE.match(text.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def _detect_check(text: str):
    m = _CHECK_RE.match(text.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def _detect_i_know_trains(text: str):
    m = _IKNOW_TRAINS_RE.match(text.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def _detect_check_trains(text: str):
    m = _CHECK_TRAINS_RE.match(text.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def _detect_image_query(text: str) -> bool:
    lower = text.lower().strip()
    return any(kw in lower for kw in _IMAGE_KEYWORDS)


def _detect_cloud_image_query(text: str) -> bool:
    lower = text.lower().strip()
    return any(kw in lower for kw in _CLOUD_IMAGE_KEYWORDS)


IMAGE_MAP = {
    "barbell": "test_images/barbell.jpg",
    "bench": "test_images/bench.jpg",
    "dumbbell": "test_images/dumbbell.jpg",
    "kettlebell": "test_images/kettlebell.jpg",
    "pull up bar": "test_images/pull_up_bar.jpg",
    "pull-up bar": "test_images/pull_up_bar.jpg",
    "pull_up_bar": "test_images/pull_up_bar.jpg",
    "rowing machine": "test_images/rowing_machine.jpg",
    "rowing_machine": "test_images/rowing_machine.jpg",
    "squat rack": "test_images/squat_rack.jpg",
    "squat_rack": "test_images/squat_rack.jpg",
    "treadmill": "test_images/treadmill.jpg",
}


def extract_equipment_request(user_input: str):
    text = user_input.lower().strip()

    patterns = [
        r"show me (?:an image of |a picture of |a |an )(.+)",
        r"can i see (?:an image of |a picture of |a |an )(.+)",
        r"show (?:an image of |a picture of |a |an )(.+)",
        r"give me (?:an image of |a picture of |a |an )(.+)",
    ]

    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            item = match.group(1).strip(" ?.!").lower()
            return item
    return None


def show_equipment_image(equipment_name: str):
    if equipment_name not in IMAGE_MAP:
        return False

    image_path = IMAGE_MAP[equipment_name]

    if not os.path.exists(image_path):
        print(f"Chatbot: Sorry, I could not find an image for {equipment_name}.")
        return True

    print(f"Chatbot: Here is an image of {equipment_name}.")
    img = mpimg.imread(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(equipment_name.replace("_", " ").title())
    plt.show()
    return True


def speak_text(text: str):
    try:
        clean_text = text.replace("*", "")
        clean_text = clean_text.replace("\n", " ")
        clean_text = " ".join(clean_text.split())
        tts_engine.say(clean_text)
        tts_engine.runAndWait()
    except Exception:
        pass


def listen_to_voice():
    text = transcribe_voice_with_google_cloud()
    if text:
        print(f"You (voice): {text}")
        return text

    print("Chatbot: Sorry, I could not understand your voice.")
    return None


def handle_image_query(cnn_model) -> str:
    if cnn_model is None:
        return "Image classification is currently unavailable."

    print("\nChatbot: Please enter the path to the image file you want to classify.")
    print("         Example: test_images/dumbbell.jpg")
    print("Image path: ", end="", flush=True)
    img_path = input("Image path: ").strip()

    if not img_path:
        return "No path entered. Please try again and enter a valid file path."

    if not os.path.exists(img_path):
        return f"I could not find a file at '{img_path}'. Please check the path and try again."

    predicted_class, confidence = predict_image(cnn_model, img_path)
    return get_friendly_response(predicted_class, confidence)


def handle_cloud_image_query() -> str:
    print("\nChatbot: Please enter the path to the image file for Google Cloud Vision.")
    print("         Example: test_images/dumbbell.jpg")
    print("Image path: ", end="", flush=True)
    img_path = input("Image path: ").strip()

    if not img_path:
        return "No path entered. Please try again."

    if not os.path.exists(img_path):
        return f"I could not find a file at '{img_path}'. Please check the path and try again."

    try:
        labels = cloud_classify_image(img_path)
    except Exception as e:
        return f"Google Cloud Vision error: {e}"

    if not labels:
        return "Google Cloud Vision could not identify anything clearly in this image."

    label_names = [name.lower() for name, score in labels]

    gym_keywords = {
        "gym", "fitness", "exercise", "workout", "training",
        "dumbbell", "barbell", "kettlebell", "treadmill",
        "bench", "pull-up", "pull up", "squat rack",
        "weight", "weights", "weight training",
        "exercise equipment", "strength training",
        "physical fitness", "resistance training"
    }

    is_gym_related = any(
        any(keyword in label for keyword in gym_keywords)
        for label in label_names
    )

    formatted = ", ".join([f"{name} ({score}%)" for name, score in labels])

    if not is_gym_related:
        return (
            f"This does not appear to be a gym-related image.\n"
            f"Google Cloud Vision labels: {formatted}"
        )

    return f"Google Cloud Vision labels: {formatted}"


def get_response(user_input: str, kernel: aiml.Kernel,
                 matcher: SimilarityMatcher, engine: LogicEngine,
                 cnn_model, voice_mode: bool = False) -> str:
    raw = user_input.strip()
    if not raw:
        return "Please type a question and I will do my best to help!"

    lower = raw.lower()

    if lower in ('quit', 'exit', 'bye', 'goodbye'):
        return "__EXIT__"

    if lower == 'show kb':
        engine.display_kb()
        return "Knowledge base displayed above."

    x, y = _detect_i_know(raw)
    if x and y:
        return engine.handle_i_know(x, y)

    x, y = _detect_check(raw)
    if x and y:
        return engine.handle_check(x, y)

    x, y = _detect_i_know_trains(raw)
    if x and y:
        return engine.handle_i_know_trains(x, y)

    x, y = _detect_check_trains(raw)
    if x and y:
        return engine.handle_check_trains(x, y)

    if _detect_cloud_image_query(raw):
        return handle_cloud_image_query()

    if _detect_image_query(raw):
        return handle_image_query(cnn_model)

    aiml_response = kernel.respond(raw.upper())

    aiml_matched = (
        aiml_response
        and aiml_response.strip() != "SIMILARITY_FALLBACK"
    )

    if aiml_matched:
        return aiml_response.strip()

    if voice_mode:
        best_question, best_answer, best_score = matcher.get_best_match(raw)
        if best_answer and best_score >= 0.60:
            return best_answer
        return cloud_answer_text(raw)

    sim_response = matcher.get_best_answer(raw)
    if sim_response:
        return sim_response

    return (
        "I could not find a strong gym-related match for that. "
        "Please ask me a gym-related question about exercises, technique, "
        "muscle groups, recovery, equipment, or training."
    )


def main():
    kernel, matcher, engine, cnn_model = startup()

    global _ORIGINAL_STDERR
    sys.stderr = _ORIGINAL_STDERR

    print("Chatbot: Hello, I am GymAsis. How can I help you today?")
    print("Commands: listen | classify image | cloud classify image | show kb | quit")

    while True:
        try:
            print("You: ", end="", flush=True)
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChatbot: Goodbye!")
            speak_text("Goodbye!")
            break

        if not user_input:
            continue

        used_voice_input = False

        if user_input.lower() in ["listen to my voice", "voice input", "voice mode", "listen"]:
            spoken_text = listen_to_voice()
            if not spoken_text:
                continue
            user_input = spoken_text
            used_voice_input = True

        requested_item = extract_equipment_request(user_input)
        if requested_item:
            if show_equipment_image(requested_item):
                continue
            else:
                print("Chatbot: Sorry, I do not have an image for that equipment yet.\n")
                speak_text("Sorry, I do not have an image for that equipment yet.")
                continue

        response = get_response(
            user_input,
            kernel,
            matcher,
            engine,
            cnn_model,
            voice_mode=used_voice_input
        )

        if response == "__EXIT__":
            print("Chatbot: Goodbye!")
            speak_text("Goodbye!")
            break

        print(f"Chatbot: {response}\n")
        speak_text(response)


if __name__ == '__main__':
    main()