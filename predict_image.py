"""
predict_image.py
================
CNN image prediction module for the Gym Training Chatbot.
Implements the inference side of Task-c.

This file is SEPARATE from the chatbot as required by the spec.
It provides:
  - load_model_once(): loads gym_cnn_model.h5 with caching
  - predict_image():   preprocesses an image and returns (class, confidence)
  - get_friendly_response(): formats a chatbot-ready response string

Classes (gym-training focused):
  barbell, dumbbell, kettlebell, pull_up_bar,
  treadmill, rowing_machine, bench, squat_rack

These classes are visuallly distinct gym equipment/items which
gives the CNN a reasonable chance of good accuracy even with a
small dataset. They are directly relevant to the gym-training topic.

Can be run standalone to test a prediction:
  python predict_image.py --image test_images/dumbbell.jpg
"""

import os
import sys
import argparse
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ------------------------------------------------------------------ #
#  Configuration — must match train_cnn.py exactly                   #
# ------------------------------------------------------------------ #

MODEL_PATH = 'gym_cnn_model.h5'
IMG_HEIGHT = 128
IMG_WIDTH  = 128

# Class names in the SAME ORDER used during training
# (ImageDataGenerator.flow_from_directory uses alphabetical order)
CLASS_NAMES = [
    'barbell',
    'bench',
    'dumbbell',
    'kettlebell',
    'pull_up_bar',
    'rowing_machine',
    'squat_rack',
    'treadmill',
]

# Gym-training descriptions for each class
CLASS_DESCRIPTIONS = {
    'barbell':        'a barbell — the primary tool for compound strength training movements such as the squat, deadlift, bench press, and overhead press.',
    'bench':          'a weight bench — used for bench press, dumbbell exercises, step-ups, and tricep dips. A fundamental piece of gym furniture.',
    'dumbbell':       'a dumbbell — a versatile piece of free weight equipment used for both compound and isolation exercises. Ideal for unilateral training.',
    'kettlebell':     'a kettlebell — a cast iron weight used for ballistic exercises such as swings, cleans, and Turkish get-ups. Excellent for conditioning.',
    'pull_up_bar':    'a pull-up bar — used for pull-ups, chin-ups, and hanging core exercises. One of the best pieces of equipment for upper back development.',
    'rowing_machine': 'a rowing machine (ergometer) — a full-body cardiovascular machine that also builds upper back and leg endurance. Very low impact.',
    'squat_rack':     'a squat rack (power rack) — the most important piece of strength training equipment in a gym. Used for squats, bench press, overhead press, and barbell rows safely.',
    'treadmill':      'a treadmill — a cardiovascular machine for walking, jogging, or running indoors. Used for warm-up, cardio training, and interval work.',
}

# Module-level model cache (load once, reuse)
_cached_model = None


def load_model_once(model_path: str = MODEL_PATH):
    """
    Load the saved CNN model, caching it after the first load.

    Args:
        model_path: Path to the .h5 model file.

    Returns:
        Loaded Keras model, or None if the file does not exist.
    """
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not os.path.exists(model_path):
        print(f"[predict_image] Model not found at '{model_path}'.")
        print("[predict_image] Please run 'python train_cnn.py' to train and save the model.")
        return None

    print(f"[predict_image] Loading CNN model from '{model_path}'...")
    _cached_model = load_model(model_path)
    print("[predict_image] CNN model loaded successfully.")
    return _cached_model


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load and preprocess an image for CNN prediction.

    Steps: load at (IMG_HEIGHT, IMG_WIDTH) -> numpy array ->
           add batch dimension -> normalise to [0, 1].
    """
    img = keras_image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_image(model, img_path: str) -> tuple[str, float]:
    """
    Classify an image using the loaded CNN model.

    Args:
        model:    Loaded Keras model.
        img_path: Path to image file.

    Returns:
        (class_name, confidence_percentage) or ("unknown", 0.0) on error.
    """
    if not os.path.exists(img_path):
        print(f"[predict_image] File not found: '{img_path}'")
        return "unknown", 0.0

    try:
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index]) * 100.0
        class_name = CLASS_NAMES[predicted_index]

        # Print all class probabilities (useful for documentation/demo)
        print("\n[predict_image] All class probabilities:")
        for i, name in enumerate(CLASS_NAMES):
            bar = '█' * int(predictions[0][i] * 20)
            print(f"  {name:18s}: {predictions[0][i]*100:5.1f}%  {bar}")

        return class_name, confidence

    except Exception as e:
        print(f"[predict_image] Prediction error: {e}")
        return "unknown", 0.0


def get_friendly_response(class_name: str, confidence: float) -> str:
    """
    Build a chatbot-friendly response using percentage confidence.
    confidence is expected as a percentage, e.g. 99.3
    """

    if class_name == "unknown":
        return (
            "I was unable to classify this image.\n"
            "Please use a clearer image and try again."
        )

    if confidence < 60:
        return (
            "I am not confident enough to classify this image reliably.\n"
            f"Best guess: {class_name}\n"
            f"Confidence: {confidence:.1f}%"
        )

    if confidence >= 90:
        return (
            f"I am very confident this is {class_name}.\n"
            f"Confidence: {confidence:.1f}%"
        )

    if confidence >= 75:
        return (
            f"I think this is {class_name}.\n"
            f"Confidence: {confidence:.1f}%"
        )

    return (
        f"My best guess is {class_name}.\n"
        f"Confidence: {confidence:.1f}%"
    )


# ------------------------------------------------------------------ #
#  Standalone CLI                                                     #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify a gym equipment image using the trained CNN model.'
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file to classify.')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help=f'Path to the .h5 model file. Default: {MODEL_PATH}')
    args = parser.parse_args()

    model = load_model_once(args.model)
    if model is None:
        sys.exit(1)

    cls, conf = predict_image(model, args.image)
    print(f"\nResult: {get_friendly_response(cls, conf)}")
