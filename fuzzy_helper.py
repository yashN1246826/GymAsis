"""
fuzzy_helper.py
===============
Simple fuzzy-style reasoning helper for GymBot.
Returns low / medium / high style answers for a few gym-related queries.
"""

import re

FUZZY_KB = {
    ("squat", "difficulty", "beginners"): "high",
    ("deadlift", "difficulty", "beginners"): "high",
    ("treadmill", "cardio", "fitness"): "high",
    ("dumbbell", "suitability", "beginners"): "medium",
    ("bench press", "difficulty", "beginners"): "medium",
    ("plank", "core", "training"): "high",
}

def handle_fuzzy_query(user_input: str):
    text = user_input.lower().strip()

    patterns = [
        (r"how difficult is (.+?) for beginners", "difficulty", "beginners"),
        (r"how good is (.+?) for cardio", "cardio", "fitness"),
        (r"how suitable is (.+?) for beginners", "suitability", "beginners"),
        (r"how good is (.+?) for core training", "core", "training"),
    ]

    for pattern, attribute, context in patterns:
        match = re.match(pattern, text)
        if match:
            item = match.group(1).strip()
            key = (item, attribute, context)

            if key in FUZZY_KB:
                value = FUZZY_KB[key]
                return (
                    f"Using my fuzzy-style knowledge, the {attribute} of {item} "
                    f"for {context} is {value}."
                )
            else:
                return (
                    f"I do not have a fuzzy-style rating for {item} "
                    f"in that category yet."
                )

    return None