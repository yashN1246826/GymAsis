````markdown
# GymAsis – AI Gym Chatbot

GymAsis is a gym-focused AI chatbot built in Python. It combines conversational AI, logical reasoning, image classification, voice interaction, and cloud-based recognition into one interactive system.

The chatbot can answer gym-related questions, check knowledge-based facts, classify gym equipment images, and support both text and voice interaction.

## Features

- Gym Q&A chatbot using AIML pattern matching
- Similarity-based fallback using TF-IDF and cosine similarity
- First-order logic reasoning using NLTK
- Knowledge base support for facts such as:
  - `I know that X is Y`
  - `Check that X is Y`
  - `I know that X trains Y`
  - `Check that X trains Y`
- CNN-based gym equipment image classification using TensorFlow/Keras
- Google Cloud Vision support for cloud-based image recognition
- Google Cloud Speech-to-Text support for voice input
- Text-to-speech output using `pyttsx3`
- API-assisted validation before storing new user facts

## Tech Stack

- Python
- AIML
- TF-IDF
- Cosine Similarity
- NLTK
- TensorFlow / Keras
- Google Cloud Speech-to-Text
- Google Cloud Vision
- pyttsx3

## Project Structure

```text
GymAsis/
│
├── main_chatbot.py              # Main chatbot controller
├── gym_chatbot.aiml             # AIML rules and direct responses
├── similarity_matcher.py        # TF-IDF and cosine similarity fallback
├── gym_qa.csv                   # Question-answer dataset
│
├── logic_engine.py              # First-order logic reasoning
├── gym_kb.csv                   # Knowledge base facts and rules
│
├── train_cnn.py                 # CNN model training script
├── predict_image.py             # Image prediction script
├── gym_cnn_model.h5             # Trained CNN model
├── gym_cnn_checkpoint.h5        # CNN checkpoint file
│
├── cloud_speech_helper.py       # Google Cloud Speech-to-Text helper
├── cloud_text_helper.py         # Cloud text/API fallback helper
├── cloud_vision_helper.py       # Google Cloud Vision helper
│
├── test_images/                 # Sample test images
├── dataset/                     # Training/testing image dataset
├── training_history.png         # CNN training graph
└── requirements.txt             # Python dependencies
````

## How It Works

GymAsis uses a hybrid AI approach.

For direct questions, it uses AIML pattern matching to return fixed gym-related responses. If the user asks the same question in different wording, the system uses TF-IDF and cosine similarity to find the closest matching question from the stored Q&A dataset.

The chatbot also includes a reasoning layer. It uses a first-order logic knowledge base through NLTK, allowing users to add and check facts. This means the chatbot can respond with whether a statement is correct, incorrect, or unknown.

For image recognition, the system includes a locally trained CNN model that classifies gym equipment images. It also includes Google Cloud Vision as a second recognition method, which can identify image labels and detect whether an image is not gym-related.

## Example Commands

```text
hi
show kb
listen
classify image
cloud classify image
I know that squat trains legs
Check that squat trains legs
I know that dumbbell is equipment
Check that dumbbell is equipment
quit
```

## Running the Project

1. Clone the repository:

```bash
git clone https://github.com/yashN1246826/GymAsis.git
cd GymAsis
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the chatbot:

```bash
python main_chatbot.py
```

## Cloud API Setup

Some features use Google Cloud services, such as Speech-to-Text and Cloud Vision.

To use these features, set up a Google Cloud service account and configure your credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json"
```

On Windows PowerShell:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account.json"
```

Cloud-based features may not work unless the credentials are configured correctly.

## Model Training

To train the CNN model again:

```bash
python train_cnn.py
```

To test image prediction separately:

```bash
python predict_image.py
```

## Key AI Techniques Used

* Rule-based chatbot responses using AIML
* Text similarity matching using TF-IDF and cosine similarity
* First-order logic reasoning with NLTK
* CNN image classification using TensorFlow/Keras
* Cloud-based image recognition using Google Cloud Vision
* Voice interaction using Google Cloud Speech-to-Text

## Screenshots / Demo

Add screenshots or demo GIFs here showing:

* AIML chatbot response
* Similarity-based response
* Voice command
* Knowledge base reasoning
* CNN image classification
* Cloud Vision recognition

## Future Improvements

* Add a web interface
* Improve CNN accuracy with a larger dataset
* Add more gym equipment and exercise classes
* Store conversation history
* Improve fact validation using a stronger knowledge graph
* Deploy the chatbot as a cloud-based web application

## Author

**Yash Kumar**
Computer Science with Artificial Intelligence
Nottingham Trent University

```

This README matches your actual project: AIML, TF-IDF/cosine similarity, NLTK logic reasoning, CNN classification, Google Cloud Vision, Speech-to-Text, and the specific files shown in your documentation. :contentReference[oaicite:0]{index=0}
```
