import speech_recognition as sr
from google.cloud import speech

recognizer = sr.Recognizer()


def transcribe_voice_with_google_cloud():
    try:
        with sr.Microphone() as source:
            print("Chatbot: Listening... please speak now.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)

        wav_bytes = audio.get_wav_data()

        client = speech.SpeechClient()
        audio_config = speech.RecognitionAudio(content=wav_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="en-GB",
        )

        response = client.recognize(config=config, audio=audio_config)

        if not response.results:
            return None

        return response.results[0].alternatives[0].transcript

    except Exception as e:
        print(f"Chatbot: Google Cloud Speech error: {e}")
        return None