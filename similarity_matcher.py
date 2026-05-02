"""
similarity_matcher.py
=====================
Similarity-based Q/A matching for the Gym Training Chatbot.
Implements Task-a: lemmatisation + TF-IDF + cosine similarity.
"""

import csv
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data silently
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)


class SimilarityMatcher:
    """
    Matches user input to the closest question in the CSV Q/A file
    using lemmatisation, TF-IDF vectorisation, and cosine similarity.
    """

    def __init__(self, csv_path: str, threshold: float = 0.10):
        """
        Initialise the matcher.

        Args:
            csv_path:  Path to gym_qa.csv.
            threshold: Minimum cosine similarity to return a match.
        """
        self.threshold = threshold
        self.lemmatizer = WordNetLemmatizer()

        base_stops = set(stopwords.words('english'))
        keep_words = {
            'how', 'what', 'why', 'when', 'which', 'do', 'does',
            'should', 'can', 'will', 'is', 'are', 'not', 'no'
        }
        self.stop_words = base_stops - keep_words

        self.questions = []
        self.answers = []
        self._load_csv(csv_path)

        self.processed_questions = [self._preprocess(q) for q in self.questions]

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

        print(f"[SimilarityMatcher] Loaded {len(self.questions)} Q/A pairs.")
        print(f"[SimilarityMatcher] Vocabulary size: {len(self.vectorizer.vocabulary_)} terms.")

    def _load_csv(self, csv_path: str):
        """Load question/answer pairs from CSV file."""
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.questions.append(row['question'].strip())
                self.answers.append(row['answer'].strip())

    def _preprocess(self, text: str) -> str:
        """
        Clean and lemmatise a text string.
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()
        processed = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]

        return ' '.join(processed) if processed else text

    def get_best_answer(self, user_input: str) -> str | None:
        """
        Find the best-matching question and return its answer.
        """
        processed_input = self._preprocess(user_input)

        if not processed_input.strip():
            return None

        try:
            user_vector = self.vectorizer.transform([processed_input])
        except Exception:
            return None

        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        best_index = int(similarities.argmax())
        best_score = float(similarities[best_index])

        print(
            f"[SimilarityMatcher] Best match: '{self.questions[best_index][:60]}...' "
            f"(score={best_score:.3f})"
        )

        if best_score >= self.threshold:
            return self.answers[best_index]

        print(
            f"[SimilarityMatcher] Score {best_score:.3f} below threshold {self.threshold}. "
            f"No match returned."
        )
        return None

    def get_best_match(self, user_input: str):
        """
        Return (best_question, best_answer, best_score)
        for voice-mode filtering.
        """
        processed_input = self._preprocess(user_input)

        if not processed_input.strip():
            return None, None, 0.0

        try:
            user_vector = self.vectorizer.transform([processed_input])
        except Exception:
            return None, None, 0.0

        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()

        best_index = int(similarities.argmax())
        best_score = float(similarities[best_index])

        if best_score <= 0:
            return None, None, 0.0

        best_question = self.questions[best_index]
        best_answer = self.answers[best_index]

        print(f"[SimilarityMatcher] Best match: '{best_question}' (score={best_score:.3f})")
        return best_question, best_answer, best_score