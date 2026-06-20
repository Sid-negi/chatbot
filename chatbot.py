    # To install the required libraries, run:
# pip install nltk scikit-learn numpy sentence-transformers gradio

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gradio as gr
import numpy as np

# Download necessary NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# 1. Collect FAQs on Product Support
faq_data = [
    {"question": "How do I reset my password?", "answer": "To reset your password, go to the login page and click on 'Forgot Password'. Follow the instructions sent to your ema        il."},
    {"question": "What are the system requirements for this software?", "answer": "The software requires Windows 10 or later, 8GB RAM, and 20GB of free disk space."},
    {"question": "How can I contact customer support?", "answer": "You can contact customer support via email at support@example.com or call us at 1-800-555-0123."},
    {"question": "Is there a free trial available?", "answer": "Yes, we offer a 30-day free trial. You can sign up on our website."},
    {"question": "How do I update my billing information?", "answer": "You can update your billing information by logging into your account and navigating to the 'Billing' section."},
    {"question": "What payment methods do you accept?", "answer": "We accept Visa, MasterCard, American Express, and PayPal."},
    {"question": "How do I cancel my subscription?", "answer": "To cancel your subscription, log in to your account, go to 'Settings' -> 'Subscription', and click 'Cancel Subscription'."},
    {"question": "Can I get a refund?", "answer": "Refunds are available within 14 days of purchase. Please contact support with your order details."},
    {"question": "Where can I find user manuals?", "answer": "User manuals are available for download on our support page under the 'Documentation' section."},
    {"question": "How to troubleshoot common errors?", "answer": "For common errors, please refer to our troubleshooting guide on the support page or contact customer support if the issue persists."},
    {"question": "What is the warranty period?", "answer": "Our products come with a 1-year limited warranty from the date of purchase."},
    {"question": "How do I activate the product?", "answer": "After installation, launch the software and enter your license key when prompted to activate the product."}
]

# Extract questions and answers
faq_questions = [item["question"] for item in faq_data]
faq_answers = [item["answer"] for item in faq_data]

# 2. Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Tokenizes, removes punctuation, lowercases, and removes stopwords from the given text.
    """
    # Tokenize
    tokens = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# 3. Match user questions using sentence-transformers and cosine similarity
# Load the pre-trained sentence-transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all FAQ questions (original, not preprocessed, for better semantic understanding)
faq_question_embeddings = model.encode(faq_questions)

def get_answer(user_question, history): # Added 'history' argument
    _similarity_threshold = 0.2
    # Ensure FAQ embeddings are not empty before proceeding
    if len(faq_question_embeddings) == 0:
        return "Chatbot is not configured with any FAQ questions. Please try again later."

    # 1. Check if user typed anything meaningful
    if not user_question or not user_question.strip():
        return "Please ask a valid question."

    # 2. Preprocess the question
    preprocessed_user_question = preprocess_text(user_question)

    # 3. If preprocessing removed everything (e.g., "!!!", "the the")
    if not preprocessed_user_question.strip():
        return "Your question seems to contain only punctuation or common words. Please rephrase."

    # 4. Generate embedding for the user question (original, not preprocessed)
    user_question_embedding = model.encode([user_question])

    # 5. Compute similarities with all FAQs
    similarities = cosine_similarity(user_question_embedding, faq_question_embeddings)[0]

    # 6. Safety check: if for some reason we got an empty array
    if len(similarities) == 0:
        return "No FAQ questions available. Please try again later."

    # 7. Find the best match
    best_match_index = np.argmax(similarities)
    max_similarity = float(similarities[best_match_index]) # Explicitly cast to float

    # 8. Return answer if similarity is high enough
    if max_similarity >= _similarity_threshold:
        return faq_answers[best_match_index]
    else:
        return "I don't have an answer for that. Please try rephrasing or contact support."

# 5. UI: Create a simple chat user interface using Gradio
if __name__ == "__main__":
    # Create Gradio ChatInterface
    interface = gr.ChatInterface(
        fn=get_answer,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask me a question about product support...", container=False, scale=7),
        title="Product Support FAQ Chatbot",
        description="Ask questions about product support and get instant answers.",
#        theme="soft",
        examples=[
            "How do I reset my password?",
            "What are the system requirements?",
            "How to contact customer service?",
            "Is there a free trial?",
            "I need help with billing."
        ],
#        cache_examples=False,
#        retry_btn=None,
#        undo_btn="Delete Last Message",
#        clear_btn="Clear Chat"
    )

    # Launch the Gradio interface
    interface.launch(inbrowser=True)

