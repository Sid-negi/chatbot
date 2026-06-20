# Product Support FAQ Chatbot

A simple, fast, and intelligent FAQ Chatbot built with Python, Gradio, and Sentence Transformers. It uses semantic similarity to match user queries to the best available FAQ answer, making it more robust than simple keyword matching.

---

## 🚀 Features

- **Semantic Search**: Uses the state-of-the-art `all-MiniLM-L6-v2` Sentence Transformer model to understand the *meaning* of user questions rather than just checking for exact keywords.
- **Robust Preprocessing**: Cleans user queries using NLTK (lowercasing, punctuation removal, and stopword filtering) to handle typos or noise.
- **Modern UI**: Powered by Gradio, providing a clean, responsive, and easy-to-use web chat interface.
- **Ready to Deploy**: Easily customizable dataset of FAQs.

---

## 🛠️ Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd faq-Chatbot
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required libraries listed at the top of `chatbot.py`:
```bash
pip install nltk scikit-learn numpy sentence-transformers gradio
```

---

## 💻 How to Run

To start the chatbot application:

```bash
python chatbot.py
```

After running, Gradio will spin up a local server and output a URL (usually `http://127.0.0.1:7860`). Open this link in your web browser to start chatting with the bot!

---

## ⚙️ How It Works

1. **FAQ Data (`faq_data`)**: The chatbot holds a list of predefined question-and-answer pairs covering common support queries (e.g., passwords, billing, system requirements, etc.).
2. **Preprocessing**: User input is tokenized and filtered to remove punctuation and standard English stopwords using NLTK.
3. **Embeddings**: Both the predefined FAQ questions and the user's query are converted into 384-dimensional dense vector embeddings using the `all-MiniLM-L6-v2` transformer.
4. **Similarity Matching**: The bot calculates the **Cosine Similarity** between the user's question embedding and all FAQ embeddings.
5. **Answer Retrieval**: If the highest similarity score meets or exceeds the defined threshold (`0.2`), the bot returns the corresponding FAQ answer. Otherwise, it gracefully asks the user to rephrase or contact support.

---

## 🔧 Customizing the FAQs

You can easily customize the dataset by editing the `faq_data` array inside `chatbot.py`:

```python
faq_data = [
    {"question": "Your Custom Question?", "answer": "Your Custom Answer."},
    # Add more dictionary objects as needed
]
```
