---

## 📄 DocumentCompanion

**DocumentCompanion** is an AI-powered web application built with **Streamlit** that helps users quickly extract, summarize, and interact with the content of uploaded documents.

---

### 🚀 Features

* 📁 Upload documents (PDF, TXT, etc.)
* 📄 View and preview content in-app
* 🧠 Summarize large text automatically
* 🔍 Extract keywords and main points
* 💬 Ask questions and get answers from your document

---

### ⚙️ Tech Stack

* Python 3
* Streamlit
* NLP libraries (e.g., spaCy or NLTK)
* PDF/Text processing tools
* dotenv for environment variables

---

### 🔧 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/DocumentCompanion.git
cd DocumentCompanion

# (Optional) Create virtual environment
uv venv
uv pip install -r requirements.txt

# Run the app
uv run streamlit run app.py
```

---

### 🔒 Environment Variables

To store API keys and config values, create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

`.env` is included in `.gitignore` and won’t be tracked.

---

### 📌 To-Do

* Add OCR support for scanned PDFs
* Improve UI/UX
* Add document saving/export options

---

