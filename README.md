# LangGraph RAG App with Astra DB and Wikipedia Search

This is a LangChain + LangGraph + FastAPI-based RAG (Retrieval-Augmented Generation) app that lets you upload a set of URLs, stores their content into a vector database (Astra DB), and intelligently answers questions by choosing between searching Wikipedia or your uploaded content.

---

## 🧠 What This App Does

1. **Upload URLs**  
   You provide URLs of blog posts or articles.  
   Example:
   ```json
   {
     "urls": [
       "https://lilianweng.github.io/posts/2023-06-23-agent/",
       "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
     ]
   }
    ````

2. **Ingest & Store**
   The content of those URLs is:

   * Downloaded
   * Split into chunks
   * Converted to embeddings
   * Stored in Astra Vector DB

3. **Ask Questions**
   When you ask a question, the system:

   * Uses an LLM to decide: "Should I use Wikipedia or the uploaded documents?"
   * If it's related to the uploaded topics, it fetches from Astra DB.
   * Otherwise, it searches Wikipedia and returns an answer.

---

## 🚀 How to Run the App

### 1. Clone the repository

```bash
git clone https://github.com/jayraj1906/langgraph-rag-app.git
cd langgraph-rag-app
```

### 2. Install dependencies

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 3. Create a `.env` file

Create a `.env` file in the root directory with the following keys:

```env
ASTRA_DB_APPLICATION_TOKEN=your_astra_token
ASTRA_DB_ID=your_db_id
ASTRA_DB_KEYSPACE=your_keyspace
GROQ_API_KEY=your_groq_api_key
```

Replace the values with your actual credentials.

### 4. Run the FastAPI server

```bash
python main.py
```

This will start the API at: `http://127.0.0.1:8000`

---

## 📡 API Endpoints

### 🔹 `/ingest` (POST)

Upload URLs to ingest and store content.

**Request body:**

```json
{
  "urls": [
    "https://example.com/article1",
    "https://example.com/article2"
  ]
}
```

### 🔹 `/predict` (POST)

Ask a question. The system will choose to either use RAG or Wikipedia.

**Query param:**

```
/predict?question=What is prompt engineering?
```

### 🔹 `/predict-stream` (POST)

Same as `/predict`, but runs the internal LangGraph graph in a streaming fashion (logged to console).

---

## 🧰 Tech Stack

* **LangChain + LangGraph** for reasoning and document workflow
* **FastAPI** for serving the application
* **Astra DB Vector Store** for document storage
* **Wikipedia API** for fallback external search
* **Groq API** with Gemma model for routing logic

---

## 📝 Example Flow

1. You call `/ingest` with 3 blog post URLs.
2. Later, you ask:

   ```
   /predict?question=What are LLM agents?
   ```
3. LLM sees it's related to your uploaded content, fetches the best match from Astra DB, and returns the answer.
4. For unrelated questions, like:

   ```
   /predict?question=Who was Albert Einstein?
   ```

   It will search Wikipedia instead.


## 💬 Questions?

Feel free to open an issue or contact me on [GitHub](https://github.com/jayraj1906).

