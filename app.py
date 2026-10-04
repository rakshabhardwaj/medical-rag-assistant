import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "medical-rag"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4
app = Flask(__name__)

# Load everything once at startup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)
llm = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a medical reference assistant. Answer the question using ONLY the "
    "context passages provided from the medical book. If the context does not "
    "contain the answer, say you could not find it in the book. Be clear and "
    "concise. Do not invent facts."
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # 1. Embed the question and find the most similar chunks
    vector = embedder.encode(question).tolist()
    results = index.query(vector=vector, top_k=TOP_K, include_metadata=True)
    context = "\n\n---\n\n".join(m["metadata"]["text"] for m in results["matches"])

    # 2. Ask the LLM, grounded in those chunks
    completion = llm.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return jsonify({"answer": completion.choices[0].message.content})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


