# Medical RAG Assistant

A chat assistant that answers medical questions using only the contents of a medical reference book (PDF). Instead of relying on an LLM's general knowledge, it retrieves the most relevant passages from the book first and then generates an answer from them. If the book doesn't cover the question, it says so.

<!-- Add a screenshot: save it as screenshot.png in this folder, then uncomment the next line -->
<!-- ![Screenshot](screenshot.png) -->

## How it works

1. **PDF extraction:** PyMuPDF pulls the text out of the PDF (`pdf_reader.py`).
2. **Chunking:** LangChain's `RecursiveCharacterTextSplitter` splits the text into 1000-character chunks with 200 overlap.
3. **Embeddings:** each chunk is converted to a 384-dimension vector with `all-MiniLM-L6-v2` (Hugging Face, via sentence-transformers).
4. **Vector search:** vectors are stored in Pinecone (`upload.py`). At question time, the 4 most similar chunks are retrieved.
5. **Answer generation:** the retrieved chunks and the question are sent to Groq (`openai/gpt-oss-120b`) with instructions to answer only from that context.
6. **Interface:** a Flask backend (`app.py`) serves a single-page chat UI (`templates/index.html`).

## Tech stack

Python, Flask, PyMuPDF, LangChain text splitters, sentence-transformers, Pinecone, Groq

## Setup

1. Clone the repo and create a virtual environment:

   ```
   git clone https://github.com/rakshabhardwaj/medical-rag-assistant.git
   cd medical-rag-assistant
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file (see `.env.example`) with your keys:

   ```
   PINECONE_API_KEY=your-pinecone-key
   GROQ_API_KEY=your-groq-key
   ```

3. Put your PDF at `uploads/Medical_book.pdf`.

4. Extract the text, then upload it to Pinecone (one time only):

   ```
   python pdf_reader.py
   python upload.py
   ```

5. Start the app and open http://127.0.0.1:5000:

   ```
   python app.py
   ```

## Notes

- If the PDF is scanned (no selectable text), `pdf_reader.py` will need OCR instead of plain text extraction.
- The Pinecone index must use dimension 384 and the cosine metric to match the embedding model.
- The book itself is not included in this repo.
- For study and reference only. Not a substitute for professional medical advice.
