from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-rag"

# 1. Load and chunk
with open("medical_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)
print(f"{len(chunks)} chunks")

# 2. Connect to Pinecone (create the index if it doesn't exist)
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 outputs 384 dims
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# 3. Embed and upload in batches
model = SentenceTransformer("all-MiniLM-L6-v2")
BATCH = 100

for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i + BATCH]
    embeddings = model.encode(batch).tolist()
    vectors = [
        (f"chunk-{i + j}", emb, {"text": chunk})
        for j, (emb, chunk) in enumerate(zip(embeddings, batch))
    ]
    index.upsert(vectors=vectors)
    print(f"Uploaded {min(i + BATCH, len(chunks))}/{len(chunks)}")

print("DONE!")
