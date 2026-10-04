from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("medical_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks")
print("\nSample chunk:\n", chunks[100])