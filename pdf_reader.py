import pymupdf

pdf_path = "uploads/Medical_book.pdf"
doc = pymupdf.open(pdf_path)

print("Total pages:", len(doc))

all_text = ""

for page_number in range(len(doc)):
    print(f"Reading page {page_number + 1}/{len(doc)}...")
    page = doc[page_number]
    text = page.get_text()
    all_text += f"\n\n--- PAGE {page_number + 1} ---\n\n" + text

with open("medical_text.txt", "w", encoding="utf-8") as file:
    file.write(all_text)

print("\nDONE! 🎉")
print("Text saved to medical_text.txt")