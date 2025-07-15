import PyPDF2
from pathlib import Path

def extract_pdf_chunks(file_path, output_dir, max_pages=5):
    reader = PyPDF2.PdfReader(file_path)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    chunks = []
    for i in range(0, len(reader.pages), max_pages):
        writer = PyPDF2.PdfWriter()
        for j in range(i, min(i + max_pages, len(reader.pages))):
            writer.add_page(reader.pages[j])
        chunk_path = path / f"{Path(file_path).stem}_chunk_{i//max_pages + 1}.pdf"
        with open(chunk_path, 'wb') as f:
            writer.write(f)
        chunks.append(str(chunk_path))
    return chunks

def extract_text_from_pdf(path):
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
