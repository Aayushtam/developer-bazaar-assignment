from pypdf import PdfReader

try:
    reader = PdfReader("AI-Intern_Jr-AI-Developer_Assessment.pdf")
    for page in reader.pages:
        print(page.extract_text())
except Exception as e:
    print(f"Error reading PDF: {e}")
