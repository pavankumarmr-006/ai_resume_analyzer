
import fitz  # PyMuPDF
import re
import sys
import tiktoken
from pathlib import Path


def clean_text(text):
    """Remove unnecessary spaces, headers and page numbers."""

    # Normalize line endings
    text = text.replace("\r", "\n")

    # Remove common page number formats
    text = re.sub(
        r"(?im)^\s*(page\s+)?\d+\s*(of\s+\d+)?\s*$",
        "",
        text
    )

    # Remove excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    # Remove repeated header/footer lines
    # (lines that appear multiple times)
    counts = {}
    for line in lines:
        counts[line.lower()] = counts.get(line.lower(), 0) + 1

    lines = [
        line for line in lines
        if counts[line.lower()] < 3
    ]

    text = "\n".join(lines)

    # Normalize extra spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def read_resume(path):
    """
    Extract and clean resume text from PDF or TXT files.
    Returns cleaned text.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        extracted_pages = []

        try:
            with fitz.open(path) as pdf:
                for page in pdf:
                    extracted_pages.append(page.get_text())

        except Exception as e:
            raise ValueError(f"Unable to read PDF: {e}")

        raw_text = "\n".join(extracted_pages)

        if not raw_text.strip():
            raise ValueError(
                "Error - no text found (scanned or empty PDF)"
            )

    elif extension == ".txt":
        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = path.read_text(encoding="latin-1")

    else:
        raise ValueError(
            "Unsupported file format. Use PDF or TXT."
        )

    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        raise ValueError("Error - resume is empty after cleaning")

    return cleaned_text


def count_tokens(text):
    """Count tokens using the cl100k_base tokenizer."""

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def analyze_resume(path):
    """Extract resume and report character and token counts."""

    filename = Path(path).name

    try:
        text = read_resume(path)

        characters = len(text)
        tokens = count_tokens(text)

        print(
            f"{filename}: {characters} characters, "
            f"{tokens} tokens"
        )

        return {
            "filename": filename,
            "characters": characters,
            "tokens": tokens,
            "text": text
        }

    except (FileNotFoundError, ValueError) as error:
        print(f"{filename}: Error - {error}")
        return None


def main():
    resume_folder = Path("resumes")

    if not resume_folder.exists():
        print("Error - resumes folder not found")
        sys.exit(1)

    resume_files = sorted(
        list(resume_folder.glob("*.pdf")) +
        list(resume_folder.glob("*.txt"))
    )

    if not resume_files:
        print("No PDF or TXT resumes found.")
        return

    print("\nAI RESUME ANALYZER - TASK 1")
    print("-" * 45)

    for resume in resume_files:
        analyze_resume(resume)


if __name__ == "__main__":
    main()
