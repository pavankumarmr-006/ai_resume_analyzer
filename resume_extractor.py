
import json
import os
from pathlib import Path

from openai import OpenAI
from jsonschema import validate, ValidationError

from resume_parser import read_resume


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"

SCHEMA_PATH = Path("schemas/resume_schema.json")
PROMPT_PATH = Path("prompts/resume_extraction_v1.txt")
RESUME_FOLDER = Path("resumes")
OUTPUT_FOLDER = Path("outputs")


def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8")


def extract_resume(resume_text, prompt, schema):
    system_prompt = prompt.replace(
        "{{RESUME_TEXT}}",
        resume_text
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt
            }
        ]
    )

    result = json.loads(
        response.choices[0].message.content
    )

    # Validate extracted JSON against schema
    validate(instance=result, schema=schema)

    return result


def main():
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    schema = load_schema()
    prompt = load_prompt()

    resume_files = sorted(
        list(RESUME_FOLDER.glob("*.pdf")) +
        list(RESUME_FOLDER.glob("*.txt"))
    )

    all_results = []

    for resume_file in resume_files:

        print(f"\nProcessing: {resume_file.name}")

        try:
            resume_text = read_resume(resume_file)

            extracted_data = extract_resume(
                resume_text,
                prompt,
                schema
            )

            all_results.append({
                "filename": resume_file.name,
                "extracted_data": extracted_data
            })

            print(json.dumps(
                extracted_data,
                indent=4
            ))

            print("Extraction successful!")

        except (ValueError, ValidationError, json.JSONDecodeError) as error:
            print(f"Extraction error: {error}")

        except Exception as error:
            print(f"Unexpected error: {error}")

    output_path = OUTPUT_FOLDER / "extracted_resumes.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
