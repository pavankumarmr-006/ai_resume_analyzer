
import json
from validation import ResumeData
from pydantic import ValidationError


def validate_with_retry(llm_function, resume_text):

    max_attempts = 3
    previous_error = None
    logs = []

    for attempt in range(1, max_attempts + 1):

        response = llm_function(
            resume_text=resume_text,
            correction_error=previous_error
        )

        usage = response.get("usage", {})

        logs.append({
            "attempt": attempt,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        })

        try:
            data = ResumeData.model_validate_json(
                response["content"]
            )

            print(f"Attempt {attempt}: success")

            return {
                "status": "success",
                "data": data.model_dump(),
                "logs": logs
            }

        except ValidationError as error:

            previous_error = str(error)

            print(f"Attempt {attempt}: validation error")
            print(previous_error)

    return {
        "status": "failed",
        "message": "Validation failed after 3 attempts",
        "logs": logs
    }
