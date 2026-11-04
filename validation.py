
from pydantic import BaseModel, Field, ValidationError
from typing import Optional


class ResumeData(BaseModel):
    candidate_name: Optional[str] = None

    skills: list[str] = Field(min_length=1)

    years_of_experience: Optional[int] = Field(
        default=None,
        ge=0,
        le=50
    )

    education: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


def validate_resume(json_data):

    try:
        resume = ResumeData.model_validate_json(json_data)
        print("Validation successful!")
        return resume

    except ValidationError as error:
        print("Validation failed:")
        print(error)
        return None
