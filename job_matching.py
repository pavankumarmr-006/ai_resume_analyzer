
JOBS = {
    "Python Developer": {
        "required_skills": [
            "Python",
            "Django",
            "Flask",
            "REST API",
            "SQL",
            "Git"
        ],
        "experience": "2-4 years"
    },

    "Data Analyst": {
        "required_skills": [
            "Python",
            "SQL",
            "Excel",
            "Power BI",
            "Tableau",
            "Statistics"
        ],
        "experience": "1-3 years"
    }
}


def create_matching_prompt(resume, job):

    return f"""
    You are an objective resume matching assistant.

    Compare the candidate resume with this job.

    RESUME:
    {resume}

    JOB:
    {job}

    Return valid JSON only:

    {{
        "score": 0,
        "matched_skills": [],
        "missing_skills": [],
        "reasoning": ""
    }}

    Constraints:
    - Score must be between 0 and 100.
    - Never invent skills or experience.
    - Use only resume evidence.
    - Evaluate all candidates consistently.
    """
