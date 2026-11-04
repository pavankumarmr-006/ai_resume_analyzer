
import json
import pandas as pd

from job_matching import JOBS
from scoring import keyword_match_score


# Load extracted resumes from Part 1
with open("outputs/extracted_resumes.json", "r") as file:
    resumes = json.load(file)


print("Loaded resumes successfully!")


# Display available resumes
print("Number of resumes:", len(resumes))


# Display job descriptions
for job_name, job in JOBS.items():
    print("\nJob:", job_name)
    print("Required skills:", job["required_skills"])


# Keyword scoring
results = []

for index, resume in enumerate(resumes, start=1):

    skills = resume.get("skills", [])

    for job_name, job in JOBS.items():

        score = keyword_match_score(
            skills,
            job["required_skills"]
        )

        results.append({
            "Candidate": resume.get(
                "candidate_name", f"Resume {index}"
            ),
            "Job": job_name,
            "Keyword Score": score
        })


df = pd.DataFrame(results)

print("\nKEYWORD MATCHING RESULTS")
print(df)

df.to_csv("outputs/rankings.csv", index=False)

print("\nResults saved to outputs/rankings.csv")
