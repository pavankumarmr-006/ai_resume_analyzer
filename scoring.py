
def keyword_match_score(resume_skills, required_skills):

    resume_skills = {
        skill.lower() for skill in resume_skills
    }

    required_skills = {
        skill.lower() for skill in required_skills
    }

    if not required_skills:
        return 0

    matched = resume_skills.intersection(required_skills)

    score = (
        len(matched) / len(required_skills)
    ) * 100

    return round(score, 2)
