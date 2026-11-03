
import fitz
from pathlib import Path

folder = Path("resumes")
folder.mkdir(exist_ok=True)

resumes = [
    ("resume_01.pdf", "Software Engineer"),
    ("resume_02.pdf", "Data Analyst"),
    ("resume_03.txt", "Python Developer"),
    ("resume_04.pdf", "Web Developer"),
    ("resume_05.txt", "Java Developer"),
    ("resume_06.pdf", "Machine Learning Engineer"),
    ("resume_07.txt", "Business Analyst"),
    ("resume_08.pdf", "Cloud Engineer"),
]

for index, (filename, role) in enumerate(resumes, start=1):

    content = f"""
CANDIDATE {index}
Name: Dummy Candidate {index}
Email: candidate{index}@example.com

PROFESSIONAL SUMMARY
Motivated {role} with experience in software development,
problem solving and technology.

SKILLS
Python, Java, SQL, Git, HTML, CSS, Communication

WORK EXPERIENCE
Software Professional
ABC Technologies
2022 - 2025

Worked on development projects, debugging,
testing and team collaboration.

EDUCATION
Bachelor of Engineering
Computer Science
2022

PROJECTS
Developed applications using Python and SQL.
Worked with databases and modern development tools.

CERTIFICATIONS
Python Programming Certification
"""

    path = folder / filename

    if filename.endswith(".txt"):
        path.write_text(content, encoding="utf-8")

    else:
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            content,
            fontsize=11
        )
        pdf.save(path)
        pdf.close()

    print(f"Created: {filename}")

print("\n8 dummy resumes created successfully!")
