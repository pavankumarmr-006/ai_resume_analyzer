
import os
import uuid
import logging
import asyncio
import json
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

import fitz
import tiktoken

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, APITimeoutError, RateLimitError, APIError


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "5")) * 1024 * 1024
MAX_RESUME_TOKENS = int(os.getenv("MAX_RESUME_TOKENS", "10000"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

INPUT_PRICE = float(os.getenv("INPUT_COST_PER_MILLION", "0.15"))
OUTPUT_PRICE = float(os.getenv("OUTPUT_COST_PER_MILLION", "0.60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger("resume_api")
request_id_context = ContextVar("request_id", default="unknown")

app = FastAPI(
    title="AI Resume Analyzer API",
    description="API for resume extraction and job matching",
    version="1.0.0"
)

client = AsyncOpenAI(
    api_key=API_KEY or "missing-key",
    timeout=TIMEOUT,
    max_retries=0
)


# --------------------------------------------------
# 2. RESPONSE SCHEMA
# --------------------------------------------------

class AnalyzeResult(BaseModel):
    candidate_name: Optional[str] = None
    score: int = Field(ge=0, le=100)
    matched_skills: list[str]
    missing_skills: list[str]
    reasoning: str
    token_usage: dict
    estimated_cost_usd: float


# --------------------------------------------------
# 3. REQUEST ID MIDDLEWARE
# --------------------------------------------------

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):

    request_id = str(uuid.uuid4())
    token = request_id_context.set(request_id)

    try:
        logger.info(
            "Request started: %s %s",
            request.method,
            request.url.path
        )

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request completed: status=%s",
            response.status_code
        )

        return response

    finally:
        request_id_context.reset(token)


# --------------------------------------------------
# 4. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "AI Resume Analyzer",
        "version": "1.0.0"
    }


# --------------------------------------------------
# 5. RESUME TEXT EXTRACTION
# --------------------------------------------------

def extract_resume_text(filename, content):

    extension = Path(filename).suffix.lower()

    if extension not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=415,
            detail="Only PDF and TXT files are supported."
        )

    if extension == ".txt":

        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Text file must use UTF-8 encoding."
            )

    else:

        try:
            document = fitz.open(
                stream=content,
                filetype="pdf"
            )

            if document.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Password-protected PDFs are not supported."
                )

            text = "\n".join(
                page.get_text() for page in document
            )

            document.close()

        except HTTPException:
            raise

        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Unable to read the PDF file."
            )

    text = " ".join(text.split())

    if not text:
        raise HTTPException(
            status_code=422,
            detail="No text found. The PDF may be scanned or empty."
        )

    return text


# --------------------------------------------------
# 6. TOKEN COUNT
# --------------------------------------------------

def count_tokens(text):

    try:
        encoding = tiktoken.encoding_for_model(MODEL)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


# --------------------------------------------------
# 7. LLM ANALYSIS
# --------------------------------------------------

async def analyze_with_llm(resume_text, job_description):

    prompt = f"""
You are an objective AI Resume Analyzer.

Analyze the resume against the job description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return JSON with exactly these fields:

{{
    "candidate_name": null,
    "score": 0,
    "matched_skills": [],
    "missing_skills": [],
    "reasoning": ""
}}

RULES:
- Score must be between 0 and 100.
- Never invent candidate information.
- Only use evidence found in the resume.
- Do not consider protected personal characteristics.
- Use null when candidate name is unavailable.
- Return valid JSON only.
"""

    previous_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            messages = [
                {
                    "role": "system",
                    "content": "You are a fair and accurate resume analyzer. Return JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            if previous_error:
                messages.append({
                    "role": "user",
                    "content": (
                        "The previous response was invalid. "
                        "Correct this error without inventing facts: "
                        + previous_error
                    )
                })

            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("LLM returned an empty response.")

            result = json.loads(content)

            validated = {
                "candidate_name": result.get("candidate_name"),
                "score": result.get("score"),
                "matched_skills": result.get("matched_skills"),
                "missing_skills": result.get("missing_skills"),
                "reasoning": result.get("reasoning")
            }

            usage = response.usage

            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            cost = (
                input_tokens * INPUT_PRICE / 1_000_000
                + output_tokens * OUTPUT_PRICE / 1_000_000
            )

            final_result = AnalyzeResult(
                **validated,
                token_usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                },
                estimated_cost_usd=round(cost, 8)
            )

            logger.info(
                "LLM analysis successful; tokens=%s",
                total_tokens
            )

            return final_result.model_dump()

        except (APITimeoutError, RateLimitError, APIError) as error:

            previous_error = type(error).__name__

            logger.warning(
                "LLM attempt %s failed: %s",
                attempt,
                previous_error
            )

            if attempt < MAX_RETRIES:
                await asyncio.sleep(attempt)

            else:

                if isinstance(error, RateLimitError):
                    raise HTTPException(
                        status_code=503,
                        detail="AI service is busy. Please try again later."
                    )

                if isinstance(error, APITimeoutError):
                    raise HTTPException(
                        status_code=503,
                        detail="AI analysis timed out. Please try again."
                    )

                raise HTTPException(
                    status_code=502,
                    detail="AI service is temporarily unavailable."
                )

        except (json.JSONDecodeError, ValueError, TypeError) as error:

            previous_error = str(error)

            logger.warning(
                "Invalid LLM output on attempt %s",
                attempt
            )

            if attempt == MAX_RETRIES:
                raise HTTPException(
                    status_code=502,
                    detail="AI returned an invalid response. Please retry."
                )

            await asyncio.sleep(attempt)

        except Exception:

            logger.exception("Unexpected analysis failure")

            raise HTTPException(
                status_code=502,
                detail="Unable to complete resume analysis."
            )


# --------------------------------------------------
# 8. POST /analyze
# --------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(
    job_description: str = Form(...),
    resume_text: Optional[str] = Form(None),
    resume_file: Optional[UploadFile] = File(None)
):

    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI service configuration is unavailable."
        )

    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty."
        )

    if bool(resume_file) == bool(resume_text and resume_text.strip()):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one resume file or resume text."
        )

    if resume_file:

        content = await resume_file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Resume file exceeds the 5 MB size limit."
            )

        filename = resume_file.filename or ""

        resume = extract_resume_text(filename, content)

    else:
        resume = resume_text.strip()

    tokens = count_tokens(resume)

    if tokens > MAX_RESUME_TOKENS:
        raise HTTPException(
            status_code=413,
            detail="Resume exceeds the allowed token limit."
        )

    return await analyze_with_llm(
        resume,
        job_description.strip()
    )


# --------------------------------------------------
# 9. BONUS: POST /analyze-batch
# --------------------------------------------------

@app.post("/analyze-batch")
async def analyze_batch(
    files: list[UploadFile] = File(...),
    job_description: str = Form(...)
):

    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI service configuration is unavailable."
        )

    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty."
        )

    if not files or len(files) > 8:
        raise HTTPException(
            status_code=400,
            detail="Upload between 1 and 8 resumes."
        )

    semaphore = asyncio.Semaphore(3)

    async def process_file(file):

        async with semaphore:

            try:

                content = await file.read(MAX_FILE_SIZE + 1)

                if len(content) > MAX_FILE_SIZE:
                    return {
                        "filename": file.filename,
                        "status": "failed",
                        "message": "File exceeds size limit."
                    }

                resume = extract_resume_text(
                    file.filename or "",
                    content
                )

                if count_tokens(resume) > MAX_RESUME_TOKENS:
                    return {
                        "filename": file.filename,
                        "status": "failed",
                        "message": "Resume exceeds token limit."
                    }

                result = await analyze_with_llm(
                    resume,
                    job_description
                )

                return {
                    "filename": file.filename,
                    "status": "success",
                    "result": result
                }

            except HTTPException as error:

                return {
                    "filename": file.filename,
                    "status": "failed",
                    "message": error.detail
                }

    results = await asyncio.gather(
        *(process_file(file) for file in files)
    )

    return {
        "total_resumes": len(results),
        "results": results
    }


# --------------------------------------------------
# 10. ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
async def root():

    return {
        "message": "Welcome to AI Resume Analyzer API",
        "docs": "/docs",
        "health": "/health"
    }
