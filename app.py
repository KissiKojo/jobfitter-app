# app.py

import os
import io
import json
import pdfplumber
from docx import Document

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import openai

# 1. Instantiate FastAPI
app = FastAPI()

# 2. Enable CORS for your front-end (http://127.0.0.1:5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",                 # local dev
        "https://jobfitter-app.onrender.com",    # Render API (if serving front-end there)
        "https://kissikojo.github.io"            # your GitHub Pages origin
        # You could also use "*" to allow any origin during dev:
        # "*"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Load your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 4. Your career-coach system prompt (paste your full prompt here)
SYSTEM_PROMPT = """You are a career-coach AI for job seekers. You will receive:
• A job description (text)
• A résumé file, a cover letter file, and a personal bio file

Your tasks:
1. Assign "fit_score" (0–100).
2. List at least 3 "strengths": exact phrases matching job requirements.
3. List at least 3 "gaps": keywords from the ad missing in the materials.
4. Provide an actionable "suggestion" to improve materials.
5. Rewrite the cover letter (or generate one if none).
6. Rewrite the résumé (or generate one if none).
7. Incorporate relevant details from the bio into both rewrites.

Always output valid JSON with keys:
{
  "fit_score": int,
  "strengths": [string],
  "gaps": [string],
  "suggestion": string,
  "rewritten_cover_letter": string,
  "rewritten_resume": string
}
"""

async def extract_text(file: UploadFile) -> str:
    """Extract plain text from an uploaded PDF, DOCX, or TXT."""
    data = await file.read()

    if file.content_type == "application/pdf":
        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    if "wordprocessingml" in file.content_type:
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text)

    # fallback for .txt or unknown
    return data.decode("utf-8", errors="ignore")


@app.post("/analyze")
async def analyze(
    job_text: str = Form(...),
    resume_file: UploadFile = File(None),
    cover_file:  UploadFile = File(None),
    bio_file:    UploadFile = File(None),
):
    # ─── 1) Extract inputs ─────────────────────────────
    resume_text = await extract_text(resume_file) if resume_file else "<none>"
    cover_text  = await extract_text(cover_file)  if cover_file  else "<none>"
    bio_text    = await extract_text(bio_file)    if bio_file    else "<none>"

    # ─── 2) Log the raw inputs ─────────────────────────
    print("=== ANALYZE CALLED ===")
    print("Job text:", job_text[:100], "…")
    print("Resume excerpt:", resume_text[:100], "…")
    print("Cover excerpt:", cover_text[:100], "…")
    print("Bio excerpt:", bio_text[:100], "…")

    # ─── 3) Build the AI prompt ────────────────────────
    user_prompt = f"""
Job Description:
{job_text}

Résumé:
{resume_text}

Cover Letter:
{cover_text}

Personal Bio:
{bio_text}

Please perform tasks 1–7 and output valid JSON.
"""
    # Log the full prompt about to go to the model
    print("=== PROMPT TO AI ===")
    print(user_prompt)

    # ─── 4) Call OpenAI ────────────────────────────────
    try:
        resp = openai.chat.completions.create(
            model="gpt-4.1",
            temperature=0.2,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_prompt}
            ]
        )
        # Log the raw AI response
        ai_text = resp.choices[0].message.content
        print("=== RAW AI RESPONSE ===")
        print(ai_text)

        # ─── 5) Parse & return ───────────────────────────
        result = json.loads(ai_text)
        print("=== PARSED RESULT ===")
        print(result)
        return result

    except Exception as e:
        # Log the error before returning
        print("=== OPENAI ERROR ===", e)
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

