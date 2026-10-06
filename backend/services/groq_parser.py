import os
import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from typing import Dict

from groq import Groq

logger = logging.getLogger("ats_resume_scorer")
GROQ_MODEL = "openai/gpt-oss-120b"

_client = None


# ============================================================
# GROQ CLIENT
# ============================================================

def _get_client():
    """
    Return Groq client if GROQ_API_KEY exists.
    Return None if Groq is not configured.
    """
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.warning(
            "GROQ_API_KEY not found. Using local NLP fallback parser."
        )
        return None

    try:
        _client = Groq(api_key=api_key)
        return _client
    except Exception as e:
        logger.warning(
            f"Could not initialize Groq client: {e}. "
            "Using local NLP fallback parser."
        )
        return None


# ============================================================
# GROQ PROMPTS
# ============================================================

RESUME_SYSTEM_PROMPT = (
    "You are a resume parser. Extract information from the resume "
    "and return ONLY a valid JSON object. No explanation, no markdown."
)

RESUME_USER_PROMPT = """Extract the following from this resume and return as JSON:
{{
  "name": "full name",
  "email": "email address",
  "phone": "phone number",
  "linkedin": "LinkedIn URL if present, otherwise null",
  "github": "GitHub URL if present, otherwise null",
  "professional_summary": "the full text of the Summary, Profile, About Me, Objective, or Professional Summary section",
  "skills": ["list", "of", "skills"],
  "experience": [
    {{
      "job_title": "",
      "company": "",
      "start_date": "",
      "end_date": "",
      "duration_months": 0,
      "description": ""
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "certifications": ["list of certifications"],
  "projects": [
    {{
      "title": "project name",
      "description": "what the project does and how it was built",
      "technologies": ["tech", "used"]
    }}
  ],
  "action_verbs": ["strong action verbs used in bullet points"],
  "keywords": ["important keywords and phrases from the resume for ATS matching"]
}}

Important instructions:
- Extract ALL technical and soft skills.
- Extract important ATS keywords.
- Extract action verbs from bullet points.
- Return ONLY valid JSON.

Resume Text:
{raw_text}"""


JD_SYSTEM_PROMPT = (
    "You are a job description parser. Extract information and "
    "return ONLY a valid JSON object. No explanation, no markdown."
)

JD_USER_PROMPT = """Extract the following from this job description and return as JSON:
{{
  "job_title": "",
  "required_skills": ["list of must-have skills"],
  "preferred_skills": ["list of nice-to-have skills"],
  "experience_required": "",
  "education_required": "",
  "key_responsibilities": ["list of responsibilities"],
  "keywords": ["important keywords and phrases for ATS matching"]
}}

Important instructions:
- required_skills: skills explicitly stated as required.
- preferred_skills: skills stated as preferred or bonus.
- keywords: important technologies, skills, certifications and domain terms.
- Return ONLY valid JSON.

Job Description Text:
{raw_text}"""


# ============================================================
# GROQ CALL
# ============================================================

def _call_groq(client, system_prompt: str, user_prompt: str) -> str:

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.0,
        max_tokens=4096
    )

    return response.choices[0].message.content.strip()


def _try_parse_json(text: str) -> dict | None:

    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):

        first_newline = (
            cleaned.index("\n")
            if "\n" in cleaned
            else len(cleaned)
        )

        cleaned = cleaned[first_newline + 1:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ============================================================
# LOCAL FALLBACK UTILITIES
# ============================================================

COMMON_SKILLS = [
    "Python",
    "C++",
    "Java",
    "JavaScript",
    "TypeScript",
    "SQL",
    "HTML",
    "CSS",
    "React",
    "React.js",
    "Node.js",
    "Express",
    "MongoDB",
    "MySQL",
    "PostgreSQL",
    "Firebase",
    "FastAPI",
    "Flask",
    "Django",
    "Git",
    "GitHub",
    "Docker",
    "AWS",
    "Azure",
    "GCP",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "AI",
    "NLP",
    "Natural Language Processing",
    "Data Science",
    "Pandas",
    "NumPy",
    "scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Keras",
    "OpenCV",
    "REST API",
    "REST APIs",
    "API",
    "JWT",
    "Linux",
    "OOP",
    "Data Structures",
    "Algorithms",
    "DSA",
    "Git",
    "Agile",
    "Communication",
    "Leadership",
    "Problem Solving",
]


ACTION_VERBS = [
    "developed",
    "designed",
    "implemented",
    "created",
    "built",
    "develop",
    "design",
    "implement",
    "created",
    "optimized",
    "improved",
    "analyzed",
    "managed",
    "led",
    "deployed",
    "integrated",
    "automated",
    "engineered",
    "tested",
    "debugged",
    "configured",
    "developing",
    "building",
    "research",
    "researched",
    "developing",
]


def _extract_email(text: str):
    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )
    return match.group(0) if match else None


def _extract_phone(text: str):
    matches = re.findall(
        r"(?:\+91[\s-]?)?[6-9]\d{9}",
        text
    )
    return matches[0] if matches else None


def _extract_linkedin(text: str):
    match = re.search(
        r"(https?://)?(www\.)?linkedin\.com/[^\s]+",
        text,
        re.IGNORECASE
    )
    return match.group(0).rstrip(".,)") if match else None


def _extract_github(text: str):
    match = re.search(
        r"(https?://)?(www\.)?github\.com/[^\s]+",
        text,
        re.IGNORECASE
    )
    return match.group(0).rstrip(".,)") if match else None


def _extract_name(text: str):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    ignored = [
        "resume",
        "curriculum vitae",
        "cv",
        "profile",
        "summary",
        "contact",
    ]

    for line in lines[:8]:

        lower = line.lower()

        if any(word in lower for word in ignored):
            continue

        if "@" in line:
            continue

        if "linkedin" in lower or "github" in lower:
            continue

        if re.search(r"\d", line):
            continue

        words = line.split()

        if 2 <= len(words) <= 5 and len(line) < 60:
            return line

    return ""


def _extract_skills(text: str):

    found = []

    lower_text = text.lower()

    for skill in COMMON_SKILLS:

        if skill.lower() in lower_text:
            if skill not in found:
                found.append(skill)

    return found


def _extract_action_verbs(text: str):

    found = []
    lower_text = text.lower()

    for verb in ACTION_VERBS:

        pattern = r"\b" + re.escape(verb.lower()) + r"\b"

        if re.search(pattern, lower_text):
            if verb not in found:
                found.append(verb)

    return found


def _extract_keywords(text: str, skills):

    keywords = list(skills)

    # Extract useful capitalized technical terms
    for match in re.findall(
        r"\b[A-Z][A-Za-z0-9+#.-]{1,20}\b",
        text
    ):

        if match not in keywords and len(match) > 2:
            keywords.append(match)

    return keywords[:100]


def _extract_summary(text: str):

    lines = text.splitlines()

    section_headers = [
        "summary",
        "professional summary",
        "profile",
        "objective",
        "about me",
    ]

    collecting = False
    summary_lines = []

    for line in lines:

        clean = line.strip()

        if not clean:
            if collecting and summary_lines:
                break
            continue

        lower = clean.lower().rstrip(":")

        if any(header == lower for header in section_headers):
            collecting = True
            continue

        if collecting:

            # Stop at another common section
            if lower in [
                "education",
                "experience",
                "work experience",
                "skills",
                "projects",
                "certifications",
                "achievements",
            ]:
                break

            summary_lines.append(clean)

    return " ".join(summary_lines)


def _extract_education(text: str):

    education = []

    degree_patterns = [
        "B.Tech",
        "B.E.",
        "Bachelor",
        "Bachelors",
        "M.Tech",
        "M.E.",
        "Master",
        "MCA",
        "BCA",
        "MBA",
        "Ph.D",
        "Diploma",
    ]

    lines = text.splitlines()

    for line in lines:

        clean = line.strip()

        if not clean:
            continue

        for degree in degree_patterns:

            if degree.lower() in clean.lower():

                year_match = re.search(
                    r"\b(19|20)\d{2}\b",
                    clean
                )

                education.append({
                    "degree": clean,
                    "institution": "",
                    "year": year_match.group(0)
                    if year_match
                    else ""
                })

                break

    return education[:10]


def _extract_certifications(text: str):

    certifications = []

    lines = text.splitlines()

    collecting = False

    for line in lines:

        clean = line.strip()

        if not clean:
            continue

        lower = clean.lower()

        if "certification" in lower or "certifications" in lower:
            collecting = True
            continue

        if collecting:

            if lower in [
                "projects",
                "experience",
                "education",
                "skills",
            ]:
                break

            if len(clean) > 3:
                certifications.append(clean)

    return certifications[:20]


def _extract_projects(text: str):

    projects = []

    lines = text.splitlines()

    collecting = False

    current = None

    for line in lines:

        clean = line.strip()

        if not clean:
            continue

        lower = clean.lower()

        if lower in ["projects", "project", "academic projects"]:
            collecting = True
            continue

        if collecting and lower in [
            "experience",
            "work experience",
            "education",
            "skills",
            "certifications",
            "achievements",
        ]:
            break

        if collecting:

            # Treat short lines as project titles
            if len(clean) < 100 and not clean.startswith(("•", "-", "*")):

                if current:
                    projects.append(current)

                current = {
                    "title": clean,
                    "description": "",
                    "technologies": []
                }

            elif current:

                clean_description = clean.lstrip("•-* ")

                current["description"] += (
                    " " + clean_description
                )

    if current:
        projects.append(current)

    return projects[:20]


# ============================================================
# LOCAL RESUME PARSER
# ============================================================

def _local_parse_resume(raw_text: str) -> Dict:

    logger.info("Using local NLP resume parser.")

    skills = _extract_skills(raw_text)
    action_verbs = _extract_action_verbs(raw_text)

    result = {
        "name": _extract_name(raw_text),
        "email": _extract_email(raw_text),
        "phone": _extract_phone(raw_text),
        "linkedin": _extract_linkedin(raw_text),
        "github": _extract_github(raw_text),
        "professional_summary": _extract_summary(raw_text),
        "skills": skills,
        "experience": [],
        "education": _extract_education(raw_text),
        "certifications": _extract_certifications(raw_text),
        "projects": _extract_projects(raw_text),
        "action_verbs": action_verbs,
        "keywords": _extract_keywords(raw_text, skills),
    }

    return _validate_resume_result(result)


# ============================================================
# RESUME PARSER
# ============================================================

def parse_resume(raw_text: str) -> Dict:

    client = _get_client()

    # No Groq → local parser
    if client is None:
        return _local_parse_resume(raw_text)

    prompt = RESUME_USER_PROMPT.format(raw_text=raw_text)

    try:

        raw_response = _call_groq(
            client,
            RESUME_SYSTEM_PROMPT,
            prompt
        )

        result = _try_parse_json(raw_response)

        if result is not None:
            return _validate_resume_result(result)

        logger.warning(
            "Groq resume parse returned invalid JSON. Retrying."
        )

        strict_prompt = (
            "Your previous response was not valid JSON. "
            "Return ONLY the raw JSON object.\n\n"
            + prompt
        )

        raw_response = _call_groq(
            client,
            RESUME_SYSTEM_PROMPT,
            strict_prompt
        )

        result = _try_parse_json(raw_response)

        if result is not None:
            return _validate_resume_result(result)

        logger.warning(
            "Groq returned invalid JSON twice. "
            "Using local parser."
        )

    except Exception as e:

        logger.warning(
            f"Groq resume parsing failed: {e}. "
            "Using local parser."
        )

    return _local_parse_resume(raw_text)


# ============================================================
# LOCAL JOB DESCRIPTION PARSER
# ============================================================

def _local_parse_job_description(raw_text: str) -> Dict:

    logger.info("Using local NLP job-description parser.")

    skills = _extract_skills(raw_text)

    keywords = _extract_keywords(
        raw_text,
        skills
    )

    lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    job_title = ""

    title_keywords = [
        "software engineer",
        "software developer",
        "data scientist",
        "data analyst",
        "machine learning engineer",
        "frontend developer",
        "backend developer",
        "full stack developer",
        "web developer",
        "intern",
        "developer",
    ]

    for line in lines[:15]:

        lower = line.lower()

        if any(
            keyword in lower
            for keyword in title_keywords
        ):
            job_title = line
            break

    required_skills = []
    preferred_skills = []

    lower_text = raw_text.lower()

    for skill in skills:

        escaped = re.escape(skill.lower())

        required_patterns = [
            rf"required.*{escaped}",
            rf"{escaped}.*required",
            rf"must have.*{escaped}",
            rf"mandatory.*{escaped}",
        ]

        preferred_patterns = [
            rf"preferred.*{escaped}",
            rf"{escaped}.*preferred",
            rf"nice to have.*{escaped}",
            rf"bonus.*{escaped}",
        ]

        if any(
            re.search(pattern, lower_text)
            for pattern in preferred_patterns
        ):
            preferred_skills.append(skill)

        elif any(
            re.search(pattern, lower_text)
            for pattern in required_patterns
        ):
            required_skills.append(skill)

        else:
            required_skills.append(skill)

    return _validate_jd_result({
        "job_title": job_title,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "experience_required": "",
        "education_required": "",
        "key_responsibilities": [],
        "keywords": keywords,
    })


# ============================================================
# JOB DESCRIPTION PARSER
# ============================================================

def parse_job_description(raw_text: str) -> Dict:

    client = _get_client()

    # No Groq → local parser
    if client is None:
        return _local_parse_job_description(raw_text)

    prompt = JD_USER_PROMPT.format(
        raw_text=raw_text
    )

    try:

        raw_response = _call_groq(
            client,
            JD_SYSTEM_PROMPT,
            prompt
        )

        result = _try_parse_json(raw_response)

        if result is not None:
            return _validate_jd_result(result)

        logger.warning(
            "Groq JD parser returned invalid JSON. Retrying."
        )

        strict_prompt = (
            "Your previous response was not valid JSON. "
            "Return ONLY the raw JSON object.\n\n"
            + prompt
        )

        raw_response = _call_groq(
            client,
            JD_SYSTEM_PROMPT,
            strict_prompt
        )

        result = _try_parse_json(raw_response)

        if result is not None:
            return _validate_jd_result(result)

    except Exception as e:

        logger.warning(
            f"Groq JD parsing failed: {e}. "
            "Using local parser."
        )

    return _local_parse_job_description(raw_text)


# ============================================================
# VALIDATION
# ============================================================

def _validate_jd_result(result: dict) -> dict:

    defaults = {
        "job_title": "",
        "required_skills": [],
        "preferred_skills": [],
        "experience_required": "",
        "education_required": "",
        "key_responsibilities": [],
        "keywords": [],
    }

    for key, default in defaults.items():

        if key not in result or result[key] is None:
            result[key] = default

        if (
            isinstance(default, list)
            and not isinstance(result[key], list)
        ):
            result[key] = default

    return result


def _validate_resume_result(result: dict) -> dict:

    defaults = {
        "name": "",
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "professional_summary": "",
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "action_verbs": [],
        "keywords": [],
    }

    for key, default in defaults.items():

        if key not in result or result[key] is None:
            result[key] = default

        if (
            isinstance(default, list)
            and not isinstance(result[key], list)
        ):
            result[key] = default

    # Validate experience entries
    for exp in result.get("experience", []):

        if not isinstance(exp, dict):
            continue

        exp.setdefault("job_title", "")
        exp.setdefault("company", "")
        exp.setdefault("start_date", "")
        exp.setdefault("end_date", "")
        exp.setdefault("duration_months", 0)
        exp.setdefault("description", "")

        try:
            exp["duration_months"] = int(
                exp["duration_months"]
            )
        except (ValueError, TypeError):
            exp["duration_months"] = 0

    # Validate project entries
    for proj in result.get("projects", []):

        if not isinstance(proj, dict):
            continue

        proj.setdefault("title", "")
        proj.setdefault("description", "")
        proj.setdefault("technologies", [])

        if not isinstance(
            proj["technologies"],
            list
        ):
            proj["technologies"] = []

    return result