"""Extraction prompts for Turkish CVs.

Modelled on the instruction format of Zhu et al. (2025), Fig. 7:
    Instruction (role + task + JSON template with a comment per field)
    /no_think
    Input (Indexed Resume Text): [0]: ... [1]: ...
    Output: {...}

Differences from the paper:
- Turkish CVs, so the comments explain Turkish formats (months, "Halen", degree names, GPA).
- No age, gender or place of origin (sensitive personal data under KVKK/GDPR).
- Skills and languages are added to the basic-info task, so we still run three tasks.
- Work descriptions use the paper's pointer idea: the model returns a line range, not text.
"""

from __future__ import annotations

import json

ROLE = (
    "You are a professional CV analysis assistant. Your task is to convert the given "
    "CV text into the JSON format specified below. The CV is usually in Turkish. "
    "(If the CV contains both a Turkish and an English version, extract only from the Turkish one.)"
)

COMMON_RULES = (
    "Copy names, schools, companies and titles exactly as written in the CV, keeping Turkish "
    "characters (ç, ğ, ı, İ, ö, ş, ü). Never guess or add information that is not in the CV."
)

DATE_RULES = (
    'Dates use "YYYY-MM" (e.g. "Eylül 2019" -> "2019-09", "09/2019" -> "2019-09"). '
    'If only the year is given, use "YYYY". If ongoing ("Halen", "Devam ediyor", "Present", '
    '"Günümüz"), use "present". Turkish months: Ocak=01, Şubat=02, Mart=03, Nisan=04, '
    "Mayıs=05, Haziran=06, Temmuz=07, Ağustos=08, Eylül=09, Ekim=10, Kasım=11, Aralık=12."
)

BASIC_TEMPLATE = """Extract the following information into a JSON object. If a field does not exist, output an empty string "". If a list is empty, output [].
{
  "basicInfo": {
    "name": "",             # Full name, e.g. "Ayşe Yılmaz"
    "email": "",            # Email address, e.g. "ayse.yilmaz@gmail.com"
    "phone": "",            # Phone number in its original format, e.g. "+90 532 123 45 67"
    "currentLocation": "",  # Current city, e.g. "İstanbul". Only if explicitly stated; do NOT infer it from schools or jobs.
    "headline": "",         # Job title or target position written at the top, e.g. "Yazılım Mühendisi". Leave empty if none.
    "links": []             # Profile links, e.g. ["linkedin.com/in/ayseyilmaz", "github.com/ayse"]
  },
  "skills": [],             # Skills as written, one per item, e.g. ["Python", "SQL", "Docker"]
  "languages": [            # Spoken languages. If none are mentioned, return [].
    {
      "language": "",       # e.g. "İngilizce"
      "level": ""           # Level as written, e.g. "İleri", "B2", "Anadil". Empty if not given.
    }
  ]
}"""

EDUCATION_TEMPLATE = """Extract all education entries into a JSON array. If a field does not exist, output an empty string "". If no education is mentioned, return [].
{
  "education": [
    {
      "school": "",      # Full name as written, e.g. "Orta Doğu Teknik Üniversitesi"
      "degree": "",      # One of: "HighSchool" (Lise), "Associate" (Önlisans), "Bachelor" (Lisans), "Master" (Yüksek Lisans), "PhD" (Doktora), "Other". Empty if unclear.
      "major": "",       # Field of study as written, e.g. "Bilgisayar Mühendisliği"
      "startDate": "",   # "YYYY-MM" or "YYYY"
      "endDate": "",     # "YYYY-MM", "YYYY" or "present"
      "gpa": "",         # Grade as written, including its scale, e.g. "3.45/4.00" or "85/100". Empty if not given.
      "location": ""     # City or country of the school, e.g. "Ankara". Only if written in the CV.
    }
  ]
}"""

WORK_TEMPLATE = """Extract all work experience entries (including internships) into a JSON array. If a field does not exist, output an empty string "". If no work experience is mentioned, return [].
{
  "workExperience": [
    {
      "company": "",          # Company name as written, e.g. "Trendyol"
      "position": "",         # Job title as written, e.g. "Yazılım Mühendisi" or "Stajyer"
      "startDate": "",        # "YYYY-MM" or "YYYY"
      "endDate": "",          # "YYYY-MM", "YYYY" or "present"
      "location": "",         # City, e.g. "İstanbul". Only if written for this job.
      "descriptionLines": []  # Line numbers [first, last] of the description lines for this job, e.g. [16, 18]. Do NOT copy the text. [] if there is no description.
    }
  ]
}"""

TASKS = {
    "basic": BASIC_TEMPLATE,
    "education": EDUCATION_TEMPLATE,
    "work": WORK_TEMPLATE,
}


def index_lines(lines: list[str]) -> str:
    """Number the CV lines like the paper: "[0]: ... [1]: ...", one line each."""
    return "\n".join(f"[{i}]: {line}" for i, line in enumerate(lines))


def build_prompt(task: str, lines: list[str], no_think: bool = True) -> str:
    """The full prompt for one extraction task (what the model sees at inference)."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}, expected one of {list(TASKS)}")
    rules = COMMON_RULES if task == "basic" else f"{COMMON_RULES} {DATE_RULES}"
    parts = [ROLE, rules, TASKS[task]]
    if no_think:
        parts.append("/no_think")
    parts.append("Input (Indexed CV Text):\n" + index_lines(lines))
    parts.append("Output:")
    return "\n\n".join(parts)


def build_sft_sample(task: str, lines: list[str], label: dict) -> dict:
    """One (instruction, input, output) training triplet, as in the paper's Appendix B."""
    prompt = build_prompt(task, lines)
    instruction, _, rest = prompt.partition("\n\nInput (Indexed CV Text):\n")
    return {
        "instruction": instruction,
        "input": rest.removesuffix("\n\nOutput:"),
        "output": json.dumps(label, ensure_ascii=False, indent=2),
    }


def parse_json_output(text: str) -> dict | None:
    """Pull the JSON out of a model answer the way the paper does: first "{" to last "}"."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def resolve_pointers(work: dict, lines: list[str]) -> dict:
    """Replace descriptionLines [a, b] with the exact text of lines a..b (grounded re-extraction)."""
    for job in work.get("workExperience", []):
        rng = job.get("descriptionLines") or []
        if len(rng) == 2 and 0 <= rng[0] <= rng[1] < len(lines):
            job["description"] = "\n".join(lines[rng[0] : rng[1] + 1])
        else:
            job["description"] = ""
    return work
