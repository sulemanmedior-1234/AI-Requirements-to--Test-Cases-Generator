"""
AI Requirements-to-Test-Cases Generator
Enter software requirements. The AI generates normal, boundary and invalid-input
test cases with expected outcomes. Review and edit them, then export to CSV.

Design notes:
  - Requirements analysis: the AI also lists ambiguities and missing details.
  - Structured generation: the AI answers in JSON, which the app validates.
  - Boundary values are computed by the code from the numbers in each requirement
    (for example "between 8 and 20" gives 7, 8, 9, 19, 20, 21). They are given to the
    AI, and the app then checks that the boundary tests really use them.
  - Test IDs, per-type limits and coverage counts are handled by the code.
  - CSV export opens correctly in Excel.

Uses the Groq REST API directly through `requests`, so the `groq` package is
not needed.

Run locally:  streamlit run app.py
"""

import io
import json
import os
import re
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Requirements to Test Cases", page_icon="🧪", layout="wide")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
TYPES = ["Normal", "Boundary", "Invalid"]
PRIORITIES = ["High", "Medium", "Low"]
COLUMNS = [
    "Test ID", "Requirement ID", "Type", "Priority", "Title",
    "Preconditions", "Test Data", "Steps", "Expected Result",
]
BATCH_SIZE = 3
MAX_REQUIREMENTS = 30
MAX_INPUT_CHARS = 12000

SAMPLE_REQUIREMENTS = """REQ-1: The password must be between 8 and 20 characters long and contain at least one digit.
REQ-2: A user can transfer between 1 and 50,000 PKR per transaction. Transfers above 50,000 PKR require OTP verification.
REQ-3: The username must be unique and may contain only letters, numbers and underscores. The maximum length is 15 characters.
REQ-4: After 3 failed login attempts the account is locked for 30 minutes.
REQ-5: Users can upload a profile picture in JPG or PNG format, up to 2 MB in size."""

# HELPERS START
ID_RE = re.compile(r"^\s*(?:[-*•]\s*)?([A-Za-z]{1,8}[-_ ]?\d{1,4}(?:\.\d+)*)\s*[:.)\-–]\s*(.+)$")
NUM_LINE_RE = re.compile(r"^\s*(?:[-*•]\s*)?(\d{1,3})[.)]\s+(.+)$")
BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+)$")


def normalize_id(raw):
    return re.sub(r"[\s_]+", "-", raw.strip().upper())


def parse_requirements(text):
    """Turn pasted text into a list of {id, text}. One requirement per line;
    indented lines continue the previous requirement."""
    reqs, seen = [], set()
    for line in text.splitlines():
        if not line.strip():
            continue
        stripped = line.strip()
        m_id = ID_RE.match(line)
        m_num = NUM_LINE_RE.match(line)
        m_bul = BULLET_RE.match(line)

        if m_id:
            rid, body = normalize_id(m_id.group(1)), m_id.group(2).strip()
        elif m_num:
            rid, body = "REQ-" + m_num.group(1), m_num.group(2).strip()
        elif m_bul:
            rid, body = "", m_bul.group(1).strip()
        else:
            is_indented = line.startswith("  ") or line.startswith("\t")
            if is_indented and reqs:
                reqs[-1]["text"] += " " + stripped
                continue
            if stripped.endswith(":"):
                continue  # a heading such as "Functional requirements:"
            rid, body = "", stripped

        if not rid or rid in seen:
            n = len(reqs) + 1
            rid = "REQ-" + str(n)
            while rid in seen:
                n += 1
                rid = "REQ-" + str(n)
        seen.add(rid)
        reqs.append({"id": rid, "text": body})
    return reqs


NUM = r"(\d[\d,]*(?:\.\d+)?)"
BOUNDARY_PATTERNS = [
    ("range", r"\bbetween\s+" + NUM + r"\s*(?:and|to|-|–)\s*" + NUM),
    ("range", r"\bfrom\s+" + NUM + r"\s*to\s*" + NUM),
    ("range", r"\b(\d+)\s*[-–]\s*(\d+)\s*(?:characters|chars|digits|years|items|days|mb|kb|gb)\b"),
    ("min", r"\b(?:at\s+least|minimum(?:\s+of)?|no\s+(?:less|fewer)\s+than|not\s+(?:less|fewer)\s+than)\s*" + NUM),
    ("max", r"\b(?:at\s+most|maximum(?:\s+(?:of|length\s+is))?|no\s+more\s+than|not\s+more\s+than|not\s+exceed(?:ing)?|up\s+to)\s*" + NUM),
    ("gt", r"\b(?:above|over|more\s+than|greater\s+than|exceeds?|exceeding)\s*" + NUM),
    ("lt", r"\b(?:below|under|less\s+than|fewer\s+than)\s*" + NUM),
]


def to_decimal(s):
    try:
        return Decimal(s.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def show_decimal(d):
    text = format(d.normalize(), "f")
    return text[:-2] if text.endswith(".0") else text


def extract_boundaries(text):
    """Find numeric limits in a requirement and return the values worth testing.

    'between 8 and 20' -> ['7', '8', '9', '19', '20', '21']
    'above 50,000'     -> ['50000', '50001']
    """
    work = text
    values = set()
    for kind, pattern in BOUNDARY_PATTERNS:
        for m in re.finditer(pattern, work, flags=re.IGNORECASE):
            nums = [to_decimal(g) for g in m.groups()]
            if any(n is None for n in nums):
                continue
            step = Decimal("0.01") if any(n != n.to_integral_value() for n in nums) else Decimal(1)
            if kind == "range":
                low, high = nums[0], nums[1]
                values.update([low - step, low, low + step, high - step, high, high + step])
            elif kind in ("min", "max"):
                n = nums[0]
                values.update([n - step, n, n + step])
            elif kind == "gt":
                values.update([nums[0], nums[0] + step])
            elif kind == "lt":
                values.update([nums[0] - step, nums[0]])
        # Hide what was matched so a later pattern does not match the same words again.
        work = re.sub(pattern, lambda m: " " * len(m.group(0)), work, flags=re.IGNORECASE)
    return [show_decimal(v) for v in sorted(values)]


def value_in_text(value, text):
    """True if the number appears in the text (also as 50,000 style)."""
    variants = {value}
    try:
        d = Decimal(value)
        if d == d.to_integral_value() and abs(d) >= 1000:
            variants.add(format(int(d), ","))
    except InvalidOperation:
        pass
    for v in variants:
        if re.search(r"(?<![\d.])" + re.escape(v) + r"(?!\d)", text):
            return True
    return False


def missing_boundary_values(hints, tests):
    """Boundary values that no Boundary test mentions."""
    text = " ".join(
        " ".join(str(t.get(k, "")) for k in ("Title", "Test Data", "Steps", "Expected Result"))
        for t in tests
        if t.get("Type") == "Boundary"
    )
    return [h for h in hints if not value_in_text(h, text)]


def build_messages(batch, types, per_type):
    system_msg = (
        "You are a senior software tester. You turn requirements into clear, precise, "
        "executable test cases and you point out ambiguous or incomplete requirements. "
        "You never invent behaviour that the requirement does not state. Reply with valid JSON only."
    )
    lines = []
    for r in batch:
        line = r["id"] + ": " + r["text"]
        if r["hints"]:
            line += "\n   BOUNDARY VALUES TO TEST (computed from the numbers above): " + ", ".join(r["hints"])
        lines.append(line)

    user_msg = (
        "Write test cases for the REQUIREMENTS below.\n\n"
        "Test types to produce: " + ", ".join(types) + "\n"
        "Maximum test cases per requirement for each type: " + str(per_type) + "\n\n"
        "Type meanings:\n"
        "- Normal: valid input and expected, typical use.\n"
        "- Boundary: values at and just beside each limit (use the BOUNDARY VALUES given).\n"
        "- Invalid: wrong or harmful input, for example empty, wrong type, wrong format, "
        "negative, too long, special characters, or a rule that is broken.\n\n"
        "Return a JSON object with exactly this shape:\n"
        "{\n"
        '  "requirements": [\n'
        '    {"id": "REQ-1", "issues": ["ambiguity or missing detail"], "assumptions": ["assumption you had to make"]}\n'
        "  ],\n"
        '  "test_cases": [\n'
        "    {\n"
        '      "requirement_id": "REQ-1",\n'
        '      "type": "Normal" or "Boundary" or "Invalid",\n'
        '      "title": "short, specific title",\n'
        '      "preconditions": "state needed before the test, or empty",\n'
        '      "test_data": "the exact input values used",\n'
        '      "steps": ["step 1", "step 2"],\n'
        '      "expected_result": "the exact expected outcome",\n'
        '      "priority": "High" or "Medium" or "Low"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Every test case must use the requirement's id in requirement_id.\n"
        "- Use exact, concrete values in test_data, not descriptions like 'a valid password'.\n"
        "- The expected_result must follow from the requirement. If the requirement does not say "
        "what should happen (for example the exact error message), write 'Not specified: ' and what "
        "needs to be clarified, and add it to that requirement's issues.\n"
        "- Do not invent error message text, limits or rules that are not in the requirement.\n"
        "- For Boundary tests, use the given BOUNDARY VALUES. Values just outside a limit are "
        "expected to be rejected, and values at or inside it accepted, unless the requirement says otherwise.\n"
        "- Give each test a different purpose. No duplicates.\n"
        "- issues: ambiguities, missing details or untestable wording. Use an empty list if there are none.\n\n"
        "REQUIREMENTS:\n" + "\n".join(lines)
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def extract_json(text):
    """Parse JSON even if the model wrapped it in code fences or extra text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("The AI did not return valid JSON.")


def match_type(value):
    v = str(value).strip().lower()
    for t in TYPES:
        if v == t.lower() or v.startswith(t.lower()[:4]):
            return t
    return None


def match_priority(value):
    v = str(value).strip().lower()
    for p in PRIORITIES:
        if v == p.lower() or v.startswith(p.lower()[:3]):
            return p
    return "Medium"


def simple_id(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def as_text(value):
    if isinstance(value, list):
        return "\n".join(str(x).strip() for x in value if str(x).strip())
    if value is None:
        return ""
    return str(value).strip()


def numbered_steps(value):
    """Return steps as '1. ...' lines."""
    if isinstance(value, list):
        steps = [str(x).strip() for x in value if str(x).strip()]
    else:
        steps = [s.strip() for s in re.split(r"\n|;\s+(?=\S)", str(value or "")) if s.strip()]
    steps = [re.sub(r"^\d+[.)]\s*", "", s) for s in steps]
    return "\n".join(str(i) + ". " + s for i, s in enumerate(steps, start=1))


def validate_batch(data, batch, types, per_type):
    """Return (tests, review, dropped) from one AI reply. IDs are assigned later."""
    if not isinstance(data, dict):
        raise ValueError("The AI reply was not in the expected format.")
    ids = {simple_id(r["id"]): r["id"] for r in batch}
    tests, dropped, counts, seen = [], 0, {}, set()

    for t in data.get("test_cases") or []:
        if not isinstance(t, dict):
            dropped += 1
            continue
        rid = ids.get(simple_id(t.get("requirement_id", "")))
        if rid is None and len(batch) == 1:
            rid = batch[0]["id"]
        ttype = match_type(t.get("type", ""))
        title = as_text(t.get("title"))
        expected = as_text(t.get("expected_result"))
        if rid is None or ttype is None or ttype not in types or not title or not expected:
            dropped += 1
            continue
        key = (rid, ttype, re.sub(r"\W+", " ", title.lower()).strip())
        if key in seen:
            dropped += 1
            continue
        if counts.get((rid, ttype), 0) >= per_type:
            dropped += 1
            continue
        seen.add(key)
        counts[(rid, ttype)] = counts.get((rid, ttype), 0) + 1
        tests.append(
            {
                "Requirement ID": rid,
                "Type": ttype,
                "Priority": match_priority(t.get("priority")),
                "Title": title,
                "Preconditions": as_text(t.get("preconditions")),
                "Test Data": as_text(t.get("test_data")),
                "Steps": numbered_steps(t.get("steps")),
                "Expected Result": expected,
            }
        )

    review = {}
    for r in data.get("requirements") or []:
        if isinstance(r, dict):
            rid = ids.get(simple_id(r.get("id", "")))
            if rid:
                review[rid] = {
                    "issues": [str(x).strip() for x in (r.get("issues") or []) if str(x).strip()],
                    "assumptions": [str(x).strip() for x in (r.get("assumptions") or []) if str(x).strip()],
                }
    return tests, review, dropped


def assign_ids(tests, reqs):
    """Sort by requirement and type, then number the tests TC-001, TC-002, ..."""
    order = {r["id"]: i for i, r in enumerate(reqs)}
    tests = sorted(tests, key=lambda t: (order.get(t["Requirement ID"], 999), TYPES.index(t["Type"])))
    for i, t in enumerate(tests, start=1):
        t["Test ID"] = "TC-" + format(i, "03d")
    return [{col: t.get(col, "") for col in COLUMNS} for t in tests]


def coverage_rows(reqs, rows):
    out = []
    for r in reqs:
        mine = [x for x in rows if x.get("Requirement ID") == r["id"]]
        row = {"Requirement": r["id"]}
        for t in TYPES:
            row[t] = sum(1 for x in mine if x.get("Type") == t)
        row["Total"] = len(mine)
        out.append(row)
    return out


def build_csv(rows):
    """CSV bytes that open correctly in Excel (UTF-8 with BOM)."""
    df = pd.DataFrame(rows, columns=COLUMNS).fillna("")
    return df.to_csv(index=False).encode("utf-8-sig")


def friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if "401" in msg or "invalid api key" in low or "invalid_api_key" in low:
        return "Your Groq API key looks invalid. Please check it and try again."
    if "429" in msg or "rate limit" in low:
        return "Groq rate limit reached. Please wait a minute and try again, or use fewer requirements."
    if "decommissioned" in low or "does not exist" in low or "404" in msg:
        return (
            "That model is not available on your account. Choose another model in the "
            "sidebar. Details: " + msg[:300]
        )
    if "connection" in low or "timeout" in low or "timed out" in low:
        return "Could not reach the Groq API. Check your internet connection and retry."
    return "Something went wrong: " + msg
# HELPERS END


def get_api_key(sidebar_value):
    """Priority: sidebar input -> st.secrets -> environment variable."""
    if sidebar_value and sidebar_value.strip():
        return sidebar_value.strip()
    try:
        secret = st.secrets["GROQ_API_KEY"]
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_models(api_key):
    """Ask Groq which chat models this API key can use right now."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": "Bearer " + api_key},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "safeguard", "embed")
        ids = [
            m["id"]
            for m in r.json().get("data", [])
            if m.get("active", True) and not any(s in m["id"].lower() for s in skip)
        ]
        return sorted(ids)
    except Exception:
        return []


def call_groq(api_key, model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}

    resp = None
    for attempt in range(3):
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)
        # Some models do not support JSON mode: retry without it.
        if resp.status_code == 400 and "response_format" in resp.text and "response_format" in payload:
            payload.pop("response_format")
            continue
        # Rate limited: wait a little and try again.
        if resp.status_code == 429 and attempt < 2:
            try:
                wait = float(resp.headers.get("retry-after", 8))
            except ValueError:
                wait = 8.0
            time.sleep(min(wait, 20.0))
            continue
        break

    if resp.status_code != 200:
        raise RuntimeError("Groq API error " + str(resp.status_code) + ": " + resp.text[:300])
    choices = resp.json().get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def read_document(name, data):
    """Return the text of an uploaded PDF, DOCX, TXT, MD or CSV file."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF support needs the 'pypdf' package. Add it to requirements.txt.")
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
        if not text:
            raise ValueError("No readable text found in the PDF (it may be a scanned image).")
        return text
    if lower.endswith(".docx"):
        try:
            from docx import Document
        except ImportError:
            raise ValueError("Word support needs the 'python-docx' package. Add it to requirements.txt.")
        text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs).strip()
        if not text:
            raise ValueError("No readable text found in the Word file.")
        return text
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc).strip()
        except Exception:
            continue
    raise ValueError("Could not read the file.")


# --------------------------------------------------------------------------- #
# Session state and callbacks
# --------------------------------------------------------------------------- #
for key, value in {"req_text": "", "file_error": "", "analysis": None, "run": 0}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_example():
    st.session_state["req_text"] = SAMPLE_REQUIREMENTS
    st.session_state["file_error"] = ""


def load_file():
    f = st.session_state.get("req_file")
    if f is None:
        return
    try:
        st.session_state["req_text"] = read_document(f.name, f.getvalue())
        st.session_state["file_error"] = ""
    except Exception as exc:
        st.session_state["file_error"] = str(exc)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    key_input = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Leave empty to use st.secrets or the GROQ_API_KEY environment variable.",
    )
    api_key = get_api_key(key_input)
    if api_key:
        st.success("API key found ✅")
    else:
        st.warning("No API key found yet")

    live_models = fetch_models(api_key) if api_key else []
    model_options = live_models or MODELS
    default_index = 0
    for preferred in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        if preferred in model_options:
            default_index = model_options.index(preferred)
            break
    model = st.selectbox("Groq model", model_options, index=default_index)

    st.subheader("Test cases")
    chosen_types = st.multiselect("Types to generate", TYPES, default=TYPES)
    per_type = st.slider("Max tests per type, per requirement", 1, 5, 2)

    st.divider()
    st.button("📋 Load an example", on_click=load_example)
    st.caption("Fills in five sample requirements.")

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🧪 AI Requirements-to-Test-Cases Generator")
st.write(
    "Enter your software requirements. The app generates normal, boundary and invalid-input "
    "test cases with expected results, points out unclear requirements, and exports to CSV."
)

st.file_uploader(
    "Upload a requirements document (optional: PDF, Word or text)",
    type=["pdf", "docx", "txt", "md"],
    key="req_file",
    on_change=load_file,
)
if st.session_state["file_error"]:
    st.error(st.session_state["file_error"])

st.caption("One requirement per line, for example: REQ-1: The password must be between 8 and 20 characters.")
st.text_area(
    "Requirements",
    height=240,
    key="req_text",
    placeholder="REQ-1: The password must be between 8 and 20 characters long.\nREQ-2: ...",
)

reqs = parse_requirements(st.session_state["req_text"])
if reqs:
    with st.expander("Requirements as the app reads them: " + str(len(reqs)) + " found"):
        st.table(
            [
                {
                    "ID": r["id"],
                    "Requirement": r["text"],
                    "Boundary values found": ", ".join(extract_boundaries(r["text"])) or "-",
                }
                for r in reqs
            ]
        )

if st.button("✨ Generate test cases", type="primary"):
    text = st.session_state["req_text"].strip()
    if not api_key:
        st.error(
            "No Groq API key found. Enter it in the sidebar, or add GROQ_API_KEY "
            "to your Streamlit secrets."
        )
    elif not reqs:
        st.warning("Please enter at least one requirement.")
    elif not chosen_types:
        st.warning("Please choose at least one test type in the sidebar.")
    elif len(text) > MAX_INPUT_CHARS:
        st.warning("The requirements are too long. Please keep them under " + str(MAX_INPUT_CHARS) + " characters.")
    else:
        work_reqs = reqs[:MAX_REQUIREMENTS]
        if len(reqs) > MAX_REQUIREMENTS:
            st.warning("Only the first " + str(MAX_REQUIREMENTS) + " requirements were used.")
        for r in work_reqs:
            r["hints"] = extract_boundaries(r["text"]) if "Boundary" in chosen_types else []
        types = [t for t in TYPES if t in chosen_types]

        all_tests, review, failed, dropped_total = [], {}, [], 0
        batches = [work_reqs[i:i + BATCH_SIZE] for i in range(0, len(work_reqs), BATCH_SIZE)]
        progress = st.progress(0.0, text="Generating test cases...")
        try:
            for n, batch in enumerate(batches):
                done = False
                for attempt in range(2):
                    try:
                        raw = call_groq(api_key, model, build_messages(batch, types, per_type))
                        tests, rev, dropped = validate_batch(extract_json(raw), batch, types, per_type)
                        if tests:
                            all_tests.extend(tests)
                            review.update(rev)
                            dropped_total += dropped
                            done = True
                            break
                    except RuntimeError:
                        raise
                    except Exception:
                        continue
                if not done:
                    failed.extend(r["id"] for r in batch)
                progress.progress((n + 1) / len(batches), text="Generating test cases... " + str(n + 1) + "/" + str(len(batches)))
            progress.empty()

            if not all_tests:
                st.error("The AI did not produce usable test cases. Please try again.")
            else:
                st.session_state["run"] += 1
                st.session_state["analysis"] = {
                    "reqs": work_reqs,
                    "rows": assign_ids(all_tests, work_reqs),
                    "review": review,
                    "types": types,
                    "failed": failed,
                    "dropped": dropped_total,
                    "run": st.session_state["run"],
                }
        except Exception as exc:
            progress.empty()
            st.error(friendly_error(exc))

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
analysis = st.session_state["analysis"]
if analysis is not None:
    reqs_done = analysis["reqs"]
    run = str(analysis["run"])

    st.divider()
    st.header("Test cases")
    st.warning(
        "These are AI-generated drafts. Check each expected result against your real "
        "specification before using the tests."
    )
    if analysis["failed"]:
        st.error("No tests could be generated for: " + ", ".join(analysis["failed"]) + ". Try generating again.")
    if analysis["dropped"]:
        st.caption(str(analysis["dropped"]) + " AI test case(s) were skipped (duplicates, over the limit, or incomplete).")

    tab_tests, tab_cov, tab_rev = st.tabs(["🧾 Test cases (editable)", "📊 Coverage", "🔍 Requirement review"])

    with tab_tests:
        st.caption("Click a cell to edit it. Use the + row at the bottom to add a test. Select a row and press Delete to remove it.")
        df = pd.DataFrame(analysis["rows"], columns=COLUMNS)
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            key="editor_" + run,
            column_config={
                "Test ID": st.column_config.TextColumn("Test ID", width="small"),
                "Requirement ID": st.column_config.TextColumn("Req", width="small"),
                "Type": st.column_config.SelectboxColumn("Type", options=TYPES, width="small"),
                "Priority": st.column_config.SelectboxColumn("Priority", options=PRIORITIES, width="small"),
                "Title": st.column_config.TextColumn("Title", width="medium"),
                "Preconditions": st.column_config.TextColumn("Preconditions", width="medium"),
                "Test Data": st.column_config.TextColumn("Test Data", width="medium"),
                "Steps": st.column_config.TextColumn("Steps", width="large"),
                "Expected Result": st.column_config.TextColumn("Expected Result", width="large"),
            },
        )
        rows = edited.fillna("").to_dict("records")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test cases", len(rows))
        for col, t in zip((m2, m3, m4), TYPES):
            col.metric(t, sum(1 for r in rows if r.get("Type") == t))

        st.download_button(
            "⬇️ Download test cases (CSV)",
            data=build_csv(rows),
            file_name="test_cases_" + datetime.now().strftime("%Y%m%d_%H%M") + ".csv",
            mime="text/csv",
        )

    with tab_cov:
        st.dataframe(pd.DataFrame(coverage_rows(reqs_done, rows)), hide_index=True)

        problems = 0
        for r in reqs_done:
            mine = [x for x in rows if x.get("Requirement ID") == r["id"]]
            for t in analysis["types"]:
                if not any(x.get("Type") == t for x in mine):
                    st.warning(r["id"] + " has no " + t + " test.")
                    problems += 1
            hints = extract_boundaries(r["text"]) if "Boundary" in analysis["types"] else []
            if hints:
                missing = missing_boundary_values(hints, mine)
                if missing:
                    st.warning(
                        r["id"] + ": boundary values not used in any Boundary test: " + ", ".join(missing)
                    )
                    problems += 1
        if problems == 0:
            st.success("Every requirement has every test type, and all boundary values are covered.")
        st.caption(
            "Boundary values come from the numbers in each requirement, for example "
            "'between 8 and 20' gives 7, 8, 9, 19, 20, 21."
        )

    with tab_rev:
        st.caption("Unclear or incomplete requirements found while writing the tests. Fix these with your team.")
        any_issue = False
        for r in reqs_done:
            info = analysis["review"].get(r["id"], {"issues": [], "assumptions": []})
            with st.expander(r["id"] + ": " + r["text"][:90], expanded=bool(info["issues"])):
                if info["issues"]:
                    any_issue = True
                    st.markdown("**Issues to clarify**")
                    for x in info["issues"]:
                        st.markdown("- " + x)
                else:
                    st.markdown("No issues reported.")
                if info["assumptions"]:
                    st.markdown("**Assumptions the AI made**")
                    for x in info["assumptions"]:
                        st.markdown("- " + x)
        if not any_issue:
            st.info("The AI did not report any unclear requirements.")
