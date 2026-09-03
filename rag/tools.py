"""Portfolio tools the chat model can call.

Each tool has a declaration (schema) and a Python handler that reads the structured data in
data/portfolio.json. The model picks the tool from the visitor's question, the server runs the
handler and feeds the result back. This is the tool-calling layer on top of RAG retrieval.

Ported from the source repository. Five of the six handlers are byte identical read-only lookups.
Two things changed, both in the sixth:

1. `CONTACT_ENDPOINT` unset means dry run. The original defaulted to the live Vercel contact route,
   so running the 41 case eval against it would have sent Firza real email, once per send case. An
   env var guards every caller at once, where a stub in the eval script guards only one. See D29.
2. The tool log records the tool name and which argument keys arrived, never their values. The
   original printed the visitor's name and email address into the log.

`send_message_to_firza` is the only tool with a side effect, so it carries guards the read-only
ones do not need. See the comments on that handler.
"""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

_PORTFOLIO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "portfolio.json")


@lru_cache(maxsize=1)
def _data():
    with open(_PORTFOLIO_PATH, encoding="utf-8") as f:
        return json.load(f)


def _contains(haystack, needle: str) -> bool:
    return needle.lower() in str(haystack).lower()


# ---- handlers -------------------------------------------------------------

def search_projects(query="", year="", category="", stack=""):
    """Filter projects by any combination of free-text query, year, category, stack.
    Returns lightweight summaries (no full case study body)."""
    out = []
    for p in _data()["projects"]:
        if year and str(p["year"]) != str(year):
            continue
        if category and not any(_contains(c, category) for c in p["categories"]):
            continue
        if stack and not any(_contains(s, stack) for s in p["stack"]):
            continue
        if query and not (
            _contains(p["title"], query)
            or _contains(p["subtitle"], query)
            or any(_contains(s, query) for s in p["stack"])
        ):
            continue
        out.append(
            {
                "slug": p["slug"],
                "title": p["title"],
                "subtitle": p["subtitle"],
                "client": p["client"],
                "year": p["year"],
                "categories": p["categories"],
                "stack": p["stack"],
            }
        )
    return {"count": len(out), "projects": out}


def get_project_detail(slug=""):
    """Return the full case study for one project by slug."""
    for p in _data()["projects"]:
        if p["slug"] == slug:
            return {
                "title": p["title"],
                "client": p["client"],
                "year": p["year"],
                "detail": p["detail"],
            }
    return {"error": f"No project with slug '{slug}'."}


def search_experience(company="", current="", internship="", stack=""):
    """Filter full-time and internship roles by company, current flag, internship flag, or stack tech."""
    out = []
    for r in _data()["experience"]:
        if company and not _contains(r["company"], company):
            continue
        if stack and not any(_contains(s, stack) for s in r.get("stack", [])):
            continue
        if current != "" and bool(r["current"]) != (str(current).lower() == "true"):
            continue
        if internship != "" and bool(r["internship"]) != (str(internship).lower() == "true"):
            continue
        out.append(r)
    return {"count": len(out), "experience": out}


def get_career_timeline():
    """Return all roles ordered earliest to latest for chronological questions."""
    roles = list(_data()["experience"])
    # experience.ts is newest-first, so reverse for chronological order.
    ordered = list(reversed(roles))
    return {
        "timeline": [
            {
                "order": i + 1,
                "title": r["title"],
                "company": r["company"],
                "period": r["period"],
                "internship": r["internship"],
                "current": r["current"],
            }
            for i, r in enumerate(ordered)
        ]
    }


def get_skills(domain=""):
    """Return skill groups, optionally filtered to groups matching a domain keyword."""
    all_groups = _data()["about"]["skills"]
    if not domain:
        return {"skills": all_groups}
    groups = [g for g in all_groups if _contains(g["group"], domain) or any(_contains(i, domain) for i in g["items"])]
    if not groups:
        # a miss on the keyword shouldn't dead-end the model; hand it the full
        # list so it can spot the relevant skill itself
        return {"note": f"No skill group matched '{domain}'. Full skill list:", "skills": all_groups}
    return {"skills": groups}


# ---- write tool ------------------------------------------------------------

# One email per request, enforced in code since a prompt rule is no guarantee. Reset by reset_send_guard().
_sent_this_request = set()

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# Posts to the website's contact route, one code path for outbound mail. Unset means dry run.
_CONTACT_ENDPOINT = os.environ.get("CONTACT_ENDPOINT", "").strip()


def reset_send_guard():
    """Clear the per-request send guard. Called once per chat request."""
    _sent_this_request.clear()


def _post_contact(payload: bytes):
    """POST the message to the contact route. Raises on any HTTP or transport error.

    Split out as its own function so a test harness can replace just the network hop.
    """
    origin = "{0.scheme}://{0.netloc}".format(urllib.parse.urlsplit(_CONTACT_ENDPOINT))
    req = urllib.request.Request(
        _CONTACT_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json", "Origin": origin},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        json.loads(resp.read().decode() or "{}")


def send_message_to_firza(name="", email="", message="", topic="Chat assistant"):
    """Forward a visitor's message to Firza by email. Has a real side effect."""
    name = str(name or "").strip()
    email = str(email or "").strip()
    message = str(message or "").strip()

    # Validate before the guard: a rejected call is not a send, so the model
    # should be free to retry with corrected arguments.
    missing = [f for f, v in (("name", name), ("email", email), ("message", message)) if not v]
    if missing:
        return {"sent": False, "error": f"Missing required field(s): {', '.join(missing)}. Ask the visitor for them."}
    if not _EMAIL_RE.match(email):
        return {"sent": False, "error": f"'{email}' is not a valid email address. Ask the visitor to confirm it."}

    if _sent_this_request:
        return {
            "sent": False,
            "error": "A message was already sent for this visitor. Tell them it is on its way; do not send another.",
        }

    # Dry run keeps the guard, so the second-send case behaves the same as in production.
    if not _CONTACT_ENDPOINT:
        _sent_this_request.add(email.lower())
        return {"sent": True, "dry_run": True, "note": f"Message from {name} delivered to Firza. He replies fastest on LinkedIn."}

    payload = json.dumps({
        "name": name[:100],
        "email": email[:254],
        "topic": f"Chat: {str(topic or 'Chat assistant').strip()[:80]}",
        "message": message[:5000],
    }).encode()

    try:
        _post_contact(payload)
    except urllib.error.HTTPError as e:
        # 429 from the contact route means the shared abuse limit tripped.
        detail = "too many messages have been sent recently" if e.code == 429 else f"HTTP {e.code}"
        return {"sent": False, "error": f"Could not send ({detail}). Point the visitor to the Contact page instead."}
    except Exception as e:
        print(json.dumps({"evt": "send_message_error", "error": str(e)[:200]}))
        return {"sent": False, "error": "Could not send the message. Point the visitor to the Contact page instead."}

    _sent_this_request.add(email.lower())
    return {"sent": True, "note": f"Message from {name} delivered to Firza. He replies fastest on LinkedIn."}


HANDLERS = {
    "search_projects": search_projects,
    "get_project_detail": get_project_detail,
    "search_experience": search_experience,
    "get_career_timeline": get_career_timeline,
    "get_skills": get_skills,
    "send_message_to_firza": send_message_to_firza,
}


# ---- function declarations -------------------------------------------------

DECLARATIONS = [
    {
        "name": "search_projects",
        "description": "Search Firza's independent and client projects. Filter by free-text query, year, category (e.g. 'Data Engineer', 'AI Engineer', 'Dashboard'), or stack/technology. Use this for any question about his projects or project work.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text keyword to match against title, subtitle, or stack."},
                "year": {"type": "string", "description": "Filter by year, e.g. '2025'."},
                "category": {"type": "string", "description": "Filter by category label."},
                "stack": {"type": "string", "description": "Filter by a technology in the stack."},
            },
        },
    },
    {
        "name": "get_project_detail",
        "description": "Get the full case study (context, problem, approach, results) for one project by its slug. Call search_projects first to find the slug.",
        "parameters": {
            "type": "object",
            "properties": {"slug": {"type": "string", "description": "The project slug."}},
            "required": ["slug"],
        },
    },
    {
        "name": "search_experience",
        "description": "Search Firza's full-time roles and internships. Filter by company name, whether it's his current role, whether it's an internship, or a technology used in the role.",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name to match."},
                "current": {"type": "string", "description": "'true' for only the current role."},
                "internship": {"type": "string", "description": "'true' for only internships, 'false' for only full-time."},
                "stack": {"type": "string", "description": "Technology used in the role, e.g. 'Kafka', 'dbt'."},
            },
        },
    },
    {
        "name": "get_career_timeline",
        "description": "Get all of Firza's roles in chronological order (earliest to latest). Use for questions about his first job, career start, or full career history.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_skills",
        "description": "Get Firza's technical skills, optionally filtered to a domain keyword (e.g. 'cloud', 'ML', 'visualization').",
        "parameters": {
            "type": "object",
            "properties": {"domain": {"type": "string", "description": "Optional domain keyword to filter skill groups."}},
        },
    },
    {
        "name": "send_message_to_firza",
        "description": (
            "Send the visitor's message to Firza by email. This actually delivers a message, so only call it "
            "when ALL of these are true: (1) the visitor has hiring, collaboration, or project inquiry intent, "
            "(2) they have given you their name, their email address, and what they want to say, and "
            "(3) they have explicitly agreed to send it after you asked for confirmation. "
            "Never call this to answer a question, never call it with details you inferred or made up, and never "
            "call it more than once in a conversation. If any detail is missing, ask the visitor for it instead of calling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The visitor's name, exactly as they gave it."},
                "email": {"type": "string", "description": "The visitor's email address, exactly as they gave it."},
                "message": {"type": "string", "description": "What the visitor wants to tell Firza, in their own words."},
                "topic": {"type": "string", "description": "Short subject line, e.g. 'Hiring inquiry' or 'Project collaboration'."},
            },
            "required": ["name", "email", "message"],
        },
    },
]


# Ollama's /api/chat wants each declaration wrapped. Same fields, one envelope.
OLLAMA_TOOLS = [{"type": "function", "function": d} for d in DECLARATIONS]


def run_tool(name: str, args: dict):
    handler = HANDLERS.get(name)
    if not handler:
        return {"error": f"Unknown tool '{name}'."}
    try:
        result = handler(**(args or {}))
        # Keys only. send_message_to_firza args carry a visitor's name and email address.
        print(json.dumps({"evt": "tool", "name": name, "args": sorted((args or {}).keys())}))
        return result
    except TypeError as e:
        return {"error": f"Bad arguments for '{name}': {e}"}


if __name__ == "__main__":
    # Self-check: data loads, each tool runs, filters actually filter.
    assert _data()["projects"], "no projects loaded"
    assert search_projects()["count"] == len(_data()["projects"]), "empty filter should return all"
    assert search_projects(year="1999")["count"] == 0, "impossible year should return none"
    tl = get_career_timeline()["timeline"]
    assert tl[0]["order"] == 1 and len(tl) == len(_data()["experience"]), "timeline malformed"
    assert "error" in get_project_detail(slug="nope"), "unknown slug should error"
    assert search_experience(stack="kafka")["count"] >= 1, "stack filter should find Kafka role"
    assert search_experience(stack="cobol")["count"] == 0, "stack filter should exclude unknown tech"
    assert get_skills(domain="data engineering")["skills"], "skill miss should fall back to full list"
    assert get_skills(domain="cloud")["skills"] != get_skills()["skills"], "matched domain should filter"

    # send_message_to_firza: only the paths that reject before any network call.
    # These must never reach the network, so a passing self check sends no email.
    assert not send_message_to_firza()["sent"], "empty args must not send"
    assert "name" in send_message_to_firza(email="a@b.co", message="hi")["error"], "should name the missing field"
    assert not send_message_to_firza(name="A", email="not-an-email", message="hi")["sent"], "bad email must not send"
    assert "valid email" in send_message_to_firza(name="A", email="a@b", message="hi")["error"], "should flag bad email"

    # the one-send guard blocks a second call even with perfectly valid args
    _sent_this_request.add("someone@example.com")
    blocked = send_message_to_firza(name="B", email="b@c.co", message="second attempt")
    assert not blocked["sent"] and "already sent" in blocked["error"], "guard must block a second send"
    reset_send_guard()
    assert not _sent_this_request, "reset_send_guard should clear the guard"

    # Dry run must report sent, arm the guard, and never open a socket.
    assert not _CONTACT_ENDPOINT, "run this self check with CONTACT_ENDPOINT unset"

    def _explode(_):
        raise AssertionError("dry run reached the network")

    _real_post, globals()["_post_contact"] = _post_contact, _explode
    dry = send_message_to_firza(name="A", email="a@b.co", message="hi")
    assert dry["sent"] and dry["dry_run"], dry
    assert _sent_this_request, "dry run must arm the one-send guard"
    globals()["_post_contact"] = _real_post
    reset_send_guard()

    assert len(OLLAMA_TOOLS) == len(DECLARATIONS) == len(HANDLERS), "tool lists out of sync"
    assert {t["function"]["name"] for t in OLLAMA_TOOLS} == set(HANDLERS), "declared and handled names differ"

    print(f"OK: {len(HANDLERS)} tools, {len(_data()['projects'])} projects, {len(tl)} roles, send path dry.")
