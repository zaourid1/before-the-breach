
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "granite4:micro"


def generate_briefing(cve):
    """
    Generate an analyst briefing using IBM Granite.

    cve: dictionary containing snapshot evidence.
    Returns: generated text.
    """

    evidence = {
        "cve_id": cve["cve_id"],
        "priority_rank": int(cve["rank"]),
        "model_ranking_score": float(cve["ml_score"]),
        "epss": float(cve["epss"]),
        "exploitdb": int(cve["exploitdb"]),
        "metasploit": int(cve["metasploit"]),
        "nuclei": int(cve["nuclei"]),
        "poc_github": int(cve["poc_github"]),
        "age_days": int(cve["age_days"]),
    }

    
    instructions = """
    You are a cybersecurity analyst assisting with
    vulnerability triage.

    The application already displays the verified data.
    Do NOT reproduce the raw evidence as a list.

    Your ONLY job is to provide a short interpretation.

    Write exactly two sections:

    ### Why this deserves investigation
    Explain why the combination of the supplied
    indicators warrants analyst review.

    ### Recommended analyst actions
    Give 3 practical, defensive next steps.

    Rules:
    - Never infer critical severity from a priority rank.
    - Never claim confirmed active exploitation.
    - Never claim that a CVE is on CISA KEV.
    - The ML score is NOT a calibrated probability.
    - Do not interpret the ML score as a percentage.
    - Nuclei=1 means a scanning template was recorded.
    - GitHub PoC=1 means public demonstration code exists.
    - A zero flag means NOT OBSERVED, not nonexistent.
    - Do not invent vendor, product, CVSS, exploit method,
    attack history, or affected organization.
    - EPSS is an external 30-day exploitation estimate.
    - Use cautious, precise security language.
    - Do not repeat numbers unnecessarily.
    - Keep the entire response under 120 words.
    - Vulnerability age is context, not proof that
    exploitation likelihood has increased.
    - Never infer low threat activity from missing
    Metasploit, ExploitDB, or other flags.
    - Do not assume a patch or mitigation exists.
    Recommend checking vendor advisories first.
    - Do not suggest a particular detection rule exists
    without evidence.
    - Treat the ranking as prioritization for review,
    not as confirmation of exploitability.
    """



    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": (
                "Analyze this vulnerability evidence:\n"
                + json.dumps(evidence, indent=2)
            ),
        },
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 350,
        },
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)

        return result["message"]["content"]

    except (URLError, HTTPError, TimeoutError) as error:
        raise RuntimeError(
            "Could not connect to IBM Granite through Ollama. "
            "Check that Ollama is running and the model "
            "has been downloaded."
        ) from error
