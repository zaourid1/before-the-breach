
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
You are an IBM Granite-powered cybersecurity
analyst assistant.

Your task is to explain a vulnerability's
prioritization using ONLY the supplied evidence.

Use these exact headings:

1. Observed Evidence
2. Why This CVE Deserves Review
3. Recommended Next Steps
4. Limitations and Uncertainties

CRITICAL RULES:

- Distinguish observed facts from model inferences.
- A high ranking does not establish critical severity.
- The ML ranking score is NOT a calibrated probability.
- Do not interpret it as a percentage chance of attack.
- EPSS estimates exploitation probability over 30 days.
- A GitHub PoC indicates public demonstration code,
  NOT confirmed active exploitation.
- A Nuclei flag indicates a scanning template exists,
  NOT that an attack occurred.
- Missing exploit indicators mean "not observed
  in this dataset", NOT that no exploit exists.
- Do not invent vendors, affected products, CVSS
  scores, attack methods, or confirmed incidents.
- Do not claim that the CVE is on the KEV list.
- Recommend verification, asset exposure assessment,
  and appropriate patch prioritization.

Keep the response under 200 words.
Use specific supplied numbers where relevant.
If information is unavailable, explicitly say so.
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
