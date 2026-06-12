CRITICAL = [
    "breach",
    "security",
    "fraud",
    "outage",
    "service unavailable",
    "500",
    "internal server error",
    "api error",
    "data loss",
    "payment failed",
    "unable to login",
    "login failed",
    "crash"
]

HIGH = [
    "not syncing",
    "cannot access",
    "failed",
    "error",
    "dashboard not loading",
    "refund",
    "payment issue"
]

NEGATIONS = [
    "unable",
    "cannot",
    "not working",
    "failed"
]

def get_rule_score(text):

    text = str(text).lower()

    evidence = []

    for kw in CRITICAL:
        if kw in text:
            evidence.append(kw)
            return 3, evidence

    for kw in HIGH:
        if kw in text:
            evidence.append(kw)
            return 2, evidence

    return 0, evidence