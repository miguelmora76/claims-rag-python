import re

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"\b\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_MRN = re.compile(r"\b(mrn|member id|patient id)[:#\s]*[A-Z0-9-]{5,}\b", re.IGNORECASE)
_DOB = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def redact(s: str) -> str:
    """Best-effort pattern redaction applied to questions before they leave the service.

    Demonstrates the pattern only. It is NOT a HIPAA de-identification method and misses names, addresses and free text.
    """
    for pat, token in (
        (_SSN, "[SSN]"),
        (_PHONE, "[PHONE]"),
        (_EMAIL, "[EMAIL]"),
        (_MRN, "[ID]"),
        (_DOB, "[DATE]"),
    ):
        s = pat.sub(token, s)
    return s
