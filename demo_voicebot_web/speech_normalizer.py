import re


MOJIBAKE_FIXES = {
    "Äƒ": "ă",
    "Ã¢": "â",
    "Ã®": "î",
    "È™": "ș",
    "Èš": "Ș",
    "ÅŸ": "ș",
    "È›": "ț",
    "Å£": "ț",
    "Ä‚": "Ă",
    "Ã‚": "Â",
    "ÃŽ": "Î",
    "Ã®": "î",
    "È˜": "Ș",
    "Åž": "Ș",
    "Èš": "Ț",
    "Å¢": "Ț",
}


ABBREVIATIONS = {
    "SMS": "es em es",
    "PIN": "pin",
    "IBAN": "i ban",
    "ATM": "a te me",
    "STT": "es te te",
    "TTS": "te te es",
    "API": "a pe i",
    "URL": "u er el",
    "RON": "lei",
    "EUR": "euro",
    "USD": "dolari",
    "ID": "i de",
    "CNP": "ce ne pe",
    "OTP": "o te pe",
}


def normalize_for_tts(text: str) -> str:
    fixed = fix_mojibake(text)
    fixed = expand_email(fixed)
    fixed = expand_abbreviations(fixed)
    fixed = spell_sensitive_numbers(fixed)
    fixed = re.sub(r"\s+", " ", fixed).strip()
    return fixed


def fix_mojibake(text: str) -> str:
    fixed = text
    for source, target in MOJIBAKE_FIXES.items():
        fixed = fixed.replace(source, target)
    return fixed


def expand_abbreviations(text: str) -> str:
    fixed = text
    for source, target in ABBREVIATIONS.items():
        fixed = re.sub(rf"\b{re.escape(source)}\b", target, fixed, flags=re.IGNORECASE)
    return fixed


def expand_email(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        value = value.replace("@", " arond ")
        value = value.replace(".", " punct ")
        value = value.replace("_", " underscore ")
        value = value.replace("-", " minus ")
        return value

    return re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", replace, text)


def spell_sensitive_numbers(text: str) -> str:
    fixed = re.sub(
        r"\b(?:terminat|terminată|terminația|terminatia|cardul|card)\s+(?:\S+\s+)?(\d(?:[\s.-]?\d){3})\b",
        lambda match: match.group(0).replace(match.group(1), spell_digits(match.group(1))),
        text,
        flags=re.IGNORECASE,
    )
    fixed = re.sub(
        r"\b(07\d{8})\b",
        lambda match: spell_digits(match.group(1)),
        fixed,
    )
    fixed = re.sub(
        r"\b(\d{10,16})\b",
        lambda match: spell_digits(match.group(1)),
        fixed,
    )
    return fixed


def spell_digits(value: str) -> str:
    return " ".join(re.sub(r"\D", "", value))
