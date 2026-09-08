"""Lab 4: Arabic normalisation profiles and clitic segmentation."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    """Apply the requested Arabic model-normalisation profile."""

    normalized = text

    # Remove tatweel
    normalized = normalized.replace("ـ", "")

    # Remove Arabic diacritics when requested
    if profile.dediacritize:
        normalized = _ARABIC_DIACRITICS.sub("", normalized)

    if profile.name == "bayan_ar_v1":
        normalized = (
            normalized
            .replace("أ", "ا")
            .replace("إ", "ا")
            .replace("آ", "ا")
            .replace("ؤ", "و")
            .replace("ئ", "ي")
            .replace("ى", "ي")
            .replace("ة", "ه")
        )

    return normalized


_mle = None
_d3_tokenizer = None


def segment(text: str) -> list[str]:
    """Segment Arabic clitics using CAMeL Tools d3tok."""

    global _mle, _d3_tokenizer

    # Lazy imports so Arabic normalisation tests do not
    # require CAMeL models to load.
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer
    from camel_tools.tokenizers.word import simple_word_tokenize

    if _mle is None:
        _mle = MLEDisambiguator.pretrained(
            "calima-msa-r13"
        )

    if _d3_tokenizer is None:
        _d3_tokenizer = MorphologicalTokenizer(
            disambiguator=_mle,
            scheme="d3tok",
            split=True,
            diac=False,
        )

    words = simple_word_tokenize(text)

    return _d3_tokenizer.tokenize(words)