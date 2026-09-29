import re


SECTION_PATTERNS = [ #we beed to chnag this section patterns
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "approach",
    "model",
    "architecture",
    "encoder",
    "decoder",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "references",
]


def detect_section(text: str):

    lines = text.split("\n")

    for line in lines[:10]:

        clean = line.strip().lower()

        for section in SECTION_PATTERNS:

            if clean == section:
                return section.title()

            if re.match(r"^\d+\.?\s+" + re.escape(section) + r"$", clean):
                return section.title()

    return "Unknown"
