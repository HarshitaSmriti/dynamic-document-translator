import sys, re
sys.stdout.reconfigure(encoding="utf-8")

def clean_indic_text(text: str, original_text: str = "") -> str:
    """
    Cleans up common translation messes in Indic/English outputs.
    """
    if not text:
        return ""

    t = text.strip()

    # 1. Fix spaces before punctuation
    t = re.sub(r'\s+([,;:!?।॥\.\%\)\]\}])', r'\1', t)

    # 2. Fix spaces after opening brackets and quotes
    t = re.sub(r'([\(\[\{“"\'])(\s+)', r'\1', t)

    # 3. Fix numbers and decimals (e.g. 1 . 0 -> 1.0, 99 . 9 -> 99.9)
    t = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', t)

    # 4. Fix hyphenated identifiers and versions (e.g. REQ - 001 -> REQ-001, v 2.4 -> v2.4, FR - 001 -> FR-001)
    t = re.sub(r'([A-Za-z0-9]+)\s*-\s*([A-Za-z0-9]+)', r'\1-\2', t)
    t = re.sub(r'\b([vV])\s+(\d+)', r'\1\2', t)

    # 5. Fix percentages: 95 % -> 95%
    t = re.sub(r'(\d+)\s*%', r'\1%', t)

    # 6. Preserve trailing colon if original had colon
    if original_text and original_text.rstrip().endswith(":") and not t.endswith(":"):
        if t.endswith("।") or t.endswith("."):
            t = t[:-1] + ":"
        else:
            t = t + ":"

    # 7. Collapse multiple spaces
    t = re.sub(r'\s{2,}', ' ', t)

    return t

sample_messes = [
    ("REQ - 001 : Automatic Detection .", "REQ-001: Automatic Detection:"),
    ("Version 2 . 4 ( Final Draft )", "Version 2.4 (Final Draft)"),
    ("Over 95 % accuracy [ 1 ] ।", "Over 95% accuracy [1]:"),
    ("ISO - 27001 compliance !", "ISO-27001 compliance!"),
]

for messy, orig in sample_messes:
    cleaned = clean_indic_text(messy, orig)
    print(f"Original: '{orig}'\nMessy:    '{messy}'\nCleaned:  '{cleaned}'\n")
