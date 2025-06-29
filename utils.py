import re


def fix_title(title):
    """
    Fixes movie titles by moving articles (The, A, An) from the end to the beginning.
    For example: "Matrix, The" -> "The Matrix"
    """
    match = re.match(r"^(.*),\s(The|A|An)(\s?\(.*\))?$", title)
    if match:
        prefix = match.group(2)
        main = match.group(1)
        suffix = match.group(3) if match.group(3) else ""
        return f"{prefix} {main}{suffix}"
    return title
