"""Bot language setting shared by modules that send localized replies."""

import os


def get_language() -> str:
    """Return the configured reply language; `ua` is accepted as an old alias for `uk`."""
    language = os.getenv("LANGUAGE", "uk").lower()
    return "uk" if language == "ua" else language
