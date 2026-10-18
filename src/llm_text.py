"""Pure command parsing, localized notices and plain-text citation rendering."""

import re
from urllib.parse import urlsplit

MESSAGES = {
    "busy": (
        "Зараз забагато запитів або очікування затягнулося. Спробуйте трохи пізніше.",
        "Too many requests or the queue wait expired. Please try again shortly.",
    ),
    "empty": ("Після web: напишіть, що потрібно знайти.", "Please add a search question after web:."),
    "daily_limit": (
        "Денний ліміт запитів до бота вичерпано. Спробуйте завтра.",
        "Your daily chat request limit has been reached. Please try again tomorrow.",
    ),
    "minute_limit": (
        "Забагато запитів за хвилину. Зачекайте хвилину та спробуйте ще раз.",
        "Too many chat requests in a minute. Please wait a minute and try again.",
    ),
    "not_configured": (
        "Відповіді ШІ недоступні: перевірте налаштування провайдера та API-ключа бота.",
        "AI replies are unavailable: check the bot's provider and API key configuration.",
    ),
    "general_info": ("Ось загальна інформація:", "Here is some general information:"),
    "fallback": (
        "Не вдалося перевірити це в інтернеті. Відповідь нижче — без вебперевірки.",
        "I couldn't verify this online. The answer below has not been web-verified.",
    ),
    "quota": (
        "Вебпошук зараз недоступний через денний ліміт; ця відповідь — без вебперевірки.",
        "Web search is currently unavailable due to the daily limit; this answer is not web-verified.",
    ),
    "gemini": (
        "Вебпошук підтримується лише через Grok. Ця відповідь — без вебперевірки.",
        "Web search is supported only with Grok. This answer is not web-verified.",
    ),
    "failure": (
        "Вибачте, не вдалося підготувати відповідь. Спробуйте пізніше.",
        "Sorry, I couldn't prepare an answer. Please try again later.",
    ),
    "sources": ("Джерела:", "Sources:"),
}


def localized(key, language):
    """All fixed LLM messages have Ukrainian and English variants."""
    return MESSAGES[key][0 if language == "uk" else 1]


def plain_gemini_text(text):
    """Remove common paired Markdown delimiters, preserving literal operators/URLs."""
    text = re.sub(r"(?m)^ {0,3}```[\w+-]*[ \t]*(?:\n|$)", "", text)
    text = re.sub(r"(?m)^ {0,3}#{1,6}[ \t]+", "", text)
    parts = re.split(r"(https?://[^\s<>]+)", text)
    for index in range(0, len(parts), 2):
        part = re.sub(r"(?<![\w*])\*\*(?=\S)([^*\n]*?\S|\S)\*\*(?![\w*])", r"\1", parts[index])
        part = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]*?\S|\S)\*(?![\w*])", r"\1", part)
        parts[index] = re.sub(r"`([^`\n]+)`", r"\1", part)
    return "".join(parts).strip()


def parse_command(text):
    """Return leading command and original-case content; never match quoted commands."""
    match = re.match(r"^\s*(?:ботяра(?![^\W\d_])|bot\s*:)[^\w\s]*\s*(.*)$", text, re.I | re.S)
    if not match:
        return "chat", text.strip()
    content = match.group(1).strip()
    command = re.match(r"^(image|web)\s*:\s*(.*)$", content, re.I | re.S)
    if command:
        return command.group(1).lower(), command.group(2).strip()
    return "chat", content


def safe_url(url):
    """Accept complete public-style HTTP(S) links, without control characters/userinfo."""
    if any(char.isspace() or ord(char) < 32 for char in url):
        return False
    try:
        parsed = urlsplit(url)
        return parsed.scheme in {"https", "http"} and bool(parsed.hostname) and parsed.username is None
    except ValueError:
        return False


def render_citations(text, searched=False, language="uk", known_sources=None):
    """Handle the provider's numbered links, including balanced parentheses in URLs.

    Returns presentation text and context text without source markers/footers.
    Unrecognized syntax is preserved. Numbering is derived from complete URLs,
    not provider labels that can repeat between content blocks.
    """
    body, context, urls = [], [], []
    position = 0
    marker = re.compile(r"\[\[\d+\]\]\(")
    for match in marker.finditer(text):
        if match.start() < position:
            continue
        start = match.end()
        end, depth = start, 1
        while end < len(text) and end - start <= 8192:
            char = text[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            if char.isspace() or ord(char) < 32:
                break
            end += 1
        url = text[start:end]
        if depth != 0 or not safe_url(url):
            continue
        body.append(text[position : match.start()])
        context.append(text[position : match.start()])
        if searched and (known_sources is None or url in known_sources):
            if url not in urls:
                urls.append(url)
            body.append(f"[{urls.index(url) + 1}]")
        else:
            body.append(url)
        position = end + 1
    body.append(text[position:])
    context.append(text[position:])
    rendered = "".join(body).strip()
    if urls:
        rendered += (
            "\n\n"
            + localized("sources", language)
            + "\n"
            + "\n".join(f"[{index}] {url}" for index, url in enumerate(urls, 1))
        )
    return rendered, "".join(context).strip()


def split_reply(text, limit=3800):
    """Split plain text using UTF-16 units, preferring paragraph/word boundaries."""
    if limit < 2:
        raise ValueError("Reply limit must allow an astral code point")
    chunks = []
    while text:
        units, end = 0, 0
        for char in text:
            width = 2 if ord(char) > 0xFFFF else 1
            if units + width > limit:
                break
            units += width
            end += 1
        if end < len(text):
            boundary = max(text.rfind("\n", 0, end), text.rfind(" ", 0, end))
            if boundary > end // 2:
                end = boundary + 1
        chunks.append(text[:end])
        text = text[end:]
    return chunks
