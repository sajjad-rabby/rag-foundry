
"""Document loading and text extraction.

Supported formats: PDF, DOCX, HTML and plain text.

The PDF extractor is intentionally stdlib-only and supports the common case
of FlateDecode content streams with literal/hex strings and basic text
operators. It is a minimal content-stream reader rather than a full PDF
library. Known limitations are documented deliberately:
- Unescaped balanced inner parentheses inside strings are not handled.
- No encrypted PDFs, object streams (ObjStm), or inline images.
- CID/Type3 fonts and custom ToUnicode maps are not decoded.
"""

from __future__ import annotations

import html.parser
import os
import re
import xml.etree.ElementTree as ET
import zlib
import zipfile
from typing import Callable, Dict, Iterator, List, Tuple

# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

_STREAM_RE = re.compile(rb"stream\r?\n(.*?)endstream", re.DOTALL)
_TOKEN_RE = re.compile(
    rb"\((?:\\.|[^\\()])*\)"      # literal string
    b"|<[0-9A-Fa-f\\s]*>"         # hex string
    b"|([\\[])|(\\])"             # array delimiters
    b"|(Tj|TJ|T\\*|Td|TD|Tm|BT|ET)"  # text operators
    b"|\\S+"                      # everything else (names, numbers, Tf ...)
)
_ESCAPE_RE = re.compile(rb"\\(\r\n|\r|\n|[0-7]{1,3}|[nrtbf()\\/])")

_ESCAPE_MAP = {
    b"n": b"\n",
    b"r": b"\r",
    b"t": b"\t",
    b"b": b"\b",
    b"f": b"\f",
    b"(": b"(",
    b")": b")",
    b"\\": b"\\",
    b"/": b"/",
}

_TEXT_OPS = frozenset([b"Tj", b"TJ", b"T*", b"Td", b"TD", b"Tm", b"BT", b"ET"])


def _escape_sub(match: "re.Match[bytes]") -> bytes:
    body = match.group(1)
    if body in (b"\r\n", b"\r", b"\n"):
        return b""  # backslash-newline continuation inside a string
    if body.isdigit():
        return bytes([int(body, 8) & 0xFF])
    return _ESCAPE_MAP.get(body, body)


def _decode_pdf_string(raw: bytes) -> str:
    """Decode a PDF string body (bytes between the delimiters)."""
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace")
    text = _ESCAPE_RE.sub(_escape_sub, raw)
    return text.decode("latin-1", errors="replace")


def _extract_text_from_content(content: bytes) -> str:
    """Recover text from a PDF content stream.

TJ-array pieces are joined so a single word split across operators

    Strings accumulate in ``pending``; any text operator (Tj/TJ/T*/Td/TD/Tm/
    BT/ET) flushes the pending strings as one line. The ``in_array`` flag
    tracks bracket nesting so TJ-array pieces join into a single line.
    """
    lines: List[str] = []
    line_parts: List[str] = []
    pending: List[str] = []
    in_array = False

    def flush_line() -> None:
        if pending:
            line_parts.extend(pending)
            pending.clear()
        if line_parts:
            lines.append("".join(line_parts).strip())
            line_parts.clear()

    for match in _TOKEN_RE.finditer(content):
        token = match.group(0)
        if token.startswith(b"("):
            pending.append(_decode_pdf_string(token[1:-1]))
        elif token.startswith(b"<"):
            hexed = re.sub(rb"[^0-9A-Fa-f]", b"", token[1:-1])
            if len(hexed) % 2:
                hexed += b"0"
            try:
                pending.append(
                    bytes.fromhex(hexed.decode("ascii")).decode(
                        "utf-16-be", errors="replace"
                    )
                )
            except ValueError:
                continue
        elif token == b"[":
            in_array = True
        elif token == b"]":
            in_array = False
        elif token in _TEXT_OPS:
            flush_line()
        # other tokens (numbers, names, Tf, cm, ...) are ignored
    flush_line()
    return "\n".join(line for line in lines if line)


def extract_pdf(path: str) -> str:
    """Extract text from a PDF by decompressing FlateDecode content streams."""
    with open(path, "rb") as fh:
        data = fh.read()
    if b"/Encrypt" in data[:4096]:
        raise ValueError(f"{path}: encrypted PDFs are not supported")
    parts: List[str] = []
    for match in _STREAM_RE.finditer(data):
        blob = match.group(1)
        try:
            d = zlib.decompressobj()
            text = d.decompress(blob) + d.flush()
        except zlib.error:
            continue
        extracted = _extract_text_from_content(text)
        if extracted:
            parts.append(extracted)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_docx(path: str) -> str:
    """Extract paragraph text from a .docx by reading word/document.xml."""
    with zipfile.ZipFile(path) as zf:
        xml_data = zf.read("word/document.xml")
    root = ET.fromstring(xml_data)
    paragraphs: List[str] = []
    for para in root.iter(_W_NS + "p"):
        parts: List[str] = []
        for node in para.iter():
            if node.tag == _W_NS + "t" and node.text:
                parts.append(node.text)
            elif node.tag == _W_NS + "tab":
                parts.append("\t")
            elif node.tag in (_W_NS + "br", _W_NS + "cr"):
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_BLOCK_TAGS = frozenset(
    [
        "p", "div", "br", "li", "tr", "hr", "pre", "blockquote", "section",
        "article", "header", "footer", "main", "nav", "aside", "ul", "ol",
        "table", "thead", "tbody", "td", "th", "h1", "h2", "h3", "h4", "h5",
        "h6", "title", "caption", "dt", "dd", "figure", "figcaption",
    ]
)


class _HTMLTextParser(html.parser.HTMLParser):
    """Collect visible text, adding newlines around block tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        """Record that a start tag was seen, tracking skip depth.

        Args:
            tag: Lower-cased tag name.
            attrs: Attribute mapping supplied by the parser.
        """
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Record that an end tag was seen, tracking skip depth.

        Args:
            tag: Lower-cased tag name.
        """
        if tag in ("script", "style"):
            if self._skip:
                self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Accumulate a text fragment unless inside a skipped element.

        Args:
            data: Raw text fragment from the parser.
        """
        if not self._skip and data.strip():
            self.parts.append(data)


def extract_html(path: str) -> str:
    """Extract visible text from an HTML file."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    parser = _HTMLTextParser()
    parser.feed(raw)
    text = "".join(parser.parts)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Plain text + dispatch
# ---------------------------------------------------------------------------


def extract_text_file(path: str) -> str:
    """Read a UTF-8 text file (md/txt)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


_EXTRACTORS: Dict[str, Callable[[str], str]] = {
    ".pdf": extract_pdf,
    ".docx": extract_docx,
    ".html": extract_html,
    ".htm": extract_html,
    ".md": extract_text_file,
    ".txt": extract_text_file,
}


def extract_text(path: str) -> str:
    """Dispatch to the right extractor based on the file extension."""
    ext = os.path.splitext(path)[1].lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"unsupported file type: {ext!r} ({path})")
    return extractor(path)


def iter_files(paths: Iterator[str] or List[str]) -> Iterator[str]:
    """Yield files from a list of paths, walking directories recursively."""
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for name in sorted(files):
                    yield os.path.join(root, name)
        else:
            yield p


def load_documents(paths: List[str]) -> Iterator[Tuple[str, str]]:
    """Yield ``(source, text)`` for every supported file in ``paths``.

Raises:
    UnsupportedFileError: For an explicitly named file this

    Unsupported extensions inside directories are skipped silently; an
    explicitly named unsupported file raises via :func:`extract_text`.
    """
    for file_path in iter_files(iter(paths)):
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in _EXTRACTORS:
            continue
        yield file_path, extract_text(file_path)
