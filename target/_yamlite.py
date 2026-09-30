"""Minimal YAML-subset loader with no third-party dependency.

Why this exists
---------------
``build_target()`` runs inside Blender, and Blender's bundled Python ships numpy but
*not* PyYAML (true of Blender 4.2, which BlenderProc 2.8 pins, and of 5.x).
``kal_geometry.load_config`` prefers PyYAML when it is importable and falls back to this
parser when it is not. Carried over from the earlier ``kal_target`` package.

Scope
-----
This handles exactly the YAML features the project's configs use:

  * block mappings nested by indentation
  * block sequences (``- item``), including ``- key: value`` inline mappings
  * flow sequences (``[a, b]``) and flow mappings (``{}``, ``{a: 1}``) on one line
  * simple folded (``>``) and literal (``|``) block scalars, with ``-``/``+`` chomping
  * single- and double-quoted scalars, ``#`` comments, ``---`` document markers
  * int / float / bool / null / str scalars, typed as PyYAML's ``safe_load`` types them

Anything outside that subset raises :class:`YamliteError` rather than being guessed at.
A silent mis-parse would put a wrong wingspan into every render in the dataset, so this
parser is deliberately strict: it would rather refuse a config than misread one.
``tests/test_geometry.py`` asserts it agrees with PyYAML exactly on ``configs/target.yaml``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

__all__ = ["loads", "load", "YamliteError"]


class YamliteError(ValueError):
    """Raised when input uses YAML outside this parser's supported subset."""

    def __init__(self, message: str, line_no: int = -1, line: str = "") -> None:
        if line_no >= 0:
            message = "{} (line {}: {!r})".format(message, line_no, line.rstrip())
        super().__init__(message)
        self.line_no = line_no


# ---------------------------------------------------------------------------
# Scalar typing -- mirrors PyYAML safe_load's resolver for the types we use.
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"^[-+]?[0-9][0-9_]*$")
_HEX_RE = re.compile(r"^[-+]?0x[0-9a-fA-F_]+$")
_OCT_RE = re.compile(r"^[-+]?0o[0-7_]+$")
_FLOAT_RE = re.compile(
    r"^[-+]?(?:[0-9][0-9_]*)?\.[0-9_]*(?:[eE][-+]?[0-9]+)?$"
    r"|"
    r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"
)
_INF_RE = re.compile(r"^[-+]?\.(?:inf|Inf|INF)$")
_NAN_RE = re.compile(r"^\.(?:nan|NaN|NAN)$")

_TRUE = frozenset(["true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"])
_FALSE = frozenset(["false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"])
_NULL = frozenset(["", "null", "Null", "NULL", "~"])


def _scalar(text: str, line_no: int, raw: str) -> Any:
    """Type an unquoted scalar token the way PyYAML's safe_load would."""
    s = text.strip()
    if s in _NULL:
        return None
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    if _INT_RE.match(s):
        return int(s.replace("_", ""))
    if _HEX_RE.match(s):
        return int(s.replace("_", ""), 16)
    if _OCT_RE.match(s):
        return int(s.replace("_", ""), 8)
    if _FLOAT_RE.match(s):
        return float(s.replace("_", ""))
    if _INF_RE.match(s):
        return float("-inf") if s.startswith("-") else float("inf")
    if _NAN_RE.match(s):
        return float("nan")
    if s[:1] in ("&", "*", "!"):
        raise YamliteError(
            "anchors, aliases and tags are outside the fallback parser's subset; "
            "install PyYAML into this interpreter or simplify the config",
            line_no,
            raw,
        )
    return s


def _strip_comment(text: str) -> str:
    """Drop a trailing ``#`` comment, respecting quotes and flow collections."""
    out: List[str] = []
    quote = ""
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            out.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                if quote == "'" and i + 1 < n and text[i + 1] == "'":
                    out.append("'")
                    i += 2
                    continue
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch in "[{":
            depth += 1
            out.append(ch)
        elif ch in "]}":
            depth -= 1
            out.append(ch)
        elif ch == "#" and depth == 0 and (not out or out[-1] in " \t"):
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _unquote(text: str, line_no: int, raw: str) -> Tuple[str, bool]:
    """Return ``(value, was_quoted)``. Quoted scalars are never type-coerced."""
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] == "'":
        return s[1:-1].replace("''", "'"), True
    if len(s) >= 2 and s[0] == s[-1] == '"':
        body = s[1:-1]
        try:
            return body.encode("latin-1", "backslashreplace").decode("unicode_escape"), True
        except (UnicodeDecodeError, UnicodeEncodeError):
            return body, True
    return s, False


# ---------------------------------------------------------------------------
# Flow collections
# ---------------------------------------------------------------------------


def _split_flow(body: str, line_no: int, raw: str) -> List[str]:
    """Split a flow-collection body on top-level commas."""
    parts: List[str] = []
    buf: List[str] = []
    quote = ""
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(body[i + 1])
                i += 2
                continue
            if ch == quote:
                if quote == "'" and i + 1 < n and body[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "[{":
            depth += 1
            buf.append(ch)
        elif ch in "]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    if quote:
        raise YamliteError("unterminated quote in flow collection", line_no, raw)
    if depth:
        raise YamliteError("unbalanced brackets in flow collection", line_no, raw)
    tail = "".join(buf)
    if parts or tail.strip():
        parts.append(tail)
    return parts


def _parse_flow(text: str, line_no: int, raw: str) -> Any:
    s = text.strip()
    if s.startswith("["):
        if not s.endswith("]"):
            raise YamliteError(
                "multi-line flow sequences are outside the fallback parser's subset",
                line_no,
                raw,
            )
        return [_parse_value(p, line_no, raw) for p in _split_flow(s[1:-1], line_no, raw)]
    if not s.endswith("}"):
        raise YamliteError(
            "multi-line flow mappings are outside the fallback parser's subset", line_no, raw
        )
    out: Dict[Any, Any] = {}
    for part in _split_flow(s[1:-1], line_no, raw):
        if not part.strip():
            continue
        if ":" not in part:
            raise YamliteError("flow-mapping entry without ':'", line_no, raw)
        k, _, v = part.partition(":")
        out[_parse_key(k, line_no, raw)] = _parse_value(v, line_no, raw)
    return out


def _parse_value(text: str, line_no: int, raw: str) -> Any:
    s = text.strip()
    if s[:1] in ("[", "{"):
        return _parse_flow(s, line_no, raw)
    value, quoted = _unquote(s, line_no, raw)
    return value if quoted else _scalar(value, line_no, raw)


def _parse_key(text: str, line_no: int, raw: str) -> Any:
    value, quoted = _unquote(text, line_no, raw)
    return value if quoted else _scalar(value, line_no, raw)


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(
    r"^(?P<key>"
    r'"[^"]*"'
    r"|'(?:[^']|'')*'"
    r"|[^:#\[\]{},]+?"
    r")\s*:(?:\s+(?P<rest>.*))?$"
)


class _Line:
    """One significant source line, with its comment stripped and indent measured."""

    __slots__ = ("no", "raw", "indent", "text")

    def __init__(self, no: int, raw: str, indent: int, text: str) -> None:
        self.no = no
        self.raw = raw
        self.indent = indent
        self.text = text

    @classmethod
    def from_raw(cls, no: int, raw: str) -> "_Line":
        stripped = _strip_comment(raw).rstrip()
        indent = len(stripped) - len(stripped.lstrip(" "))
        return cls(no, raw, indent, stripped.strip())


def _is_block_scalar_header(rest: str) -> bool:
    return bool(rest) and rest[0] in "|>" and all(c in "+-0123456789 " for c in rest[1:])


class _Parser:
    def __init__(self, text: str) -> None:
        for i, raw in enumerate(text.splitlines(), 1):
            if "\t" in _strip_comment(raw):
                raise YamliteError("tabs are not valid YAML indentation", i, raw)

        lines: List[_Line] = []
        for i, raw in enumerate(text.splitlines(), 1):
            ln = _Line.from_raw(i, raw)
            if ln.text in ("", "---", "..."):
                continue
            lines.append(ln)

        # Normalise "- key: value" into a bare "-" marker plus an indented mapping
        # line, so sequences and mappings can be parsed by one uniform rule.
        self.lines: List[_Line] = []
        for ln in lines:
            if ln.text == "-" or not ln.text.startswith("- "):
                self.lines.append(ln)
                continue
            body = ln.text[1:]
            offset = len(body) - len(body.lstrip(" "))
            rest = body.strip()
            if rest.startswith("- "):
                raise YamliteError(
                    "nested inline sequences are outside the fallback parser's subset",
                    ln.no,
                    ln.raw,
                )
            looks_like_mapping = rest[:1] not in ("[", "{", '"', "'") and _KEY_RE.match(rest)
            if looks_like_mapping:
                item_indent = ln.indent + 1 + offset
                self.lines.append(_Line(ln.no, ln.raw, ln.indent, "-"))
                self.lines.append(_Line(ln.no, ln.raw, item_indent, rest))
            else:
                self.lines.append(ln)
        self.pos = 0

    def peek(self):
        return self.lines[self.pos] if self.pos < len(self.lines) else None

    # -- block scalars ------------------------------------------------------

    def _collect_block_lines(self, parent_indent: int) -> List[str]:
        raw_lines: List[str] = []
        while self.pos < len(self.lines):
            ln = self.lines[self.pos]
            if ln.indent <= parent_indent and ln.text:
                break
            raw_lines.append(ln.raw.rstrip())
            self.pos += 1
        return raw_lines

    def _block_scalar(
        self, header: str, parent_indent: int, raw_lines: List[str], line_no: int, raw: str
    ) -> str:
        style = header[0]
        chomp = ""
        explicit_indent = 0
        for ch in header[1:]:
            if ch in "+-":
                chomp = ch
            elif ch.isdigit():
                explicit_indent = int(ch)
            elif ch != " ":
                raise YamliteError("unsupported block-scalar header " + repr(header), line_no, raw)

        if not raw_lines:
            return "" if chomp == "-" else ""

        indents = [len(l) - len(l.lstrip(" ")) for l in raw_lines if l.strip()]
        base = parent_indent + explicit_indent if explicit_indent else (min(indents) if indents else 0)
        body = [(l[base:] if len(l) >= base else l.lstrip(" ")) for l in raw_lines]
        while body and not body[-1].strip():
            body.pop()
        if not body:
            return ""

        if style == "|":
            out = "\n".join(body) + "\n"
        else:
            # Folded. Keep the supported subset narrow: refuse blank lines and
            # more-indented lines rather than risk diverging from PyYAML on the
            # paragraph/literal folding rules.
            for l in body:
                if not l.strip():
                    raise YamliteError(
                        "blank lines inside a folded ('>') block scalar are outside the "
                        "fallback parser's subset; use '|' or install PyYAML",
                        line_no,
                        raw,
                    )
                if l.startswith(" "):
                    raise YamliteError(
                        "more-indented lines inside a folded ('>') block scalar are outside "
                        "the fallback parser's subset; use '|' or install PyYAML",
                        line_no,
                        raw,
                    )
            out = " ".join(l.strip() for l in body) + "\n"

        if chomp == "-":
            out = out.rstrip("\n")
        return out

    # -- collections --------------------------------------------------------

    def parse_block(self, indent: int) -> Any:
        ln = self.peek()
        if ln is None:
            return None
        if ln.text == "-" or ln.text.startswith("- "):
            return self.parse_sequence(indent)
        return self.parse_mapping(indent)

    def parse_sequence(self, indent: int) -> List[Any]:
        items: List[Any] = []
        while self.pos < len(self.lines):
            ln = self.lines[self.pos]
            if ln.indent < indent:
                break
            if ln.indent > indent:
                raise YamliteError("unexpected indentation inside a sequence", ln.no, ln.raw)
            if not (ln.text == "-" or ln.text.startswith("- ")):
                break
            rest = ln.text[1:].strip()
            self.pos += 1
            if rest == "":
                nxt = self.peek()
                if nxt is not None and nxt.indent > ln.indent:
                    items.append(self.parse_block(nxt.indent))
                else:
                    items.append(None)
                continue
            items.append(_parse_value(rest, ln.no, ln.raw))
        return items

    def parse_mapping(self, indent: int) -> Dict[Any, Any]:
        out: Dict[Any, Any] = {}
        while self.pos < len(self.lines):
            ln = self.lines[self.pos]
            if ln.indent < indent:
                break
            if ln.indent > indent:
                raise YamliteError("unexpected indentation inside a mapping", ln.no, ln.raw)
            if ln.text == "-" or ln.text.startswith("- "):
                break
            m = _KEY_RE.match(ln.text)
            if not m:
                raise YamliteError("expected 'key: value'", ln.no, ln.raw)
            key = _parse_key(m.group("key"), ln.no, ln.raw)
            if key in out:
                raise YamliteError("duplicate key " + repr(key), ln.no, ln.raw)
            rest = (m.group("rest") or "").strip()
            self.pos += 1

            if _is_block_scalar_header(rest):
                raw_lines = self._collect_block_lines(ln.indent)
                out[key] = self._block_scalar(rest, ln.indent, raw_lines, ln.no, ln.raw)
                continue
            if rest == "":
                nxt = self.peek()
                if nxt is None or nxt.indent < ln.indent:
                    out[key] = None
                elif nxt.indent > ln.indent:
                    out[key] = self.parse_block(nxt.indent)
                elif nxt.text == "-" or nxt.text.startswith("- "):
                    # A sequence may sit at the same indent as its parent key.
                    out[key] = self.parse_sequence(ln.indent)
                else:
                    out[key] = None
                continue
            out[key] = _parse_value(rest, ln.no, ln.raw)
        return out

    def parse(self) -> Any:
        if not self.lines:
            return None
        value = self.parse_block(self.lines[0].indent)
        if self.pos != len(self.lines):
            ln = self.lines[self.pos]
            raise YamliteError("trailing content the parser could not attach", ln.no, ln.raw)
        return value


def loads(text: str) -> Any:
    """Parse a YAML-subset string. Raises :class:`YamliteError` outside the subset."""
    return _Parser(text).parse()


def load(path) -> Any:
    """Parse a YAML-subset file at *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return loads(fh.read())
