"""The Kestrel Cycles world: a fake company with a filesystem, a SQL database, a mock web, a
mail outbox, a calendar, a calculator and a currency converter.

Every entity is fictional, so a model cannot answer from memory: a correct answer requires
using the tools. Every tool is deterministic, so a trajectory can be replayed exactly. Each
trajectory gets a fresh `World` (the FS is copied, the SQLite DB is rebuilt in memory).

`faults` injects realistic breakages for the regression-detector experiment, e.g. a
permissions change that denies reads under one directory or an upstream site going down.
"""

from __future__ import annotations

import ast
import math
import operator
import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .base import Tool, ToolEnv, ToolFailure, params

WORLD_DIR = Path(__file__).resolve().parents[1] / "world"

FX_TO_INR = {"INR": 1.0, "USD": 84.0, "EUR": 91.5, "GBP": 106.0, "JPY": 0.56}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]+$", re.I)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@lru_cache(maxsize=1)
def _fs_fixture() -> dict[str, str]:
    root = WORLD_DIR / "fs"
    return {"/" + p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


@lru_cache(maxsize=1)
def _sql_fixture() -> str:
    return (WORLD_DIR / "kestrel.sql").read_text()


@lru_cache(maxsize=1)
def _web_fixture() -> dict[str, dict]:
    pages = {}
    for p in sorted((WORLD_DIR / "web").glob("*.md")):
        head, body = p.read_text().split("\n\n", 1)
        meta = dict(line.split(": ", 1) for line in head.splitlines())
        pages[meta["url"]] = {"url": meta["url"], "title": meta["title"], "body": body.strip()}
    return pages


@dataclass
class Faults:
    deny_paths: tuple[str, ...] = ()   # read_file/list_dir/search_files under these prefixes -> permission denied
    down_hosts: tuple[str, ...] = ()   # fetch_page for these hosts -> HTTP 503
    broken_tools: tuple[str, ...] = ()  # these tools fail every call (a dependency outage)

    @classmethod
    def parse(cls, specs: list[str] | None) -> "Faults":
        deny, down, broken = [], [], []
        for s in specs or []:
            kind, _, arg = s.partition(":")
            if kind == "deny":
                deny.append(arg)
            elif kind == "down":
                down.append(arg)
            elif kind == "break":
                broken.append(arg)
            else:
                raise ValueError(f"unknown fault {s!r}")
        return cls(tuple(deny), tuple(down), tuple(broken))


@dataclass
class World:
    fs: dict[str, str] = field(default_factory=lambda: dict(_fs_fixture()))
    db: sqlite3.Connection = field(default=None)
    web: dict[str, dict] = field(default_factory=_web_fixture)
    outbox: list[dict] = field(default_factory=list)
    meetings: list[dict] = field(default_factory=list)
    faults: Faults = field(default_factory=Faults)

    def __post_init__(self):
        if self.db is None:
            self.db = sqlite3.connect(":memory:", check_same_thread=False)
            self.db.executescript(_sql_fixture())
            self.db.execute("PRAGMA query_only = ON")

    def snapshot(self) -> dict:
        """Final world state that state-checkers look at (files written, emails, meetings)."""
        base = _fs_fixture()
        changed = {p: c for p, c in self.fs.items() if base.get(p) != c}
        deleted = [p for p in base if p not in self.fs]
        return {"files_changed": changed, "files_deleted": deleted, "outbox": self.outbox, "meetings": self.meetings}

    # ------------------------------------------------------------------ filesystem
    def _check_read(self, path: str) -> None:
        for prefix in self.faults.deny_paths:
            if path.startswith(prefix):
                raise ToolFailure(f"permission denied: {path}")

    @staticmethod
    def _norm(path: str) -> str:
        path = "/" + path.strip().lstrip("/")
        return re.sub(r"/+", "/", path)

    def list_dir(self, path: str = "/") -> str:
        path = self._norm(path).rstrip("/") + "/"
        self._check_read(path)
        entries = set()
        for p in self.fs:
            if p.startswith(path):
                rest = p[len(path):]
                entries.add(rest.split("/", 1)[0] + ("/" if "/" in rest else ""))
        if not entries:
            raise ToolFailure(f"no such directory: {path}")
        return "\n".join(sorted(entries))

    def read_file(self, path: str) -> str:
        path = self._norm(path)
        self._check_read(path)
        if path not in self.fs:
            raise ToolFailure(f"no such file: {path}")
        return self.fs[path]

    def write_file(self, path: str, content: str) -> str:
        path = self._norm(path)
        if path.endswith("/"):
            raise ToolFailure("path must be a file, not a directory")
        existed = path in self.fs
        self.fs[path] = content
        return f"{'overwrote' if existed else 'wrote'} {path} ({len(content)} chars)"

    def search_files(self, text: str, path: str = "/") -> str:
        path = self._norm(path)
        needle = text.lower()
        hits = []
        for p, content in self.fs.items():
            if not p.startswith(path) or any(p.startswith(d) for d in self.faults.deny_paths):
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if needle in line.lower():
                    hits.append(f"{p}:{i}: {line.strip()}")
        if not hits:
            return "no matches"
        more = f"\n... {len(hits) - 40} more matches" if len(hits) > 40 else ""
        return "\n".join(hits[:40]) + more

    # ------------------------------------------------------------------ database
    def get_db_schema(self) -> str:
        rows = self.db.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return "\n".join(r[0] + ";" for r in rows)

    def sql_query(self, query: str) -> str:
        q = query.strip().rstrip(";")
        if not re.match(r"(?is)^\s*(select|with)\b", q):
            raise ToolFailure("only read-only SELECT queries are allowed")
        try:
            cur = self.db.execute(q)
        except sqlite3.Error as e:
            raise ToolFailure(f"SQL error: {e}") from None
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(51)
        if not rows:
            return " | ".join(cols) + "\n(0 rows)"
        lines = [" | ".join(cols)] + [" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:50]]
        if len(rows) > 50:
            lines.append("... (more than 50 rows; refine the query)")
        return "\n".join(lines)

    # ------------------------------------------------------------------ web
    def web_search(self, query: str) -> str:
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1]
        if not terms:
            raise ToolFailure("empty query")
        scored = []
        for page in self.web.values():
            title, body = page["title"].lower(), page["body"].lower()
            score = sum(3 * title.count(t) + body.count(t) for t in terms)
            if score:
                scored.append((-score, page["url"], page))
        scored.sort()
        if not scored:
            return "no results"
        out = []
        for _, _, page in scored[:3]:
            snippet = page["body"][:160].replace("\n", " ")
            out.append(f"{page['title']}\n{page['url']}\n{snippet}...")
        return "\n\n".join(out)

    def fetch_page(self, url: str) -> str:
        url = url.strip()
        page = self.web.get(url) or self.web.get(url.rstrip("/"))
        host = re.sub(r"^https?://", "", url).split("/", 1)[0]
        if host in self.faults.down_hosts:
            raise ToolFailure(f"HTTP 503 Service Unavailable: {url}")
        if page is None:
            raise ToolFailure(f"HTTP 404 Not Found: {url}")
        return f"{page['title']}\n\n{page['body']}"

    # ------------------------------------------------------------------ actions
    def send_email(self, to: str, subject: str, body: str) -> str:
        if not EMAIL_RE.match(to.strip()):
            raise ToolFailure(f"invalid email address: {to!r}")
        self.outbox.append({"to": to.strip().lower(), "subject": subject, "body": body})
        return f"email sent to {to.strip()}"

    def schedule_meeting(self, title: str, date: str, attendees: list) -> str:
        if not DATE_RE.match(date):
            raise ToolFailure("date must be YYYY-MM-DD")
        bad = [a for a in attendees if not isinstance(a, str) or not EMAIL_RE.match(a)]
        if bad or not attendees:
            raise ToolFailure(f"attendees must be a non-empty list of email addresses; bad: {bad}")
        self.meetings.append({"title": title, "date": date, "attendees": sorted(a.lower() for a in attendees)})
        return f"meeting '{title}' scheduled on {date} with {len(attendees)} attendees"

    # ------------------------------------------------------------------ utilities
    @staticmethod
    def calculate(expression: str) -> str:
        return _fmt(_safe_eval(expression))

    @staticmethod
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        f, t = from_currency.upper().strip(), to_currency.upper().strip()
        if f not in FX_TO_INR or t not in FX_TO_INR:
            raise ToolFailure(f"unsupported currency; supported: {', '.join(FX_TO_INR)}")
        value = amount * FX_TO_INR[f] / FX_TO_INR[t]
        return f"{_fmt(amount)} {f} = {_fmt(round(value, 2))} {t} (rate 1 {f} = {_fmt(round(FX_TO_INR[f] / FX_TO_INR[t], 6))} {t})"


# ---------------------------------------------------------------------- calculator
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FN = {"round": round, "min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "sum": lambda *a: sum(a)}


def _safe_eval(expr: str) -> float:
    """Evaluate arithmetic via the AST (no eval). Commas are argument separators first; if that
    fails, they are retried as digit grouping ('84,999 * 2')."""
    try:
        return _eval_once(expr)
    except ToolFailure:
        grouped = re.sub(r"(?<=\d),(?=\d)", "", expr)
        if grouped == expr:
            raise
        return _eval_once(grouped)


def _eval_once(expr: str) -> float:
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ToolFailure(f"could not parse expression: {expr!r}") from None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            left, right = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 100:
                raise ToolFailure("exponent too large")
            return _BIN[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FN and not n.keywords:
            return _FN[n.func.id](*[ev(a) for a in n.args])
        raise ToolFailure("only numbers, + - * / ** % //, parentheses and round/min/max/abs/sqrt/sum are allowed")

    try:
        return ev(tree)
    except ZeroDivisionError:
        raise ToolFailure("division by zero") from None


def _fmt(x) -> str:
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return repr(round(x, 6))
    return str(x)


# ---------------------------------------------------------------------- tool registry
def make_env(faults: list[str] | None = None) -> ToolEnv:
    w = World(faults=Faults.parse(faults))
    s, n, a = "string", "number", "array"
    tools = [
        Tool("list_dir", "List the files and folders inside a directory of the company file system. Folders end with '/'.",
             params({}, {"path": (s, "Directory path, e.g. '/' or '/docs'. Defaults to '/'.")}), w.list_dir),
        Tool("read_file", "Read the full text of a file in the company file system.",
             params({"path": (s, "Absolute file path, e.g. '/docs/handbook.md'.")}), w.read_file),
        Tool("write_file", "Create or overwrite a text file in the company file system.",
             params({"path": (s, "Absolute file path."), "content": (s, "Full file content.")}), w.write_file, side_effect=True),
        Tool("search_files", "Case-insensitive text search across files; returns matching lines as path:line: text.",
             params({"text": (s, "Text to look for.")}, {"path": (s, "Only search under this folder. Defaults to '/'.")}), w.search_files),
        Tool("get_db_schema", "Return the CREATE TABLE statements of the company SQL database (SQLite).",
             params({}), w.get_db_schema),
        Tool("sql_query", "Run a read-only SQL query (SQLite dialect) against the company database and return the rows.",
             params({"query": (s, "A single SELECT statement.")}), w.sql_query),
        Tool("web_search", "Search the web. Returns the top 3 results with title, URL and a snippet.",
             params({"query": (s, "Search query.")}), w.web_search),
        Tool("fetch_page", "Fetch the full text of a web page by URL.",
             params({"url": (s, "The page URL, as returned by web_search.")}), w.fetch_page),
        Tool("calculate", "Evaluate an arithmetic expression, e.g. '(84999 * 1.06) - 2000'. Supports + - * / ** % and round, min, max, abs, sqrt.",
             params({"expression": (s, "The expression to evaluate.")}), w.calculate),
        Tool("convert_currency", "Convert an amount between currencies (INR, USD, EUR, GBP, JPY) at today's rate.",
             params({"amount": (n, "Amount to convert."), "from_currency": (s, "ISO code, e.g. 'USD'."),
                     "to_currency": (s, "ISO code, e.g. 'INR'.")}), w.convert_currency),
        Tool("send_email", "Send an email on behalf of the user.",
             params({"to": (s, "Recipient email address."), "subject": (s, "Subject line."), "body": (s, "Email body.")}),
             w.send_email, side_effect=True),
        Tool("schedule_meeting", "Create a calendar meeting and invite attendees.",
             {"type": "object", "additionalProperties": False, "required": ["title", "date", "attendees"],
              "properties": {"title": {"type": s, "description": "Meeting title."},
                             "date": {"type": s, "description": "Date as YYYY-MM-DD."},
                             "attendees": {"type": a, "items": {"type": s}, "description": "Attendee email addresses."}}},
             w.schedule_meeting, side_effect=True),
    ]
    for t in tools:
        if t.name in w.faults.broken_tools:
            t.fn = _outage(t.name)
    return ToolEnv({t.name: t for t in tools}, world=w)


def _outage(name: str):
    def fn(**_):
        raise ToolFailure(f"{name} is unavailable (HTTP 503 from upstream service); try again later")
    return fn
