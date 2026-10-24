#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mta_commands.py
===============

Translates the settings of the Streamlit app into batch commands of
MTA_v3.py (MTA 3.7).

Each analysis in the app has an exact counterpart in the batch mode, so
the app can show the command that reproduces what is on screen, and keep
a log of the analyses of a session that can be saved as a script (.sh for
Mac/Linux, .bat for Windows) and refined by hand — as one knows it from
the command window of Stata or the script of R.

This module has no Streamlit dependency: it only builds argument lists
and renders them, so it can be tested on its own (test_commands.py).

Placeholders
------------
The app knows the names of the uploaded files, but not the folder they
came from. Commands therefore use three placeholders that the script
defines at its top and that the user can edit:

    CORPUS     the folder with the .txt files
    STOPWORDS  the stopword list
    GROUPS     the CSV with groups (page 6, only when used)

Every command is self-contained: it repeats the corpus, the matrices and
the topic model it builds on, because the batch mode recomputes them for
each action. The output of each command goes to its own folder.
"""

from __future__ import annotations

import datetime
import re
import shlex
from typing import Dict, List, Optional

# Placeholders inside argument lists
CORPUS = "\x00CORPUS"
STOPWORDS = "\x00STOPWORDS"
GROUPS = "\x00GROUPS"
OUTPUT = "\x00OUTPUT:"          # followed by the folder name of the step
_VARS = {CORPUS: "CORPUS", STOPWORDS: "STOPWORDS", GROUPS: "GROUPS"}

MTA_VERSION = "3.7"


# =============================================================================
# 1. ARGUMENT LISTS
# =============================================================================

def num(x) -> str:
    """A number as the batch mode reads it back to the same value:
    integers without decimals, floats with repr (exact round trip)."""
    if isinstance(x, bool):
        raise TypeError("bool is not a number here")
    if hasattr(x, "__index__"):          # int, numpy integer
        return str(int(x))
    return repr(float(x))


def min_df_arg(x) -> str:
    """min_df: an integer is a number of documents, a float a proportion.
    The batch mode reads '2' as int and '0.01' (or '1.0') as float."""
    return num(x)


def corpus_args(corpus: Dict) -> List[str]:
    """Corpus and matrices: what page 1 (and the chart language) fixes.

    corpus = {"min_word_length", "min_df", "max_df", "language"}
    """
    return ["--corpus", CORPUS, "--stopwords", STOPWORDS,
            "--language", corpus.get("language", "en"),
            "--min-word-length", num(corpus["min_word_length"]),
            "--min-df", min_df_arg(corpus["min_df"]),
            "--max-df", num(float(corpus["max_df"]))]


def model_args(model: str, settings: Dict) -> List[str]:
    """The topic model a result builds on.

    model = 'nmf' -> settings {"n_topics", "init", "seed"}
    model = 'lda' -> settings {"n_topics", "seed"}
    """
    a = ["--n-topics", num(int(settings["n_topics"]))]
    if model == "nmf":
        a += ["--init", settings["init"]]
    a += ["--seed", num(int(settings["seed"]))]
    return a


def words_arg(words: List[str]) -> str:
    return ",".join(words)


def command(label: str, corpus: Dict, action: List[str]) -> List[str]:
    """Full argument list of one step: corpus + action-specific part +
    its own output folder (named after `label`; numbered in a script)."""
    return _attach_dash_values(corpus_args(corpus) + action + ["--output", OUTPUT + label])


_OPTION = re.compile(r"--[a-z][a-z0-9-]*$")


def _attach_dash_values(args: List[str]) -> List[str]:
    """A value that begins with '-' (a separator '-_', a label '-links')
    would be taken by argparse for an option: write it as --opt=value."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else None
        if (_OPTION.match(a) and nxt is not None and nxt.startswith("-")
                and not _OPTION.match(nxt)):
            out.append(f"{a}={nxt}")
            i += 2
            continue
        out.append(a)
        i += 1
    return out


# =============================================================================
# 2. RENDERING
# =============================================================================

_WIN_SPECIAL = re.compile(r'[\s&|<>^(),;=%!"]')


def _quote_win(arg: str) -> str:
    """Quote one argument for cmd.exe + the C runtime of python.exe.

    A double quote inside a value cannot pass cmd.exe safely (it toggles
    cmd's own quoting, so a following ^ would no longer continue the
    line); it is written as a single quote. Words and labels never need it.
    """
    arg = arg.replace('"', "'")
    if arg and not _WIN_SPECIAL.search(arg):
        return arg
    # backslashes before a quote must be doubled, the quote escaped
    out, bs = [], 0
    for ch in arg:
        if ch == "\\":
            bs += 1
            continue
        if ch == '"':
            out.append("\\" * (2 * bs + 1) + '"')
        else:
            out.append("\\" * bs + ch)
        bs = 0
    out.append("\\" * (2 * bs))
    return '"' + "".join(out) + '"'


def _win_path(v: str) -> str:
    """No trailing separator (it would escape the closing quote), but keep
    a drive root a root: C:\\ -> C:\\. (C: alone is the current folder on C)."""
    v = v.replace('"', "")
    if re.fullmatch(r"[A-Za-z]:[\\/]*", v):
        return v[:2] + "\\."
    return v.rstrip("\\/") or v


def _posix_path(v: str) -> str:
    """Quoted for bash, but ~/ at the start is left to the shell."""
    if v == "~":
        return "~"
    if v.startswith("~/"):
        return "~/" + shlex.quote(v[2:]) if v[2:] else "~/"
    return shlex.quote(v)


def render(args: List[str], platform: str = "posix", script: bool = False,
           values: Optional[Dict[str, str]] = None, out_root: str = "MTA-Results",
           index: Optional[int] = None, multiline: bool = False) -> str:
    """Render one argument list as a command line.

    platform : 'posix' (Mac/Linux) or 'windows'
    script   : True inside a script (placeholders become variables, % is
               doubled in .bat); False for a line to paste into a terminal
               (placeholders become the values given in `values`).
    multiline: one option per line (continued with \\ or ^)
    index    : number of the step in a script; its output folder is then
               01_<label>, 02_<label>, … so that steps never overwrite
               each other.
    """
    values = values or {}
    win = platform == "windows"
    py = (r".venv\Scripts\python.exe code\MTA_v3.py" if win
          else ".venv/bin/python code/MTA_v3.py")
    parts = [py]
    starts = [0]                      # indices of parts that begin a line
    for a in args:
        if a.startswith("--"):
            starts.append(len(parts))
        if a in _VARS:
            name = _VARS[a]
            if script:
                parts.append(f'"%{name}%"' if win else f'"${name}"')
            else:
                v = values.get(name, name.lower())
                parts.append(_quote_win(_win_path(v)) if win else _posix_path(v))
            continue
        if a.startswith(OUTPUT):
            step = a[len(OUTPUT):]
            if index is not None:
                step = step_name(index, step)
            if script:
                parts.append(f'"%OUT%\\{step}"' if win else f'"$OUT/{step}"')
            else:
                parts.append(f"{out_root}\\{step}" if win else f"{out_root}/{step}")
            continue
        if win:
            q = _quote_win(a)
            if script:
                q = q.replace("%", "%%")
            parts.append(q)
        else:
            parts.append(shlex.quote(a))
    if not multiline:
        return " ".join(parts)
    lines = [" ".join(parts[a:b]) for a, b in zip(starts, starts[1:] + [len(parts)])]
    cont = " ^\n    " if win else " \\\n    "
    return cont.join(lines)


def _comment(line: str, win: bool) -> str:
    return ("REM " if win else "# ") + line if line else ("REM" if win else "#")


def script(entries: List[Dict], platform: str, values: Dict[str, str],
           header: Optional[List[str]] = None) -> str:
    """The session log as a script.

    entries : [{"time", "title", "args", "notes"}] in the order run
    values  : {"CORPUS", "STOPWORDS", "GROUPS"} — written once at the top
    header  : extra comment lines (corpus description)
    """
    win = platform == "windows"
    values = {k: v for k, v in values.items() if v}
    L = []
    if win:
        L += ["@echo off", "chcp 65001 >nul"]
    else:
        L += ["#!/bin/bash"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    L += [_comment(f"MTA {MTA_VERSION} — session log of {now}", win),
          _comment("", win),
          _comment("Each command reproduces one analysis of the app with the settings", win),
          _comment("used there. Save this file in the MTA-for-Master folder (next to", win),
          _comment("start_MTA), check the paths below, then run it" +
                   (" by double-click or from the Command Prompt (cmd)." if win
                    else ": bash <this file>"), win),
          _comment("Delete the commands you do not need and change the settings at will;", win),
          _comment("python code/MTA_v3.py --help lists all options.", win)]
    for h in header or []:
        L.append(_comment(h, win))
    L.append("")
    if win:
        def _set(name, default):
            v = _win_path(values.get(name, default)).replace("%", "%%")
            return f'set "{name}={v}"'
        L += ['cd /d "%~dp0"', _set("CORPUS", "corpus"),
              _set("STOPWORDS", "stopwords.txt")]
        if any(GROUPS in e["args"] for e in entries):
            L.append(_set("GROUPS", "groups.csv"))
        L.append('set "OUT=MTA-Results"')
    else:
        L += ['cd "$(dirname "$0")" || exit 1',
              f'CORPUS={_posix_path(values.get("CORPUS", "corpus"))}',
              f'STOPWORDS={_posix_path(values.get("STOPWORDS", "stopwords.txt"))}']
        if any(GROUPS in e["args"] for e in entries):
            L.append(f'GROUPS={_posix_path(values.get("GROUPS", "groups.csv"))}')
        L.append("OUT=MTA-Results")
    for i, e in enumerate(entries, 1):
        L += ["", _comment(f"[{i}] {e['time']} — {e['title']}", win)]
        for n in e.get("notes") or []:
            L.append(_comment("    " + n, win))
        L.append(render(e["args"], platform, script=True, index=i, multiline=True))
    L.append("")
    if win:
        L += ["echo.", "echo Done. The results are in %OUT%.", "pause", ""]
    else:
        L += ['echo "Done. The results are in $OUT."', ""]
    text = "\n".join(L)
    return text.replace("\n", "\r\n") if win else text


def step_name(index: int, action_label: str) -> str:
    """Output folder of a step: 01_nmf, 02_word-weights, …"""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", action_label).strip("-")
    return f"{index:02d}_{safe}"
