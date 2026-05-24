# MTA — Archive (versions ≤ 2.0)

This folder contains the **original single-script version of MTA**,
preserved for users who still rely on it — typically for reproducing
earlier analyses or because their Stata do-files / shell pipelines
were written for `MTA.py`'s line-by-line input format.

> **New here?** Don't use this folder. Go back to the repository root
> and follow [`MTA-for-Master/README.md`](../MTA-for-Master/README.md).
> Version 3.0 is easier to install, has a visual interface, and adds
> several analyses that are not in the archived script.

## Contents

| File              | What it is                                                              |
| ----------------- | ----------------------------------------------------------------------- |
| `MTA.py`          | The original MTA script (last archived state: version 2.0, January 2025)|
| `requirements.txt`| Python dependencies for the archived script                             |
| `install.md`      | Installation instructions (Anaconda for Windows/macOS, pip for Unix)    |
| `doc.md`          | Getting started: paths, UTF-8, stopwords, language for plots            |
| `automate.md`     | How to automate `MTA.py` and use it from Stata via piped input files    |

These four documents are the reference for `MTA.py` and are kept here
**unchanged**. Their instructions apply to this archived script only.
For installation and usage instructions for the current version of
MTA, see [`../MTA-for-Master/README.md`](../MTA-for-Master/README.md).

## Status

  - **No new features will be added** to the archived script.
  - **Bug fixes are not actively backported.** If you find a bug here,
    the recommended fix is to migrate to version 3.0 in
    [`../MTA-for-Master/`](../MTA-for-Master/).
  - The script is left in place so that existing workflows continue to
    run and so that older analyses remain reproducible.

## Migrating to version 3.0

If you have a Stata do-file or shell pipeline that pipes a text input
file into `MTA.py` (the technique described in `automate.md`), the
replacement in version 3.0 is the argument-based batch mode of
`MTA_v3.py`. From `MTA-for-Master/`:

```bash
./.venv/bin/python code/MTA_v3.py \
    --corpus    /path/to/corpus \
    --stopwords /path/to/stop.txt \
    --action    nmf --n-topics 5 --language de
```

Run `python code/MTA_v3.py --help` for the full list of actions and
options, or pick **[3] Show batch / scripting usage** from the
launcher menu.

See [`../changes.md`](../changes.md) for the complete list of changes
introduced in version 3.0.
