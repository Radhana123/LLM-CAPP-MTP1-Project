"""Why the last scripts/run_eval.py stopped (MTP-1 Step 43).

scripts/g2.py starts one run_eval after the other and shows the status grid at the end. The reason a run stopped (a quota, a refused key,
five API errors in a row, an error in the code) is printed by that run in the middle of a long output, and the grid scrolls it away. So
run_eval also writes the reason, in one line, to <out>/last_stop.txt when it stops, and removes the file when it starts; g2.py reads the file
after a run that exited with a code other than 0 and prints the reason at the very end, under the grid.

The file is next to the run folders, not in one (a run folder is a result and is packed; this is a note about the last attempt). A missing
file means: run_eval stopped for a reason it could not write (a wrong argument, a sealed dataset without a freeze: those say so in the
first lines of its output) or it did not stop.
"""
import os

NAME = "last_stop.txt"
MAX_CHARS = 700


def path(out_dir: str) -> str:
    return os.path.join(out_dir, NAME)


def clear(out_dir: str) -> None:
    try:
        os.remove(path(out_dir))
    except OSError:
        pass


def write(out_dir: str, text: str) -> None:
    """Writes the reason; never raises (a note must not turn a stopped run into a second error)."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(path(out_dir), "w", encoding="utf-8") as fh:
            fh.write(str(text).strip() + "\n")
    except OSError:
        pass


def read(out_dir: str) -> str:
    """The reason on one line of plain ASCII (a console that is not UTF-8 must not fail on it), at most MAX_CHARS long; '' if there is none."""
    try:
        with open(path(out_dir), encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return ""
    text = " ".join(text.split()).encode("ascii", "replace").decode("ascii")
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS - 3] + "..."
