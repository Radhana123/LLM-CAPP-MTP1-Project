# Puts the repo root on sys.path so tests can `import capp`.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _corrections_log_in_a_temp_folder(tmp_path_factory):
    """week6/app.py appends to a corrections log each time Generate is pressed: a test run must not write into the real results/ folder."""
    old = os.environ.get("CAPP_CORRECTIONS_LOG")
    os.environ["CAPP_CORRECTIONS_LOG"] = str(tmp_path_factory.mktemp("corrections") / "corrections_log.jsonl")
    yield
    if old is None:
        os.environ.pop("CAPP_CORRECTIONS_LOG", None)
    else:
        os.environ["CAPP_CORRECTIONS_LOG"] = old
