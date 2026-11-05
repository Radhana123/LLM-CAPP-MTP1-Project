"""Per-operation dimensions (Step 51): every operation gets the dimensions of ITS cut, filled from the drawing.

    from capp import op_dims as OPD
    ops   = OPD.autoselect(features)                              # the operations the features need (Facing and Inspection included)
    cells = OPD.cells(ops, features, extraction, part_dims)       # {op: {dim: {"value", "source": "drawing" | "part", "from"}}}
    per_op = OPD.per_op_from_values(cells, values)                # {op: {dim: mm}}: what goes into dims["per_op"] (week3/agents.op_dims)

Why. The time model reads four numbers (diameter, length, depth, width) for the whole part. A drawing gives different numbers to different
features (a hole of 8 mm and 25 mm deep, a slot 40 mm long, a step of 30 mm diameter). This module hands each operation the numbers of the
feature it makes, and the part's own number where the drawing gives none. Nothing is invented: a value outside the limits of the app is
dropped, never clipped.

Rules (each one is tested):
  * Which dimensions an operation uses is USES. It was read off week3/agents._op_time by varying each dimension (tests/test_op_dims.py
    repeats that check), so a field is shown only when the time model reads it.
  * An operation takes the numbers of the features it FINISHES: those whose first alternative chain (the one the Route Builder takes first)
    ends with it (Drilling for a Hole, Counterboring for a Counterbore). An operation that only prepares a feature (the drill before a
    reamer) gets no number of its own: it keeps the part's value, and the user can type one.
  * Several rows for one operation (two steps, two holes): the largest value, i.e. the longer cut. It is an upper bound, not an average.
  * A cell that shows the part's value is NOT written to dims["per_op"] unless the user changed it. A cell filled from the drawing is.
    So with no drawing numbers and no edit, per_op is empty and every result is what it was before this step.
  * The selection of operations does not change the route the planner makes. It only decides which operations show a row here; an
    operation of the route that has no row uses the part's dimensions.
"""
from capp import legacy  # noqa: F401
import agents
import feature_vocab as fv
import precedence_graph as pg

from capp import dimensions as DIM
from capp import process_tables as PT

PER_OP = agents.PER_OP
DIM_KEYS = ("diameter", "length", "depth", "width")
LABEL = {"diameter": "Diameter", "length": "Length", "depth": "Depth", "width": "Width"}
ROW_KEY = {"diameter": "diameter_mm", "length": "length_mm", "depth": "depth_mm", "width": "width_mm"}      # the key in a feature row
# what the number boxes accept (mm). Lower than the part's own limits (1 mm): a hole of 0.8 mm or a groove 0.5 mm deep is real.
LIMITS = {"diameter": (0.1, 500.0), "length": (0.1, 1000.0), "depth": (0.1, 300.0), "width": (0.1, 500.0)}
# numbers the calculator model (capp/calculator.py) reads from a drawing row besides the four dimensions the v1 time model reads
EXTRA_DRAWN = {"External Threading": ("pitch",), "Tapping": ("pitch",),
               "Grooving/Necking": ("width",), "Internal Grooving": ("width",), "Undercutting": ("width",)}      # the groove length of the calculator
EXTRA_LIMITS = {"pitch": (0.05, 20.0)}
EXTRA_ROW_KEY = {"pitch": "pitch_mm"}
SELECTION_KEY = "op_sel"
SIGNATURE_KEY = "op_sel_sig"
AUTO_KEY = "opdim_auto"

# every operation of the vocabulary: Facing first and Inspection last (every route starts and ends with them), the rest as in the process tables
ALL_OPS = (pg.ALWAYS_FIRST,) + tuple(o for o in PT.OPERATIONS if o not in (pg.ALWAYS_FIRST, pg.ALWAYS_LAST)) + (pg.ALWAYS_LAST,)

# the dimensions the time model (week3/agents._op_time) reads for each operation; () = a fixed time
USES = {
    "Facing": ("diameter", "depth"),
    "Plain/Cylindrical Turning": ("diameter", "length", "depth"),
    "Step Turning": ("diameter", "length", "depth"),
    "Taper Turning": ("diameter", "length", "depth"),
    "Contour Turning": ("diameter", "length", "depth"),
    "Eccentric Turning": ("diameter", "length", "depth"),
    "Grooving/Necking": ("diameter", "depth"),
    "Internal Grooving": ("diameter", "depth"),
    "Undercutting": ("diameter", "depth"),
    "Parting-off": ("diameter",),
    "External Threading": ("diameter", "length"),
    "Knurling": ("diameter", "length", "depth"),
    "Forming": ("diameter", "depth"),
    "Polishing/Burnishing": ("diameter", "length", "depth"),
    "Boring": ("diameter", "depth"),
    "Center Drilling": (),
    "Chamfering": (),
    "Counterboring": ("diameter", "depth"),
    "Countersinking": ("diameter", "depth"),
    "Drilling": ("diameter", "depth"),
    "Reaming": ("diameter", "depth"),
    "Tapping": ("diameter", "length"),
    "Inspection": (),
    "Angular Milling": ("length", "depth"),
    "Corner Rounding/Filleting": ("length", "depth"),
    "Dovetail Milling": ("length", "depth"),
    "Engraving": ("length", "depth"),
    "Face Milling": ("length", "depth"),
    "Form Milling": ("length", "depth"),
    "Gang Milling": ("length", "depth"),
    "Gear/Spline Milling": ("length", "depth"),
    "Helical Milling": ("diameter", "depth"),
    "Pocket Milling": ("length", "depth", "width"),
    "Profile Milling": ("length", "depth", "width"),
    "Slab/Peripheral Milling": ("length", "depth"),
    "Slot Milling": ("length", "depth"),
    "Spotfacing": ("diameter", "depth"),
    "Surface Contouring": ("length", "depth", "width"),
    "T-Slot Milling": ("length", "depth"),
    "Thread Milling": ("diameter", "length"),
    "Woodruff Keyway Milling": ("length", "depth"),
}


def key_of(op: str, dim: str) -> str:
    """The session / widget key of one cell."""
    return f"opdim|{op}|{dim}"


def uses(op: str) -> tuple:
    return USES.get(op, ())


def autoselect(features) -> list:
    """The operations the features need, in ALL_OPS order: Facing, the first alternative chain of every feature (the Route Builder's
    default, fv.get_required_operations_set) and Inspection. Names that are not features of the vocabulary are ignored."""
    valid = [f for f in features if fv.is_valid_feature(f)]
    need = set(fv.get_required_operations_set(valid)) | {pg.ALWAYS_FIRST, pg.ALWAYS_LAST}
    return [op for op in ALL_OPS if op in need]


def finishing(features) -> dict:
    """{operation: [feature]}: for each feature, the LAST operation of its first alternative chain is the one that makes it."""
    out = {}
    for f in features:
        if not fv.is_valid_feature(f):
            continue
        last = fv.get_operations_for_feature(f)[0][-1]
        out.setdefault(last, [])
        if f not in out[last]:
            out[last].append(f)
    return out


def drawing_values(extraction, features) -> dict:
    """{operation: {dim: {"value": mm, "from": text}}}: the numbers the drawing gives to the features each operation makes. Only features
    that are in `features` now count (one the user removed gives nothing); only the dimensions the operation uses (and the pitch of a
    threading operation, for the calculator model); values outside the limits are dropped. {} when there is no successful extraction or its dimension pass failed."""
    if not extraction or not extraction.get("success"):
        return {}
    read = (extraction.get("part_dimensions") or {}).get("read") or {}
    if not read.get("ok"):
        return {}
    rows = [r for r in read.get("features") or [] if isinstance(r, dict)]
    out = {}
    for op, feats in finishing(features).items():
        for dim in uses(op) + EXTRA_DRAWN.get(op, ()):
            lo, hi = LIMITS[dim] if dim in LIMITS else EXTRA_LIMITS[dim]
            key = ROW_KEY[dim] if dim in ROW_KEY else EXTRA_ROW_KEY[dim]
            found = []                                   # (value, feature, row)
            for r in rows:
                v = r.get(key)
                if r.get("feature") in feats and isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi:
                    found.append((float(v), r["feature"], r))
            if not found:
                continue
            value, name, row = max(found, key=lambda t: t[0])
            text = DIM.feature_numbers_text(row)
            n = len(found)
            out.setdefault(op, {})[dim] = {"value": value, "from": f"{name}" + (f" ({text})" if text else "") + (f", largest of {n}" if n > 1 else "")}
    return out


def cells(ops, features, extraction, part_dims, hole_diameter=None) -> dict:
    """{operation: {dim: {"value", "source", "from"}}} for the selected operations and the dimensions each uses. source "drawing": the
    drawing gives the number. source "part": the part's value (the sidebar fields; for the diameter of a hole operation the hole diameter
    when one is set). `part_dims` = {"diameter", "length", "depth", "width"}."""
    drawn = drawing_values(extraction, features)
    out = {}
    for op in ops:
        row = {}
        for dim in uses(op):
            hit = (drawn.get(op) or {}).get(dim)
            if hit:
                row[dim] = {"value": hit["value"], "source": "drawing", "from": hit["from"]}
                continue
            value, text = float(part_dims[dim]), "whole part"
            lo, hi = LIMITS[dim]
            if dim == "diameter" and op in agents.HOLE_OPS and hole_diameter and lo <= hole_diameter <= hi:
                value, text = float(hole_diameter), "hole diameter of the part"
            row[dim] = {"value": value, "source": "part", "from": text}
        out[op] = row
    return out


def sync_selection(state, features, key=SELECTION_KEY, sig_key=SIGNATURE_KEY) -> None:
    """Put the operations of `features` in state[key] when the features changed since the last time (or the selection was lost).
    A selection the user changed by hand is kept until the features change. `state` = st.session_state or a dict."""
    sig = tuple(sorted(f for f in features if fv.is_valid_feature(f)))
    if state.get(sig_key) != sig or key not in state:
        state[key] = autoselect(features)
        state[sig_key] = sig


def sync_state(state, cells_) -> None:
    """Fill the number boxes. A box that still holds the value this function put there last time follows the new value (a new drawing,
    another part diameter); a box the user changed keeps what the user typed; a box that is not in the state yet (new, or dropped because it
    was not drawn in a run) is filled."""
    last = state.setdefault(AUTO_KEY, {})
    for op, row in cells_.items():
        for dim, c in row.items():
            k = key_of(op, dim)
            if state.get(k) is None or state[k] == last.get(k):
                state[k] = float(c["value"])
            last[k] = float(c["value"])


def per_op_from_values(cells_, values) -> dict:
    """{operation: {dim: mm}} for dims["per_op"]: the cells filled from the drawing and the cells whose value differs from the one
    offered. A cell that shows the part's value and was not touched is left out (the model takes the part's value anyway). `values` =
    {key_of(op, dim): number}."""
    out = {}
    for op, row in cells_.items():
        for dim, c in row.items():
            v = values.get(key_of(op, dim))
            if v is None:
                continue
            lo, hi = LIMITS[dim]
            if not lo <= v <= hi:
                continue
            if c["source"] == "drawing" or abs(float(v) - c["value"]) > 1e-9:
                out.setdefault(op, {})[dim] = float(v)
    return out


def row_note(row_cells, values, op) -> str:
    """A few words for the last column: where the numbers of an operation come from, dimension by dimension
    ('diameter, depth: Hole (...) | whole part: length | edited: depth')."""
    if not USES.get(op):
        return "fixed time: no dimension used"
    drawn, part, edited = {}, [], []
    for dim, c in row_cells.items():
        v = values.get(key_of(op, dim))
        label = LABEL[dim].lower()
        if v is not None and abs(float(v) - c["value"]) > 1e-9:
            edited.append(label)
        elif c["source"] == "drawing":
            drawn.setdefault(c["from"], []).append(label)
        else:
            part.append(label)
    out = []
    if drawn:
        out.append("\U0001F4D0 " + "; ".join(f"{', '.join(dims)}: {src}" for src, dims in drawn.items()))
    if edited:
        out.append("\u270F\uFE0F edited: " + ", ".join(edited))
    if part:
        out.append("whole part" + (": " + ", ".join(part) if drawn or edited else ""))
    return " | ".join(out)
