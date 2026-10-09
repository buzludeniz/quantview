"""Regression tests for the optional-kernel guard.

The guard used `Name: Any` with no assignment. That satisfies mypy but creates no
runtime binding, so when black_scholes was absent the guarded names did not exist
in the module namespace and every call path raised AttributeError instead of an
actionable ImportError. A plain `pip install quantview` with no [surfaces] extra
hit exactly that.

Each case runs in a subprocess. Whether a module binds a name is decided at
import time and cached in sys.modules, so patching the import hook in-process
would only exercise whichever version of the module was imported first.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Each module guards only the kernel names it actually imports, so the expected
# set is per module. Asserting the union against every module would fail on the
# ones that legitimately do not use, say, SVICurve.
GUARDED: dict[str, tuple[str, ...]] = {
    "quantview.analytics.portfolio": (
        "OptionParams",
        "OptionType",
        "black_scholes_greeks",
    ),
    "quantview.analytics.surface": (
        "SVICurve",
        "SVIFit",
        "VolSurface",
        "OptionParams",
        "OptionType",
        "black_scholes_price",
        "implied_volatility",
    ),
    "quantview.notebook.charts": (
        "SVICurve",
        "SVIFit",
        "VolSurface",
        "OptionParams",
        "black_scholes_price",
    ),
    "quantview.notebook.widgets": ("SVICurve",),
}

# An import hook, not sys.modules[...] = None: that blocks only the exact name and
# leaves submodule paths importable, so the absence would not be real.
PREAMBLE = """\
import builtins
import sys

_real_import = builtins.__import__


def _blocked(name, *args, **kwargs):
    if name == "black_scholes" or name.startswith("black_scholes."):
        raise ImportError("No module named " + repr(name))
    return _real_import(name, *args, **kwargs)


for _name in [m for m in sys.modules if m.startswith("black_scholes")]:
    del sys.modules[_name]

builtins.__import__ = _blocked
"""


def run_child(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    """Execute body in a fresh interpreter with the kernel blocked.

    Written to a file rather than passed with -c so the child source needs no
    escaping and a failure reports a readable traceback.
    """
    script = tmp_path / "child.py"
    script.write_text(PREAMBLE + textwrap.dedent(body), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(Path.cwd()),
    )


@pytest.mark.parametrize(("module_name", "guard_names"), list(GUARDED.items()))
def test_guarded_names_are_bound_to_none_when_the_kernel_is_absent(
    module_name: str, guard_names: tuple[str, ...], tmp_path: Path
):
    """The names must exist and be None.

    A bare annotation never binds at runtime, so hasattr() was False and the call
    paths raised AttributeError.
    """
    result = run_child(
        tmp_path,
        f"""
        import importlib

        module = importlib.import_module({module_name!r})
        names = {guard_names!r}
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, "never bound: " + repr(missing)
        not_none = [n for n in names if getattr(module, n, "x") is not None]
        assert not not_none, "bound to a stale value: " + repr(not_none)
        print("OK")
        """,
    )

    assert "OK" in result.stdout, f"stdout={result.stdout!r}\nstderr={result.stderr[-2000:]}"


def test_every_guarded_module_has_a_guard_entry():
    """The table above must cover every module using the module-level guard.

    Only unindented imports count. analytics.risk imports inside the function body,
    which needs no module binding and is already exercised by
    test_call_paths_raise_import_error_naming_the_install.
    """
    root = Path(__file__).resolve().parent.parent / "src" / "quantview"
    probing = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # ast rather than a text match: the guard imports sit inside a try block,
        # so they are indented but still module level. A function-local import
        # like analytics.risk's needs no module binding and is not one of these.
        for node in tree.body:
            # A try at module level wraps the guard; a try inside a function is a
            # local import that needs no binding, so only direct children of the
            # module body count.
            candidates = ast.walk(node) if isinstance(node, ast.Try) else [node]
            for sub in candidates:
                names: list[str] = []
                if isinstance(sub, ast.ImportFrom) and sub.module:
                    names.append(sub.module)
                elif isinstance(sub, ast.Import):
                    names.extend(alias.name for alias in sub.names)
                if any(n == "black_scholes" or n.startswith("black_scholes.") for n in names):
                    rel = path.relative_to(root.parent).with_suffix("")
                    probing.add(".".join(rel.parts))
                    break
            else:
                continue
            break

    assert probing == set(GUARDED), (
        "guard table is out of date; module-level kernel imports found: "
        f"{sorted(probing)}, table covers: {sorted(GUARDED)}"
    )


def test_call_paths_raise_import_error_naming_the_install(tmp_path: Path):
    """The user-facing failure must be ImportError with an actionable message."""
    result = run_child(
        tmp_path,
        """
        import pandas as pd

        from quantview.analytics import risk, surface
        from quantview.notebook import charts

        try:
            surface.chain_to_volsurface(
                spot=100.0,
                chain=pd.DataFrame({"K": [100.0], "C": [5.0], "P": [5.0], "IV": [0.2]}),
                expiry=1.0,
                risk_free_rate=0.05,
            )
            raise AssertionError("chain_to_volsurface did not raise")
        except ImportError as exc:
            assert "black_scholes" in str(exc), exc

        try:
            risk.greeks_buckets([])
            raise AssertionError("greeks_buckets did not raise")
        except ImportError as exc:
            assert "surfaces" in str(exc), exc

        try:
            charts.vol_surface_3d(None, None)
            raise AssertionError("vol_surface_3d did not raise")
        except ImportError as exc:
            assert "black_scholes" in str(exc), exc

        print("OK")
        """,
    )

    assert "OK" in result.stdout, f"stdout={result.stdout!r}\nstderr={result.stderr[-2000:]}"


def test_analytics_do_not_need_the_kernel_at_all(tmp_path: Path):
    """A curve fit must work on a bare install; that is the point of the extra."""
    result = run_child(
        tmp_path,
        """
        import numpy as np

        from quantview.analytics.curves import fit_nss, forward_curve, par_curve

        tenors = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        rates = np.array([0.040, 0.042, 0.043, 0.044, 0.0445, 0.045])
        fit = fit_nss(tenors, rates)
        assert np.all(np.isfinite(forward_curve(fit, tenors)))
        assert np.all(np.isfinite(par_curve(fit, tenors)))
        print("OK")
        """,
    )

    assert "OK" in result.stdout, f"stdout={result.stdout!r}\nstderr={result.stderr[-2000:]}"


def test_guard_does_not_break_the_normal_installed_case():
    """With the extra present the names must resolve to the real objects."""
    pytest.importorskip("black_scholes")

    from quantview.analytics import surface

    assert surface.VolSurface is not None
    assert surface.black_scholes_price is not None
