"""Write the body of the paper's main ablation table (Table 3) from the saved result CSVs.

Usage:
    uv run python -m figures.make_ablation_table

Reads ``outputs/main/table_summary.csv`` (3-fold CV, "mean ± std" strings) and
``outputs/main/table_final_test.csv`` (one run on the frozen test drivers), and writes the
``tabular`` environment to ``printout/sec/tab_ablation_results.tex``. Within each
initialization block, the best value of every column is bold (highest accuracy and macro-F1,
lowest log-loss and driver probe), compared after rounding to the printed 3 decimals.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CV_CSV = ROOT / "outputs" / "main" / "table_summary.csv"
TEST_CSV = ROOT / "outputs" / "main" / "table_final_test.csv"
OUT_FILE = ROOT / "printout" / "sec" / "tab_ablation_results.tex"

RUNGS: list[tuple[str, str]] = [
    ("A", "Baseline"),
    ("B", "+ GRL"),
    ("C", "+ Pose"),
    ("D", "Full GRL"),
    ("E", "+ Confusion"),
    ("F", "Full Conf."),
]
INITS: list[tuple[str, str]] = [("S", ""), ("I", " (ImageNet)")]  # ID suffix, CSV name suffix
# (column key, True if higher is better)
CV_COLS: list[tuple[str, bool]] = [
    ("accuracy", True),
    ("macro_f1", True),
    ("log_loss", False),
    ("driver_probe_acc", False),
]
TEST_COLS: list[tuple[str, bool]] = [
    ("accuracy", True),
    ("macro_f1", True),
    ("log_loss", False),
    ("driver_probe_acc", False),
]


@dataclass(frozen=True)
class Cell:
    """One printed table value.

    Attributes:
        mean: The value, or the mean over folds.
        std: Standard deviation over folds, None for single-run test values.
    """

    mean: float
    std: float | None


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read a results CSV keyed by its first column.

    Args:
        path: CSV file whose first column is the configuration name.

    Returns:
        Mapping configuration name to its row.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key = reader.fieldnames[0] if reader.fieldnames else ""
        return {row[key]: row for row in reader}


def parse_cv(value: str) -> Cell:
    """Parse a "mean ± std" string from the CV summary.

    Args:
        value: Text such as "0.507 ± 0.059".

    Returns:
        The parsed cell.

    Raises:
        ValueError: If the text is not in "mean ± std" form.
    """
    mean, sep, std = value.partition("±")
    if not sep:
        raise ValueError(f"expected 'mean ± std', got {value!r}")
    return Cell(float(mean), float(std))


def config_name(rung: str, desc_row: dict[str, dict[str, str]], suffix: str) -> str:
    """Find the CSV configuration name of one rung and initialization.

    Args:
        rung: Ladder letter, "A" to "F".
        desc_row: Rows of a results CSV keyed by configuration name.
        suffix: " (ImageNet)" for the ImageNet run, "" for scratch.

    Returns:
        The matching configuration name.

    Raises:
        KeyError: If there is not exactly one match.
    """
    matches = [
        name
        for name in desc_row
        if name.startswith(f"{rung}: ") and name.endswith(" (ImageNet)") == bool(suffix)
    ]
    if len(matches) != 1:
        raise KeyError(f"expected one config for {rung}{suffix!r}, found {matches}")
    return matches[0]


def best_values(cells: list[Cell], higher_is_better: bool) -> float:
    """Return the best rounded mean of one column within a block.

    Args:
        cells: The column's cells in one initialization block.
        higher_is_better: Whether a larger value is better.

    Returns:
        The best mean, rounded to the printed 3 decimals.
    """
    rounded = [round(c.mean, 3) for c in cells]
    return max(rounded) if higher_is_better else min(rounded)


def fmt(cell: Cell, bold: bool) -> str:
    """Format a cell as LaTeX math, bold when it is the best of its block.

    Args:
        cell: Value to print.
        bold: Whether to set it in bold.

    Returns:
        The LaTeX string, without surrounding dollar signs.
    """
    body = f"{cell.mean:.3f}" if cell.std is None else f"{cell.mean:.3f} \\pm {cell.std:.3f}"
    return f"\\mathbf{{{body}}}" if bold else body


def block_rows(
    cv: dict[str, dict[str, str]], test: dict[str, dict[str, str]], init: str, suffix: str
) -> list[str]:
    """Build the LaTeX rows of one initialization block.

    Args:
        cv: CV summary rows.
        test: Frozen-test rows.
        init: ID suffix, "S" or "I".
        suffix: CSV name suffix for this initialization.

    Returns:
        One LaTeX table row per rung.
    """
    cv_cells = [[parse_cv(cv[config_name(r, cv, suffix)][k]) for k, _ in CV_COLS] for r, _ in RUNGS]
    test_cells = [
        [Cell(float(test[config_name(r, test, suffix)][k]), None) for k, _ in TEST_COLS]
        for r, _ in RUNGS
    ]
    cv_best = [best_values([row[j] for row in cv_cells], hi) for j, (_, hi) in enumerate(CV_COLS)]
    test_best = [
        best_values([row[j] for row in test_cells], hi) for j, (_, hi) in enumerate(TEST_COLS)
    ]
    rows: list[str] = []
    for (rung, desc), cv_row, test_row in zip(RUNGS, cv_cells, test_cells, strict=True):
        cv_txt = [
            f"${fmt(c, round(c.mean, 3) == b)}$" for c, b in zip(cv_row, cv_best, strict=True)
        ]
        test_txt = [
            f"${fmt(c, round(c.mean, 3) == b)}$" for c, b in zip(test_row, test_best, strict=True)
        ]
        cells = [f"\\texttt{{{rung}-{init}}}", desc.replace(".", ".\\"), *cv_txt, *test_txt]
        rows.append("    " + " & ".join(cells) + " \\\\")
    return rows


def main() -> None:
    """Build the full tabular from both CSVs and write it next to the paper sections."""
    cv, test = read_rows(CV_CSV), read_rows(TEST_CSV)
    lines = [
        "% Generated by figures/make_ablation_table.py from outputs/main/table_summary.csv and",
        "% outputs/main/table_final_test.csv. Do not edit by hand, rerun the script instead.",
        "\\begin{tabular}{@{}llcccccccc@{}}",
        "    \\toprule",
        "    & & \\multicolumn{4}{c}{3-Fold Driver-Grouped Cross-Validation}"
        " & \\multicolumn{4}{c}{Frozen Test (4 Drivers)} \\\\",
        "    \\cmidrule(lr){3-6} \\cmidrule(l){7-10}",
        # The CV and test blocks use the same column names so they read side by side.
        "    ID & Configuration & Acc & Macro-F1 & Log-Loss & Probe"
        " & Acc & Macro-F1 & Log-Loss & Probe \\\\",
        "    \\midrule",
    ]
    for i, (init, suffix) in enumerate(INITS):
        if i:
            lines.append("    \\midrule")
        lines.extend(block_rows(cv, test, init, suffix))
    lines += ["    \\bottomrule", "\\end{tabular}", ""]
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT_FILE)


if __name__ == "__main__":
    main()
