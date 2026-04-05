"""Analytics tools available to the agent's generated code."""

from __future__ import annotations

import inspect
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────

class AnalyticsTools:
    """
    Collection of analytics operations the agent can call.

    Each method is a tool the LLM can invoke by generating Python code.
    Methods that start with '_' are internal helpers and not exposed.
    """

    # Names exposed to the agent (order matters for the prompt)
    ACTION_NAMES = (
        "load_csv",
        "describe",
        "preview",
        "filter_rows",
        "count_by",
        "group_and_agg",
        "sort_table",
        "compute_stat",
        "plot_bar",
        "plot_line",
        "plot_scatter",
        "plot_histogram",
        "plot_pie",
        "plot_box",
    )

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        # tables is a shared reference — agent can add derived tables here
        self.tables = tables

    # ── Loading ───────────────────────────────────────────────

    def load_csv(self, path: str, table_name: str | None = None) -> dict[str, Any]:
        """
        Load a CSV file into the agent as a named table.
        Parameters:
          path: file path to the CSV (absolute or relative to the notebook)
          table_name: name to use for the table (default: filename without extension)
        Returns: dict with table_name, rows, columns
        Example: load_csv("../data/sales.csv", table_name="sales")
        """
        import re
        from pathlib import Path

        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: '{path}'")
        if resolved.suffix.lower() not in {".csv", ".xlsx", ".parquet"}:
            raise ValueError(f"Unsupported file type '{resolved.suffix}'. Use .csv, .xlsx, or .parquet.")

        # Derive table name from filename if not provided
        if not table_name:
            table_name = re.sub(r"[^a-zA-Z0-9]+", "_", resolved.stem.strip().lower()).strip("_")

        if resolved.suffix.lower() == ".csv":
            df = pd.read_csv(resolved)
        elif resolved.suffix.lower() == ".xlsx":
            df = pd.read_excel(resolved)
        else:
            df = pd.read_parquet(resolved)

        # Normalize column names
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )

        self.tables[table_name] = df
        return {
            "table_name": table_name,
            "rows": len(df),
            "columns": list(df.columns),
        }

    # ── Inspection ────────────────────────────────────────────

    def describe(self, table: str) -> dict[str, Any]:
        """
        Return schema and basic statistics for a table.
        Parameters:
          table: name of the loaded table (e.g. "sales")
        Returns: dict with rows, columns, dtypes, and numeric summary
        Example: info = describe("sales")
        """
        df = self._get(table)
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "numeric_summary": df.describe().to_dict(),
        }

    def preview(self, table: str, n: int = 5) -> list[dict]:
        """
        Return the first n rows of a table.
        Parameters:
          table: table name
          n: number of rows (default 5, max 25)
        Returns: list of row dicts
        Example: rows = preview("sales", n=10)
        """
        df = self._get(table)
        return df.head(min(n, 25)).to_dict(orient="records")

    # ── Transformation ────────────────────────────────────────

    def filter_rows(
        self,
        table: str,
        column: str,
        op: str,
        value: Any,
        save_as: str | None = None,
    ) -> dict[str, Any]:
        """
        Filter rows where column matches a condition.
        Parameters:
          table: source table name
          column: column to test
          op: one of ==, !=, >, <, >=, <=, contains, startswith, endswith
          value: value to compare against
          save_as: optional name to store the result as a new table AND as a variable
        Returns: dict with row_count and preview
        Note: when save_as is set, the filtered DataFrame is available as both a
              table name (for other tools) and as a Python variable in the same name.
        Example: result = filter_rows("sales", "region", "==", "West", save_as="west_sales")
                 print(result["row_count"])  # number of matching rows
                 print(west_sales)           # the filtered DataFrame directly
        """
        df = self._get(table)
        self._check_col(df, column)

        ops = {
            "==": df[column] == value,
            "!=": df[column] != value,
            ">":  df[column] > value,
            "<":  df[column] < value,
            ">=": df[column] >= value,
            "<=": df[column] <= value,
            "contains":   df[column].astype(str).str.contains(str(value), case=False, na=False),
            "startswith": df[column].astype(str).str.startswith(str(value), na=False),
            "endswith":   df[column].astype(str).str.endswith(str(value), na=False),
        }
        if op not in ops:
            raise ValueError(f"Unknown op '{op}'. Choose from: {', '.join(ops)}")

        result = df[ops[op]].reset_index(drop=True)
        if save_as:
            self.tables[save_as] = result

        return {"row_count": len(result), "preview": result.head(5).to_dict(orient="records")}

    def count_by(
        self,
        table: str,
        by: str | list[str],
        save_as: str | None = None,
    ) -> dict[str, Any]:
        """
        Count rows grouped by one or more columns.
        Use this when you want to know how many rows fall into each category.
        Parameters:
          table: source table name
          by: column name (or list of names) to group on
          save_as: optional name to store the result
        Returns: dict with result rows containing the group columns and a 'count' column
        Example: result = count_by("orders", "region", save_as="region_counts")
                 result = count_by("sleep", "sleep_disorder_risk", save_as="risk_counts")
        """
        df = self._get(table)
        by_cols = [by] if isinstance(by, str) else by
        for c in by_cols:
            self._check_col(df, c)

        result = df.groupby(by_cols, as_index=False).size().rename(columns={"size": "count"})
        result = result.sort_values("count", ascending=False).reset_index(drop=True)

        if save_as:
            self.tables[save_as] = result

        return {"rows": result.to_dict(orient="records")}

    def group_and_agg(
        self,
        table: str,
        by: str | list[str],
        column: str,
        agg: str,
        save_as: str | None = None,
    ) -> dict[str, Any]:
        """
        Group rows and aggregate a column.
        Parameters:
          table: source table name
          by: column name (or list of names) to group on
          column: column to aggregate
          agg: one of sum, mean, count, min, max, median, std
          save_as: optional name to store the result
        Returns: dict with result rows
        Example: result = group_and_agg("sales", "region", "revenue", "sum", save_as="rev_by_region")
        """
        df = self._get(table)
        by_cols = [by] if isinstance(by, str) else by
        for c in by_cols:
            self._check_col(df, c)
        self._check_col(df, column)

        valid_aggs = {"sum", "mean", "count", "min", "max", "median", "std"}
        if agg not in valid_aggs:
            raise ValueError(f"Unknown agg '{agg}'. Choose from: {', '.join(sorted(valid_aggs))}")

        result = df.groupby(by_cols, as_index=False)[column].agg(agg)
        result = result.sort_values(result.columns[-1], ascending=False).reset_index(drop=True)

        if save_as:
            self.tables[save_as] = result

        return {"rows": result.to_dict(orient="records")}

    def sort_table(
        self,
        table: str,
        by: str | list[str],
        ascending: bool | list[bool] = True,
        save_as: str | None = None,
    ) -> dict[str, Any]:
        """
        Sort a table by one or more columns.
        Parameters:
          table: source table name
          by: column name or list of column names
          ascending: True/False or list of True/False per column
          save_as: optional name to store the result
        Returns: dict with preview of sorted rows
        Example: result = sort_table("sales", "revenue", ascending=False, save_as="top_sales")
        """
        df = self._get(table)
        by_cols = [by] if isinstance(by, str) else by
        for c in by_cols:
            self._check_col(df, c)

        result = df.sort_values(by_cols, ascending=ascending).reset_index(drop=True)
        if save_as:
            self.tables[save_as] = result

        return {"preview": result.head(10).to_dict(orient="records")}

    def compute_stat(self, table: str, column: str, stat: str) -> dict[str, Any]:
        """
        Compute a single statistic on a column.
        Parameters:
          table: table name
          column: numeric column
          stat: one of mean, median, sum, min, max, std, count
        Returns: dict with label and value
        Example: result = compute_stat("sales", "revenue", "mean")
        """
        df = self._get(table)
        self._check_col(df, column)

        stat_fns = {
            "mean":   df[column].mean,
            "median": df[column].median,
            "sum":    df[column].sum,
            "min":    df[column].min,
            "max":    df[column].max,
            "std":    df[column].std,
            "count":  df[column].count,
        }
        if stat not in stat_fns:
            raise ValueError(f"Unknown stat '{stat}'. Choose from: {', '.join(sorted(stat_fns))}")

        return {"label": f"{stat}({column})", "value": stat_fns[stat]()}

    # ── Visualization ─────────────────────────────────────────

    def plot_bar(self, table: str, x: str, y: str, title: str = "") -> str:
        """
        Create a bar chart.
        Parameters:
          table: table name
          x: categorical column for x-axis
          y: numeric column for bar heights
          title: chart title (optional)
        Returns: confirmation string
        Example: plot_bar("rev_by_region", "region", "revenue", title="Revenue by Region")
        """
        df = self._get(table)
        self._check_col(df, x)
        self._check_col(df, y)

        plt.figure(figsize=(10, 5))
        plt.bar(df[x].astype(str), df[y])
        plt.title(title or f"{y} by {x}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        return f"Bar chart: {y} by {x}"

    def plot_line(self, table: str, x: str, y: str, title: str = "") -> str:
        """
        Create a line chart.
        Parameters:
          table: table name
          x: column for x-axis (often a date or ordered category)
          y: numeric column for y-axis
          title: chart title (optional)
        Returns: confirmation string
        Example: plot_line("monthly_sales", "month", "revenue", title="Monthly Revenue")
        """
        df = self._get(table)
        self._check_col(df, x)
        self._check_col(df, y)

        plt.figure(figsize=(10, 5))
        plt.plot(df[x].astype(str), df[y], marker="o")
        plt.title(title or f"{y} over {x}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        return f"Line chart: {y} over {x}"

    def plot_scatter(self, table: str, x: str, y: str, title: str = "") -> str:
        """
        Create a scatter plot to explore relationships.
        Parameters:
          table: table name
          x: numeric column for x-axis
          y: numeric column for y-axis
          title: chart title (optional)
        Returns: confirmation string
        Example: plot_scatter("sales", "discount", "profit", title="Discount vs Profit")
        """
        df = self._get(table)
        self._check_col(df, x)
        self._check_col(df, y)

        plt.figure(figsize=(8, 5))
        plt.scatter(df[x], df[y], alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title or f"{y} vs {x}")
        plt.tight_layout()
        plt.show()
        return f"Scatter plot: {y} vs {x}"

    def plot_histogram(self, table: str, column: str, bins: int = 20, title: str = "") -> str:
        """
        Create a histogram showing the distribution of a column.
        Parameters:
          table: table name
          column: numeric column to plot
          bins: number of bins (default 20)
          title: chart title (optional)
        Returns: confirmation string
        Example: plot_histogram("sales", "profit", bins=30)
        """
        df = self._get(table)
        self._check_col(df, column)

        plt.figure(figsize=(8, 5))
        plt.hist(df[column].dropna(), bins=bins, edgecolor="black", alpha=0.7)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(title or f"Distribution of {column}")
        plt.tight_layout()
        plt.show()
        return f"Histogram: {column}"

    def plot_pie(self, table: str, labels: str, values: str, title: str = "") -> str:
        """
        Create a pie chart for part-to-whole comparisons.
        Parameters:
          table: table name
          labels: column with category labels
          values: numeric column with slice sizes
          title: chart title (optional)
        Returns: confirmation string
        Example: plot_pie("segment_counts", "segment", "count", title="Orders by Segment")
        """
        df = self._get(table)
        self._check_col(df, labels)
        self._check_col(df, values)

        plt.figure(figsize=(7, 7))
        plt.pie(
            df[values],
            labels=df[labels].astype(str),
            autopct="%1.1f%%",
            startangle=140,
        )
        plt.title(title or f"{values} by {labels}")
        plt.tight_layout()
        plt.show()
        return f"Pie chart: {values} by {labels}"

    def plot_box(self, table: str, x: str, y: str, title: str = "") -> str:
        """
        Create a box plot to compare distributions across categories.
        Parameters:
          table: table name
          x: categorical column (groups)
          y: numeric column (values to distribute)
          title: chart title (optional)
        Returns: confirmation string
        Example: plot_box("orders", "region", "profit", title="Profit Distribution by Region")
        """
        df = self._get(table)
        self._check_col(df, x)
        self._check_col(df, y)

        categories = df[x].dropna().unique()
        data_by_cat = [df[df[x] == cat][y].dropna().values for cat in categories]

        plt.figure(figsize=(10, 5))
        plt.boxplot(data_by_cat, labels=[str(c) for c in categories], patch_artist=True)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title or f"{y} distribution by {x}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        return f"Box plot: {y} by {x}"

    # ── Helpers ───────────────────────────────────────────────

    def _get(self, table: str) -> pd.DataFrame:
        if table not in self.tables:
            available = ", ".join(sorted(self.tables)) or "(none loaded)"
            raise ValueError(f"Unknown table '{table}'. Available: {available}")
        return self.tables[table]

    def _check_col(self, df: pd.DataFrame, column: str) -> None:
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available: {', '.join(df.columns)}"
            )

    def describe_all(self) -> str:
        """Return a prompt-ready description of all tool methods."""
        lines = []
        for name in self.ACTION_NAMES:
            method = getattr(self, name)
            sig = inspect.signature(method)
            params = [p for p in sig.parameters if p != "self"]
            doc = inspect.getdoc(method) or ""
            lines.append(f"{name}({', '.join(params)})\n{doc}")
        return "\n\n".join(lines)
