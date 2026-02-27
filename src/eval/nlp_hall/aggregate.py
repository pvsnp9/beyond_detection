from typing import Any

try:
    import pandas as pd
except Exception:  # noqa: BLE001
    pd = None


DataFrameLike = Any


def _require_pandas() -> None:
    if pd is None:
        raise ImportError("pandas is required for aggregation outputs")


def records_to_dataframe(rows: list[dict[str, Any]]) -> DataFrameLike:
    _require_pandas()
    return pd.DataFrame(rows)


def _valid_rows(df: DataFrameLike) -> DataFrameLike:
    if "is_valid" not in df.columns:
        return df.copy()
    return df[df["is_valid"].astype(bool)].copy()


def aggregate_macro(
    df: DataFrameLike,
    metrics: list[str],
    group_cols: list[str],
) -> DataFrameLike:
    _require_pandas()
    valid = _valid_rows(df)
    if valid.empty:
        cols = list(group_cols) + [m for m in metrics if m in df.columns] + ["n_examples"]
        return pd.DataFrame(columns=cols)

    use_metrics = [m for m in metrics if m in valid.columns]

    if not group_cols:
        row: dict[str, Any] = {m: float(valid[m].mean()) for m in use_metrics}
        row["n_examples"] = int(len(valid))
        return pd.DataFrame([row])

    grouped = valid.groupby(group_cols, dropna=False)
    means = grouped[use_metrics].mean().reset_index() if use_metrics else grouped.size().reset_index(name="_tmp")
    counts = grouped.size().reset_index(name="n_examples")
    if "_tmp" in means.columns:
        means = means.drop(columns=["_tmp"])
    return means.merge(counts, on=group_cols, how="left")


def _safe_prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    r = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f1)


def aggregate_micro_facts(df: DataFrameLike, group_cols: list[str]) -> DataFrameLike:
    _require_pandas()
    valid = _valid_rows(df)
    need_cols = ["tp", "fp", "fn", "n_ref", "n_pred"]
    if valid.empty or any(c not in valid.columns for c in need_cols):
        cols = list(group_cols) + ["tp", "fp", "fn", "n_ref", "n_pred", "micro_add", "micro_omit", "micro_p", "micro_r", "micro_f1"]
        return pd.DataFrame(columns=cols)

    if not group_cols:
        tp = float(valid["tp"].sum())
        fp = float(valid["fp"].sum())
        fn = float(valid["fn"].sum())
        n_ref = float(valid["n_ref"].sum())
        n_pred = float(valid["n_pred"].sum())
        p, r, f1 = _safe_prf(tp, fp, fn)
        micro_add = (fp / n_pred) if n_pred > 0 else 0.0
        micro_omit = (fn / n_ref) if n_ref > 0 else 0.0
        return pd.DataFrame(
            [
                {
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "n_ref": int(n_ref),
                    "n_pred": int(n_pred),
                    "micro_add": float(micro_add),
                    "micro_omit": float(micro_omit),
                    "micro_p": float(p),
                    "micro_r": float(r),
                    "micro_f1": float(f1),
                }
            ]
        )

    grouped = valid.groupby(group_cols, dropna=False)[need_cols].sum().reset_index()
    out_rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        tp = float(row["tp"])
        fp = float(row["fp"])
        fn = float(row["fn"])
        n_ref = float(row["n_ref"])
        n_pred = float(row["n_pred"])
        p, r, f1 = _safe_prf(tp, fp, fn)
        add = (fp / n_pred) if n_pred > 0 else 0.0
        omit = (fn / n_ref) if n_ref > 0 else 0.0
        rec = {col: row[col] for col in group_cols}
        rec.update(
            {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "n_ref": int(n_ref),
                "n_pred": int(n_pred),
                "micro_add": float(add),
                "micro_omit": float(omit),
                "micro_p": float(p),
                "micro_r": float(r),
                "micro_f1": float(f1),
            }
        )
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def explode_quality_flags(df: DataFrameLike) -> DataFrameLike:
    _require_pandas()
    if "quality_flags" not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ["quality_flag"])

    out = df.copy()

    def to_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        if not text:
            return []
        return [part.strip() for part in text.split("|") if part.strip()]

    out["quality_flag"] = out["quality_flags"].apply(to_list)
    out = out.explode("quality_flag")
    out = out[out["quality_flag"].notna() & (out["quality_flag"] != "")]
    return out


def _merge_macro_micro(macro: DataFrameLike, micro: DataFrameLike, group_cols: list[str]) -> DataFrameLike:
    _require_pandas()
    if macro.empty and micro.empty:
        return pd.DataFrame()
    if macro.empty:
        return micro
    if micro.empty:
        return macro
    if not group_cols:
        row = {**macro.iloc[0].to_dict(), **micro.iloc[0].to_dict()}
        return pd.DataFrame([row])
    return macro.merge(micro, on=group_cols, how="outer")


def aggregate_all(df: DataFrameLike) -> dict[str, DataFrameLike]:
    _require_pandas()
    metrics = ["add", "omit", "p", "r", "f1", "eg", "eh", "ec", "eqs", "k_claims", "len_tokens"]

    overall_macro = aggregate_macro(df, metrics=metrics, group_cols=[])
    overall_micro = aggregate_micro_facts(df, group_cols=[])
    overall = _merge_macro_micro(overall_macro, overall_micro, group_cols=[])

    by_model = _merge_macro_micro(
        aggregate_macro(df, metrics=metrics, group_cols=["model_key"]),
        aggregate_micro_facts(df, group_cols=["model_key"]),
        group_cols=["model_key"],
    )
    by_source = _merge_macro_micro(
        aggregate_macro(df, metrics=metrics, group_cols=["source"]),
        aggregate_micro_facts(df, group_cols=["source"]),
        group_cols=["source"],
    )
    by_modality = _merge_macro_micro(
        aggregate_macro(df, metrics=metrics, group_cols=["modality"]),
        aggregate_micro_facts(df, group_cols=["modality"]),
        group_cols=["modality"],
    )
    by_model_source_modality = _merge_macro_micro(
        aggregate_macro(df, metrics=metrics, group_cols=["model_key", "source", "modality"]),
        aggregate_micro_facts(df, group_cols=["model_key", "source", "modality"]),
        group_cols=["model_key", "source", "modality"],
    )

    exploded = explode_quality_flags(df)
    by_quality_flag = _merge_macro_micro(
        aggregate_macro(exploded, metrics=metrics, group_cols=["quality_flag"]),
        aggregate_micro_facts(exploded, group_cols=["quality_flag"]),
        group_cols=["quality_flag"],
    )

    return {
        "overall": overall,
        "by_model": by_model,
        "by_source": by_source,
        "by_modality": by_modality,
        "by_model_source_modality": by_model_source_modality,
        "by_quality_flag": by_quality_flag,
    }
