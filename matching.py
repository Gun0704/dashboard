from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import re

from .metrics import safe_divide

_INVALID_STORE_VALUES = {"", "0", "0.0", "nan", "none", "null", "<na>"}


def _clean_store_value(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\.0$", "", text)
    if text.casefold() in _INVALID_STORE_VALUES:
        return ""
    return text


def _canonicalize_store_series(series: pd.Series) -> pd.Series:
    cleaned = series.map(_clean_store_value)
    mapping = {}
    for value in cleaned:
        if value:
            mapping.setdefault(value.casefold(), value)
    return cleaned.map(lambda x: mapping.get(x.casefold(), x) if x else "未分类店铺")


def build_detail_dataset(sales_df: pd.DataFrame, traffic_df: pd.DataFrame, map_df: pd.DataFrame, conversion_basis: str) -> pd.DataFrame:
    sales_grouped = pd.DataFrame(columns=["date", "goods_id", "product_name", "store", "sales_amount", "buyers", "total_order_items", "units_ordered"]) if sales_df.empty else (
        sales_df.groupby(["date", "goods_id", "store"], as_index=False).agg(
            product_name=("product_name", "first"),
            sales_amount=("sales", "sum"),
            buyers=("buyers", "sum"),
            total_order_items=("total_order_items", "sum"),
            units_ordered=("units_ordered", "sum"),
        )
    )
    traffic_grouped = pd.DataFrame(columns=["date", "goods_id", "product_name", "store", "impressions", "clicks"]) if traffic_df.empty else (
        traffic_df.groupby(["date", "goods_id", "store"], as_index=False).agg(
            product_name=("product_name", "first"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
        )
    )
    merged = pd.merge(traffic_grouped, sales_grouped, on=["date", "goods_id", "store"], how="outer", suffixes=("_traffic", "_sales"))
    if not map_df.empty:
        merged = merged.merge(map_df, on="goods_id", how="left", suffixes=("", "_map"))
    else:
        merged["sku"] = ""
        merged["product_name"] = ""
        merged["inventory_qty"] = 0
        merged["date_created"] = pd.NaT

    for col in ["sales_amount", "buyers", "total_order_items", "units_ordered", "impressions", "clicks", "inventory_qty"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
        else:
            merged[col] = 0
    merged["date_created"] = pd.to_datetime(merged.get("date_created"), errors="coerce")

    merged["product_name"] = merged.get("product_name", "")
    merged["product_name"] = merged["product_name"].replace(0, "").fillna("").astype(str)
    merged["product_name"] = merged["product_name"].mask(merged["product_name"].isin(["", "nan", "None"]), merged.get("product_name_sales", "").astype(str))
    merged["product_name"] = merged["product_name"].mask(merged["product_name"].isin(["", "nan", "None"]), merged.get("product_name_traffic", "").astype(str))

    store_series = merged.get("store", "")
    if not isinstance(store_series, pd.Series):
        store_series = pd.Series([store_series] * len(merged), index=merged.index)
    merged["store"] = _canonicalize_store_series(store_series)

    if conversion_basis == "订单商品数":
        merged["orders"] = merged["total_order_items"]
    elif conversion_basis == "下单件数":
        merged["orders"] = merged["units_ordered"]
    else:
        merged["orders"] = merged["buyers"]

    merged["sku"] = merged.get("sku", "").fillna("").astype(str).str.strip()
    merged["display_sku"] = merged["sku"].mask(merged["sku"].isin(["", "nan", "None"]), merged["goods_id"].astype(str))
    merged["ctr"] = safe_divide(merged["clicks"], merged["impressions"])
    merged["conversion_rate"] = safe_divide(merged["orders"], merged["clicks"])
    merged["avg_sales_per_order"] = safe_divide(merged["sales_amount"], merged["orders"])
    merged["avg_units_per_order"] = safe_divide(merged["units_ordered"], merged["orders"])
    merged["has_sales"] = (merged["sales_amount"] > 0) | (merged["orders"] > 0) | (merged["units_ordered"] > 0)
    merged["has_traffic"] = (merged["impressions"] > 0) | (merged["clicks"] > 0)
    return merged


def build_match_check(sales_df: pd.DataFrame, traffic_df: pd.DataFrame, mapping_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    sales_ids = set(sales_df["goods_id"].dropna().astype(str)) if not sales_df.empty else set()
    traffic_ids = set(traffic_df["goods_id"].dropna().astype(str)) if not traffic_df.empty else set()
    mapping_ids = set(mapping_df["goods_id"].dropna().astype(str)) if not mapping_df.empty else set()
    sales_not_in_mapping = sorted(sales_ids - mapping_ids)
    traffic_not_in_mapping = sorted(traffic_ids - mapping_ids)
    sales_not_in_traffic = sorted(sales_ids - traffic_ids)
    traffic_not_in_sales = sorted(traffic_ids - sales_ids)
    summary = {
        "销售表Goods ID数": len(sales_ids),
        "流量表Goods ID数": len(traffic_ids),
        "映射表Goods ID数": len(mapping_ids),
        "销售表未映射数": len(sales_not_in_mapping),
        "流量表未映射数": len(traffic_not_in_mapping),
        "销售有但流量没有": len(sales_not_in_traffic),
        "流量有但销售没有": len(traffic_not_in_sales),
    }
    details = {
        "sales_not_in_mapping": sales_not_in_mapping,
        "traffic_not_in_mapping": traffic_not_in_mapping,
        "sales_not_in_traffic": sales_not_in_traffic,
        "traffic_not_in_sales": traffic_not_in_sales,
    }
    return summary, details
