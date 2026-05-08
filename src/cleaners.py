from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd

SALES_DATE_KEYS = ["Date", "日期"]
SALES_ID_KEYS = ["Goods ID", "GoodsID", "商品ID", "goods id"]
SALES_NAME_KEYS = ["Goods Name", "商品名", "商品名称"]
SALES_FIELD_MAP = {
    "sales": ["sales", "Sales", "销售额", "成交金额", "Base price sales"],
    "buyers": ["Buyers", "buyers", "买家数", "支付买家数"],
    "total_order_items": ["Total order items", "订单商品数", "订单数"],
    "units_ordered": ["Units ordered", "下单件数", "销量", "件数"],
    "avg_units_per_order_item": ["Avg. units per order item", "平均每个订单商品件数"],
    "avg_sales_per_order_item": ["Avg. sales per order item", "平均每个订单商品销售额"],
}
SALES_STATUS_KEYS = [
    "Order status", "order status", "order_status", "订单状态", "订单状态名称",
    "Delivery status", "delivery status", "delivery_status", "物流状态", "配送状态",
    "Status", "status", "签收状态", "履约状态", "包裹状态",
]
SIGNED_STATUS_PATTERNS = [
    "已签收", "已完成", "已送达", "已收货", "交易成功",
    "delivered", "completed", "received", "signed", "signed for",
]
TRAFFIC_DATE_KEYS = ["Date", "日期"]
TRAFFIC_ID_KEYS = ["Goods ID", "GoodsID", "商品ID", "goods id"]
TRAFFIC_NAME_KEYS = ["Goods Name", "商品名", "商品名称"]
TRAFFIC_FIELD_MAP = {
    "impressions": ["Product impressions", "曝光量", "商品曝光量"],
    "visitor_impressions": ["Number of visitor impressions of the product", "访客曝光量"],
    "clicks": ["Product clicks", "点击量", "商品点击量"],
    "visitor_clicks": ["Number of visitor clicks on the product", "访客点击量"],
    "ctr_raw": ["CTR", "ctr", "点击率"],
}
MAP_ID_KEYS = ["Goods ID", "GoodsID", "商品ID", "goods id"]
MAP_SKU_KEYS = ["SKU", "sku", "Sku"]
MAP_STORE_KEYS = ["Store", "店铺", "Shop", "shop", "店铺名称"]
MAP_PRODUCT_KEYS = ["Product name", "Goods Name", "商品名称", "商品名"]
MAP_CREATED_KEYS = ["Date created", "创建时间", "上架时间"]
MAP_QTY_KEYS = ["Quantity", "库存", "Available quantity"]

def get_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        found = lower_map.get(str(cand).strip().lower())
        if found:
            return found
    return None

def normalize_goods_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan, "NaN": np.nan, "<NA>": np.nan})
    s = s.str.replace(r"\.0$", "", regex=True).str.replace(r"\s+", "", regex=True)
    return s.fillna("")

def parse_numeric_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    cleaned = (
        series.astype(str).str.strip()
        .replace({"": np.nan, "None": np.nan, "nan": np.nan, "--": np.nan, "N/A": np.nan, "n/a": np.nan})
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("MX$", "", regex=False)
        .str.replace("US$", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("￥", "", regex=False)
        .str.replace("¥", "", regex=False)
        .str.replace(r"[A-Za-z]+", "", regex=True)
        .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)

def parse_date_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    if parsed.isna().mean() > 0.6:
        excel_like = pd.to_numeric(series, errors="coerce")
        parsed_excel = pd.to_datetime("1899-12-30") + pd.to_timedelta(excel_like, unit="D")
        parsed = parsed.fillna(parsed_excel)
    return parsed.dt.floor("D")

def infer_store_from_filename(filename: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", filename or "")
    stem = stem.replace("_", "-").strip()
    if not stem:
        return "未分类店铺"
    parts = [p for p in stem.split("-") if p]
    if parts:
        return parts[0]
    return stem

def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().split(".")[-1]
    if suffix == "csv":
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def _pick(raw: pd.DataFrame, keys: List[str]) -> pd.Series | None:
    col = get_first_existing_column(raw, keys)
    if col:
        return raw[col]
    return None

def parse_signed_status(raw: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (is_signed, status_available) for sales/order rows.

    If the uploaded sales table has no recognizable status column, we keep the
    original rows as usable denominator rows and mark status_available=False so
    the dashboard can explain that the denominator could not be strictly filtered.
    """
    status_col = get_first_existing_column(raw, SALES_STATUS_KEYS)
    if not status_col:
        return (
            pd.Series([True] * len(raw), index=raw.index, dtype=bool),
            pd.Series([False] * len(raw), index=raw.index, dtype=bool),
        )
    status_text = raw[status_col].astype(str).str.strip().str.casefold()
    has_status = ~status_text.isin(["", "nan", "none", "null", "<na>"])
    pattern = "|".join(re.escape(x.casefold()) for x in SIGNED_STATUS_PATTERNS)
    is_signed = status_text.str.contains(pattern, na=False) & has_status
    return is_signed.fillna(False), has_status.fillna(False)

def clean_sales_df(raw: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    date_col = get_first_existing_column(raw, SALES_DATE_KEYS)
    id_col = get_first_existing_column(raw, SALES_ID_KEYS)
    name_col = get_first_existing_column(raw, SALES_NAME_KEYS)
    if not date_col or not id_col:
        raise ValueError("销售表缺少必要字段：Date / Goods ID")
    df = pd.DataFrame()
    df["date"] = parse_date_series(raw[date_col])
    df["goods_id"] = normalize_goods_id(raw[id_col])
    df["product_name"] = raw[name_col].astype(str).str.strip() if name_col else ""
    df["is_signed_order"], df["status_available"] = parse_signed_status(raw)
    for target, candidates in SALES_FIELD_MAP.items():
        s = _pick(raw, candidates)
        df[target] = parse_numeric_series(s) if s is not None else 0.0
    df["sales"] = np.where(
        (df["sales"] <= 0) & (df["total_order_items"] > 0) & (df["avg_sales_per_order_item"] > 0),
        df["total_order_items"] * df["avg_sales_per_order_item"],
        df["sales"],
    )
    df["store"] = infer_store_from_filename(source_name)
    df["source_file"] = source_name or ""
    df = df.dropna(subset=["date"])
    df = df[df["goods_id"] != ""].copy()
    return df

def clean_traffic_df(raw: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    date_col = get_first_existing_column(raw, TRAFFIC_DATE_KEYS)
    id_col = get_first_existing_column(raw, TRAFFIC_ID_KEYS)
    name_col = get_first_existing_column(raw, TRAFFIC_NAME_KEYS)
    if not date_col or not id_col:
        raise ValueError("流量表缺少必要字段：Date / Goods ID")
    df = pd.DataFrame()
    df["date"] = parse_date_series(raw[date_col])
    df["goods_id"] = normalize_goods_id(raw[id_col])
    df["product_name"] = raw[name_col].astype(str).str.strip() if name_col else ""
    for target, candidates in TRAFFIC_FIELD_MAP.items():
        s = _pick(raw, candidates)
        df[target] = parse_numeric_series(s) if s is not None else 0.0
    df["ctr_raw"] = df["ctr_raw"] / np.where(df["ctr_raw"].gt(1), 100, 1)
    df["store"] = infer_store_from_filename(source_name)
    df["source_file"] = source_name or ""
    df = df.dropna(subset=["date"])
    df = df[df["goods_id"] != ""].copy()
    return df

def clean_mapping_df(raw: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    id_col = get_first_existing_column(raw, MAP_ID_KEYS)
    sku_col = get_first_existing_column(raw, MAP_SKU_KEYS)
    if not id_col:
        raise ValueError("SKU映射表缺少必要字段：Goods ID")
    product_col = get_first_existing_column(raw, MAP_PRODUCT_KEYS)
    store_col = get_first_existing_column(raw, MAP_STORE_KEYS)
    created_col = get_first_existing_column(raw, MAP_CREATED_KEYS)
    qty_col = get_first_existing_column(raw, MAP_QTY_KEYS)

    df = pd.DataFrame()
    df["goods_id"] = normalize_goods_id(raw[id_col])
    df["sku"] = raw[sku_col].astype(str).str.strip() if sku_col else ""
    df["product_name"] = raw[product_col].astype(str).str.strip() if product_col else ""
    df["store"] = raw[store_col].astype(str).str.strip() if store_col else infer_store_from_filename(source_name)
    df["date_created"] = parse_date_series(raw[created_col]) if created_col else pd.NaT
    df["inventory_qty"] = parse_numeric_series(raw[qty_col]) if qty_col else 0.0
    df["source_file"] = source_name or ""
    df = df[df["goods_id"] != ""].copy()

    def _agg_sku(series: pd.Series) -> str:
        vals = [str(x).strip() for x in series if str(x).strip() and str(x).strip().lower() != "nan"]
        vals = list(dict.fromkeys(vals))
        return " / ".join(vals)

    grouped = (
        df.groupby("goods_id", as_index=False)
        .agg(
            sku=("sku", _agg_sku),
            product_name=("product_name", lambda s: next((str(x).strip() for x in s if str(x).strip() and str(x).strip().lower() != "nan"), "")),
            store=("store", lambda s: next((str(x).strip() for x in s if str(x).strip() and str(x).strip().lower() != "nan"), infer_store_from_filename(source_name))),
            date_created=("date_created", "min"),
            inventory_qty=("inventory_qty", "sum"),
        )
    )
    return grouped

def is_sales_file(raw: pd.DataFrame) -> bool:
    return bool(get_first_existing_column(raw, SALES_DATE_KEYS) and get_first_existing_column(raw, SALES_ID_KEYS) and (
        get_first_existing_column(raw, SALES_FIELD_MAP["sales"]) or
        get_first_existing_column(raw, SALES_FIELD_MAP["total_order_items"]) or
        get_first_existing_column(raw, SALES_FIELD_MAP["units_ordered"])
    ))

def is_traffic_file(raw: pd.DataFrame) -> bool:
    return bool(get_first_existing_column(raw, TRAFFIC_DATE_KEYS) and get_first_existing_column(raw, TRAFFIC_ID_KEYS) and (
        get_first_existing_column(raw, TRAFFIC_FIELD_MAP["impressions"]) or
        get_first_existing_column(raw, TRAFFIC_FIELD_MAP["clicks"])
    ))
