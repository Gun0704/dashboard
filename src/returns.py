from __future__ import annotations

import io
import re
from typing import Iterable

import pandas as pd

from src.cleaners import get_first_existing_column, infer_store_from_filename, parse_numeric_series

STORE_KEYS = ["Store", "店铺", "Shop", "shop", "店铺名称"]
RETURN_ID_KEYS = ["Return ID", "退货单ID", "售后单ID", "售后编号"]
ORDER_ID_KEYS = ["Order ID", "订单ID", "订单号"]
RETURN_STATUS_KEYS = ["Return status", "售后单状态", "退货状态", "退款状态"]
SKU_ID_KEYS = ["SKU ID", "SKU", "sku id", "sku", "商品SKU ID"]
ORDER_ITEM_ID_KEYS = ["Order item ID", "订单商品ID", "订单项ID"]
REASON_KEYS = ["Reason for request", "申请理由", "退货原因", "退款原因"]
RETURN_QTY_KEYS = ["Return quantity", "退货数量", "退款数量", "数量"]
AMOUNT_REQUEST_KEYS = ["Amount request to refund", "申请退款金额", "退款申请金额"]
AMOUNT_REFUND_KEYS = ["Amount refund to buyer", "退还给买家的金额", "实际退款金额", "退款金额"]
ORDER_DATE_KEYS = ["Order date", "下单日期", "订单日期"]
REQUESTED_DATE_KEYS = ["Requested date", "申请日期", "发起时间", "申请时间"]
COURIER_KEYS = ["Courier", "承运商"]
TRACKING_KEYS = ["Tracking number", "运单号", "物流单号"]
SERVICE_TYPE_KEYS = ["Types of after-sales service", "售后类型", "售后服务类型"]
RETURNLESS_KEYS = ["Returnless resolution", "仅退款解决方案", "无需退货处理"]
COST_RESPONSIBLE_KEYS = ["Responsible for covering after-sale cost", "售后费用承担方", "责任方"]

STATUS_MAP = {"refunded": "已退款", "not refunded yet": "尚未退款", "denied": "已拒绝", "rejected": "已拒绝", "closed": "已拒绝"}
SERVICE_TYPE_MAP = {"returnless refund": "仅退款", "return and refund": "退货退款"}
FIRST_REASON_RULES = [
    ("配送和产品问题", r"product\s+and\s+shipping\s+box\s+both\s+damaged|商品和运输包装均损坏"),
    ("配送问题", r"haven'?t\s+received|not\s+received|package\s+not\s+received|delivery\s+failed|派送失败|还没有收到包裹|未收到"),
    ("客户问题", r"no\s+longer\s+need|price\s+difference|inaccurate\s+website\s+description|网站描述不准确|不再需要|降价补差"),
    ("发货问题", r"wrong\s+item|missing\s+item|错发|少件|漏发"),
    ("产品问题", r"defective|doesn'?t\s+work|broken\s+parts|missing\s+or\s+broken\s+parts|damaged\s+but\s+shipping\s+box|product\s+damaged.*shipping\s+box\s+ok|商品损坏|有缺陷|无法使用|缺少配件|配件损坏"),
]

COLUMN_RENAME = {
    "store": "店铺", "return_id": "售后单ID", "order_id": "订单ID", "return_status": "售后单状态", "sku_id": "SKU ID",
    "reason_for_request": "申请理由", "return_quantity": "退货数量", "amount_request_to_refund": "申请退款金额", "amount_refund_to_buyer": "退还给买家的金额",
    "order_date": "下单日期", "requested_date": "申请日期", "service_type": "售后类型", "first_reason": "一级原因",
    "order_return_rate": "订单维度 退货率", "unit_return_rate": "产品件数 退货率", "refund_amount_metric": "退款金额",
    "return_order_count_metric": "退货退款订单数", "return_unit_count_metric": "退货退款件数",
}

def _pick(raw: pd.DataFrame, keys: list[str]) -> pd.Series | None:
    col = get_first_existing_column(raw, keys)
    return raw[col] if col else None

def _text_series(raw: pd.DataFrame, keys: list[str], default: str = "") -> pd.Series:
    s = _pick(raw, keys)
    if s is None:
        return pd.Series([default] * len(raw), index=raw.index, dtype="object")
    return s.astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})

def parse_return_datetime(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    cleaned = (series.astype(str).str.strip()
        .str.replace(r"\s+[A-Z]{2,5}\(UTC[+-]\d+\)$", "", regex=True)
        .str.replace(r"\s+[A-Z]{2,5}$", "", regex=True)
        .str.replace(r"\s+UTC[+-]?\d+$", "", regex=True))
    parsed = pd.to_datetime(cleaned, errors="coerce")
    if parsed.isna().mean() > 0.6:
        excel_like = pd.to_numeric(series, errors="coerce")
        parsed_excel = pd.to_datetime("1899-12-30") + pd.to_timedelta(excel_like, unit="D")
        parsed = parsed.fillna(parsed_excel)
    return parsed.dt.floor("D")

def classify_first_reason(reason: str) -> str:
    text = str(reason or "").strip()
    for label, pattern in FIRST_REASON_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return "其他"

def clean_return_df(raw: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    order_id_col = get_first_existing_column(raw, ORDER_ID_KEYS)
    sku_col = get_first_existing_column(raw, SKU_ID_KEYS)
    if not order_id_col or not sku_col:
        raise ValueError("售后退货表缺少必要字段：Order ID / SKU ID")
    df = pd.DataFrame(index=raw.index)
    store_col = get_first_existing_column(raw, STORE_KEYS)
    if store_col:
        df["store"] = raw[store_col].astype(str).str.strip().replace({"": "未识别店铺", "nan": "未识别店铺", "None": "未识别店铺", "<NA>": "未识别店铺"})
    else:
        inferred_store = infer_store_from_filename(source_name)
        if str(inferred_store).strip().lower() in {"return", "return report", "return-report", "return_report", "report", "order", "order reports", "order-reports"}:
            inferred_store = "未识别店铺"
        df["store"] = inferred_store
    df["return_id"] = _text_series(raw, RETURN_ID_KEYS)
    df["order_id"] = raw[order_id_col].astype(str).str.strip()
    df["return_status_raw"] = _text_series(raw, RETURN_STATUS_KEYS)
    df["return_status"] = df["return_status_raw"].str.casefold().map(STATUS_MAP).fillna(df["return_status_raw"].replace("", "未知"))
    df["sku_id"] = raw[sku_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["reason_for_request"] = _text_series(raw, REASON_KEYS)
    df["return_quantity"] = parse_numeric_series(_pick(raw, RETURN_QTY_KEYS))
    df["amount_request_to_refund"] = parse_numeric_series(_pick(raw, AMOUNT_REQUEST_KEYS))
    df["amount_refund_to_buyer"] = parse_numeric_series(_pick(raw, AMOUNT_REFUND_KEYS))
    df["order_date"] = parse_return_datetime(_pick(raw, ORDER_DATE_KEYS))
    df["requested_date"] = parse_return_datetime(_pick(raw, REQUESTED_DATE_KEYS))
    df["courier"] = _text_series(raw, COURIER_KEYS)
    df["tracking_number"] = _text_series(raw, TRACKING_KEYS)
    df["service_type_raw"] = _text_series(raw, SERVICE_TYPE_KEYS)
    df["service_type"] = df["service_type_raw"].str.casefold().map(SERVICE_TYPE_MAP).fillna(df["service_type_raw"].replace("", "未知"))
    df["returnless_resolution"] = _text_series(raw, RETURNLESS_KEYS)
    df["cost_responsible"] = _text_series(raw, COST_RESPONSIBLE_KEYS)
    df["first_reason"] = df["reason_for_request"].apply(classify_first_reason)
    df["source_file"] = source_name or ""
    df = df[(df["order_id"] != "") & (df["sku_id"] != "")].copy()
    return df.reset_index(drop=True)

def filter_return_df(return_df: pd.DataFrame, store: str = "全部店铺", sku_ids: Iterable[str] | None = None, statuses: Iterable[str] | None = None, reasons: Iterable[str] | None = None, first_reasons: Iterable[str] | None = None, start_date=None, end_date=None, search_text: str = "") -> pd.DataFrame:
    df = return_df.copy()
    if df.empty:
        return df
    if store and store != "全部店铺":
        df = df[df["store"].astype(str) == str(store)]
    if start_date is not None and end_date is not None:
        df = df[(df["requested_date"].dt.date >= start_date) & (df["requested_date"].dt.date <= end_date)]
    for col, vals in [("sku_id", sku_ids), ("return_status", statuses), ("reason_for_request", reasons), ("first_reason", first_reasons)]:
        vals = [str(x) for x in (vals or []) if str(x).strip() and str(x) != "全部"]
        if vals:
            df = df[df[col].astype(str).isin(vals)]
    keyword = str(search_text or "").strip().casefold()
    if keyword:
        mask = pd.Series(False, index=df.index)
        for col in ["return_id", "order_id", "sku_id", "reason_for_request", "tracking_number"]:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.casefold().str.contains(re.escape(keyword), na=False)
        df = df[mask]
    return df.copy()

def _denominator_sales(detail_df: pd.DataFrame, return_scope: pd.DataFrame, start_date, end_date, store: str) -> tuple[float, float, bool]:
    if detail_df is None or detail_df.empty:
        return 0.0, 0.0
    base = detail_df.copy()
    if start_date is not None and end_date is not None:
        base = base[(base["date"].dt.date >= start_date) & (base["date"].dt.date <= end_date)]
    if store and store != "全部店铺" and "store" in base.columns:
        base = base[base["store"].astype(str) == str(store)]
    if not return_scope.empty and "display_sku" in base.columns:
        selected_skus = set(return_scope["sku_id"].astype(str).dropna().unique().tolist())
        matched = base[base["display_sku"].astype(str).isin(selected_skus)]
        if not matched.empty:
            base = matched
    signed_orders_col = "signed_orders" if "signed_orders" in base.columns else "orders"
    signed_units_col = "signed_units_ordered" if "signed_units_ordered" in base.columns else "units_ordered"
    orders = float(pd.to_numeric(base.get(signed_orders_col, 0), errors="coerce").fillna(0).sum())
    units = float(pd.to_numeric(base.get(signed_units_col, 0), errors="coerce").fillna(0).sum())
    signed_status_available = bool(pd.to_numeric(base.get("status_available_count", 0), errors="coerce").fillna(0).sum() > 0) if not base.empty else False
    return orders, units, signed_status_available

def build_return_summary(return_scope: pd.DataFrame, detail_df: pd.DataFrame, start_date=None, end_date=None, store: str = "全部店铺") -> dict:
    sales_orders, sales_units, signed_status_available = _denominator_sales(detail_df, return_scope, start_date, end_date, store)
    if return_scope.empty:
        return {"refund_amount": 0.0, "return_order_count": 0, "return_unit_count": 0.0, "sales_order_count": sales_orders, "sales_unit_count": sales_units, "signed_status_available": signed_status_available, "order_return_rate": 0.0, "unit_return_rate": 0.0}
    effective = return_scope[return_scope["return_status"] != "已拒绝"].copy()
    refund_amount = float(pd.to_numeric(return_scope["amount_refund_to_buyer"], errors="coerce").fillna(0).sum())
    return_order_count = int(effective["order_id"].nunique())
    return_unit_count = float(pd.to_numeric(effective["return_quantity"], errors="coerce").fillna(0).sum())
    return {"refund_amount": refund_amount, "return_order_count": return_order_count, "return_unit_count": return_unit_count, "sales_order_count": sales_orders, "sales_unit_count": sales_units, "signed_status_available": signed_status_available, "order_return_rate": return_order_count / sales_orders if sales_orders else 0.0, "unit_return_rate": return_unit_count / sales_units if sales_units else 0.0}

def format_return_display_df(return_scope: pd.DataFrame, summary: dict) -> pd.DataFrame:
    if return_scope.empty:
        return pd.DataFrame(columns=list(COLUMN_RENAME.values()))
    show = return_scope.copy()
    show["order_return_rate"] = summary.get("order_return_rate", 0.0)
    show["unit_return_rate"] = summary.get("unit_return_rate", 0.0)
    show["refund_amount_metric"] = summary.get("refund_amount", 0.0)
    show["return_order_count_metric"] = summary.get("return_order_count", 0)
    show["return_unit_count_metric"] = summary.get("return_unit_count", 0.0)
    cols = ["store", "return_id", "order_id", "return_status", "sku_id", "reason_for_request", "return_quantity", "amount_request_to_refund", "amount_refund_to_buyer", "order_date", "requested_date", "service_type", "first_reason", "order_return_rate", "unit_return_rate", "refund_amount_metric", "return_order_count_metric", "return_unit_count_metric"]
    show = show[[c for c in cols if c in show.columns]].rename(columns=COLUMN_RENAME)
    for date_col in ["下单日期", "申请日期"]:
        if date_col in show.columns:
            show[date_col] = pd.to_datetime(show[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    for money_col in ["申请退款金额", "退还给买家的金额", "退款金额"]:
        if money_col in show.columns:
            show[money_col] = pd.to_numeric(show[money_col], errors="coerce").fillna(0).map(lambda x: f"MX${x:,.2f}")
    for qty_col in ["退货数量", "退货退款件数"]:
        if qty_col in show.columns:
            show[qty_col] = pd.to_numeric(show[qty_col], errors="coerce").fillna(0).map(lambda x: f"{x:,.0f}")
    for pct_col in ["订单维度 退货率", "产品件数 退货率"]:
        if pct_col in show.columns:
            show[pct_col] = pd.to_numeric(show[pct_col], errors="coerce").fillna(0).map(lambda x: f"{x * 100:.2f}%")
    return show

def build_return_export_file(display_df: pd.DataFrame) -> bytes:
    desc_df = pd.DataFrame({
        "列组": ["基础信息", "基础信息", "基础信息", "基础信息", "基础信息", "原因指标", "数据信息", "基础信息", "基础信息", "基础信息", "基础信息", "基础信息", "原因指标", "退货指标", "退货指标", "退货指标", "退货指标", "退货指标"],
        "字段": ["店铺", "售后单ID", "订单ID", "售后单状态", "SKU ID", "申请理由", "退货数量", "申请退款金额", "退还给买家的金额", "下单日期", "申请日期", "售后类型", "一级原因", "订单维度 退货率", "产品件数 退货率", "退款金额", "退货退款订单数", "退货退款件数"],
        "说明": ["根据导入文件名推断，可用于多店铺合并筛选", "售后/退货单唯一ID", "原订单ID", "平台退货/退款状态", "售后报表中的 SKU ID", "用户提交的售后申请理由", "售后报表 Return quantity", "售后申请退款金额", "实际退还给买家的金额", "原订单下单日期", "售后申请日期", "售后服务类型", "按规则归类的一级原因", "退货退款订单数 / 总订单数", "退货退款件数 / 总销售件数", "退还给买家的金额总和", "筛选范围内非拒绝售后订单去重计数", "筛选范围内非拒绝售后退货数量求和"],
        "格式要求": ["默认", "默认", "默认", "默认", "默认", "默认", "整数", "MX$，保留2位小数", "MX$，保留2位小数", "YYYY-MM-DD", "YYYY-MM-DD", "默认", "默认", "%百分比，保留2位小数", "%百分比，保留2位小数", "MX$，保留2位小数", "整数", "整数"],
    })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        display_df.to_excel(writer, index=False, sheet_name="售后退货统计")
        desc_df.to_excel(writer, index=False, sheet_name="字段说明")
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "align": "center", "valign": "vcenter"})
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        for sheet_name, df in [("售后退货统计", display_df), ("字段说明", desc_df)]:
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                ws.write(0, idx, col, header_fmt)
                # Robust width calculation: exact search/export may leave some columns all-null,
                # and duplicate column labels can make df[col] a DataFrame. Avoid int(NaN).
                col_data = df.iloc[:, idx] if idx < len(df.columns) else pd.Series(dtype="object")
                if df.empty:
                    content_width = 12
                else:
                    lengths = col_data.astype(str).str.len()
                    q90 = pd.to_numeric(lengths, errors="coerce").dropna().quantile(0.9)
                    content_width = int(q90) if pd.notna(q90) else 12
                width = max(12, min(34, max(len(str(col)) + 2, content_width)))
                ws.set_column(idx, idx, width, wrap_fmt if col in {"申请理由", "说明"} else None)
            ws.autofilter(0, 0, len(df), max(len(df.columns) - 1, 0))
            ws.freeze_panes(1, 0)
    return output.getvalue()
