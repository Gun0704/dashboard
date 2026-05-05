from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from src.charts import make_clicks_conversion_chart, make_frontend_price_sales_chart, make_impressions_ctr_chart
from src.cleaners import (
    clean_mapping_df,
    clean_sales_df,
    clean_traffic_df,
    is_sales_file,
    is_traffic_file,
    read_uploaded_table,
    parse_numeric_series,
)
from src.config import DEFAULT_CONVERSION_BASIS, DEFAULT_INDUSTRY_CONVERSION, DEFAULT_INDUSTRY_CTR, QUICK_TAG_OPTIONS
from src.matching import build_detail_dataset, build_match_check
from src.metrics import build_alerts, build_analysis_text, build_daily_dataset, build_daily_detail_dataset, build_sku_tables, build_tag_snapshot, safe_divide
from src.ui_helpers import build_export_file, format_pct, pick_existing_columns, style_daily_detail_table

st.set_page_config(page_title="营销数据看板", layout="wide")
st.title("营销数据看板")

@st.cache_data(show_spinner=False)
def load_uploaded_table(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    suffix = file_name.lower().split(".")[-1]
    bio = BytesIO(file_bytes)
    if suffix == "csv":
        last_error = None
        for encoding in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
            try:
                bio.seek(0)
                return pd.read_csv(bio, encoding=encoding)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"CSV 读取失败：{last_error}")
    return pd.read_excel(bio)


@st.cache_data(show_spinner=False)
def process_inputs(
    sales_inputs: list[tuple[str, bytes]],
    traffic_inputs: list[tuple[str, bytes]],
    mapping_inputs: list[tuple[str, bytes]],
    mixed_inputs: list[tuple[str, bytes]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    sales_df = pd.DataFrame()
    traffic_df = pd.DataFrame()
    mapping_df = pd.DataFrame()
    messages: list[str] = []
    unknown_files: list[str] = []
    sales_parts, traffic_parts, mapping_parts = [], [], []

    if sales_inputs:
        for name, content in sales_inputs:
            sales_parts.append(clean_sales_df(load_uploaded_table(name, content), name))
        sales_df = pd.concat(sales_parts, ignore_index=True) if sales_parts else pd.DataFrame()
        messages.append(f"销售表载入 {len(sales_inputs)} 个文件，共 {len(sales_df):,} 行")
    if traffic_inputs:
        for name, content in traffic_inputs:
            traffic_parts.append(clean_traffic_df(load_uploaded_table(name, content), name))
        traffic_df = pd.concat(traffic_parts, ignore_index=True) if traffic_parts else pd.DataFrame()
        messages.append(f"流量表载入 {len(traffic_inputs)} 个文件，共 {len(traffic_df):,} 行")
    if mapping_inputs:
        for name, content in mapping_inputs:
            mapping_parts.append(clean_mapping_df(load_uploaded_table(name, content), name))
        mapping_df = pd.concat(mapping_parts, ignore_index=True) if mapping_parts else pd.DataFrame()
        if not mapping_df.empty:
            mapping_df = mapping_df.sort_values(["goods_id", "inventory_qty"], ascending=[True, False]).drop_duplicates("goods_id")
        messages.append(f"商品信息表载入 {len(mapping_inputs)} 个文件，共 {len(mapping_df):,} 个 Goods ID")
    if mixed_inputs:
        mixed_sales_parts, mixed_traffic_parts = [], []
        for name, content in mixed_inputs:
            raw_df = load_uploaded_table(name, content)
            if is_sales_file(raw_df):
                mixed_sales_parts.append(clean_sales_df(raw_df, name))
            elif is_traffic_file(raw_df):
                mixed_traffic_parts.append(clean_traffic_df(raw_df, name))
            else:
                unknown_files.append(name)
        if mixed_sales_parts:
            mixed_sales_df = pd.concat(mixed_sales_parts, ignore_index=True)
            sales_df = pd.concat([sales_df, mixed_sales_df], ignore_index=True) if not sales_df.empty else mixed_sales_df
        if mixed_traffic_parts:
            mixed_traffic_df = pd.concat(mixed_traffic_parts, ignore_index=True)
            traffic_df = pd.concat([traffic_df, mixed_traffic_df], ignore_index=True) if not traffic_df.empty else mixed_traffic_df
        messages.append(f"混合上传自动识别：销售文件 {len(mixed_sales_parts)} 个，流量文件 {len(mixed_traffic_parts)} 个")

    return sales_df, traffic_df, mapping_df, messages, unknown_files


@st.cache_data(show_spinner=False)
def compute_base_datasets(
    sales_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    conversion_basis: str,
) -> tuple[pd.DataFrame, dict, dict]:
    detail_df = build_detail_dataset(sales_df, traffic_df, mapping_df, conversion_basis)
    match_summary, match_details = build_match_check(sales_df, traffic_df, mapping_df)
    return detail_df, match_summary, match_details


@st.cache_data(show_spinner=False)
def load_frontend_order_df(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    df = load_uploaded_table(file_name, file_bytes)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required_cols = ["purchase date", "retail price (tax excl.)", "quantity purchased"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"前端价订单表缺少字段：{', '.join(missing)}")

    cleaned = df.copy()
    purchase_text = (
        cleaned["purchase date"].astype(str).str.strip()
        .str.replace(r"\s+[A-Z]{2,5}\(UTC[+-]\d+\)$", "", regex=True)
        .str.replace(r"\s+UTC[+-]?\d+$", "", regex=True)
    )
    purchase_dt = pd.to_datetime(purchase_text, errors="coerce")
    if purchase_dt.isna().any():
        fallback_dt = pd.to_datetime(purchase_text, format="%b %d, %Y, %I:%M %p", errors="coerce")
        purchase_dt = purchase_dt.fillna(fallback_dt)
    cleaned["purchase_dt"] = purchase_dt
    cleaned["date"] = cleaned["purchase_dt"].dt.normalize()
    cleaned["retail price (tax excl.)"] = parse_numeric_series(cleaned["retail price (tax excl.)"])
    cleaned["quantity purchased"] = parse_numeric_series(cleaned["quantity purchased"])

    for status_col in ["order status", "order item status"]:
        if status_col in cleaned.columns:
            cleaned[status_col] = cleaned[status_col].astype(str).str.strip()
    if "contribution sku" in cleaned.columns:
        cleaned["contribution sku"] = cleaned["contribution sku"].astype(str).str.strip()
        cleaned["contribution sku"] = cleaned["contribution sku"].replace({"nan": "", "None": ""})

    invalid_status_mask = pd.Series(False, index=cleaned.index)
    for status_col in ["order status", "order item status"]:
        if status_col in cleaned.columns:
            invalid_status_mask = invalid_status_mask | cleaned[status_col].str.contains(
                r"cancel|closed|void", case=False, na=False
            )

    cleaned = cleaned.dropna(subset=["date"]).copy()
    cleaned = cleaned[cleaned["retail price (tax excl.)"] > 0].copy()
    cleaned = cleaned[cleaned["quantity purchased"] > 0].copy()
    cleaned = cleaned[~invalid_status_mask.loc[cleaned.index]].copy()

    if cleaned.empty:
        raise ValueError("前端价订单表清洗后无有效数据")
    return cleaned


@st.cache_data(show_spinner=False)
def build_frontend_daily_dataset(order_df: pd.DataFrame) -> pd.DataFrame:
    def _agg(group: pd.DataFrame) -> pd.Series:
        total_qty = pd.to_numeric(group["quantity purchased"], errors="coerce").fillna(0).sum()
        total_retail = pd.to_numeric(group["retail price (tax excl.)"], errors="coerce").fillna(0).sum()
        unit_price_tax_excl = total_retail / total_qty if total_qty else 0
        return pd.Series({
            "frontend_price": unit_price_tax_excl * 1.16,
            "frontend_price_tax_excl": unit_price_tax_excl,
            "quantity_purchased": total_qty,
            "retail_price_tax_excl_total": total_retail,
        })

    daily = order_df.groupby("date").apply(_agg).reset_index()
    return daily.sort_values("date")


def normalize_text_key(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    return (
        series.astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .fillna("")
    )


@st.cache_data(show_spinner=False)
def enrich_frontend_order_df(
    frontend_order_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if frontend_order_df.empty:
        return frontend_order_df.copy(), {"total_rows": 0, "mapped_rows": 0, "mapped_ratio": 0.0, "sku_hit_rows": 0, "name_hit_rows": 0}

    enriched = frontend_order_df.copy()
    enriched["product_name_key"] = normalize_text_key(enriched.get("product name"))
    enriched["frontend_sku"] = normalize_text_key(enriched.get("contribution sku"))
    enriched["variation_key"] = normalize_text_key(enriched.get("variation"))

    name_parts: list[pd.DataFrame] = []
    if not sales_df.empty and {"product_name", "goods_id"}.issubset(sales_df.columns):
        name_parts.append(sales_df[["product_name", "goods_id"]].copy())
    if not traffic_df.empty and {"product_name", "goods_id"}.issubset(traffic_df.columns):
        name_parts.append(traffic_df[["product_name", "goods_id"]].copy())
    if not mapping_df.empty and {"product_name", "goods_id"}.issubset(mapping_df.columns):
        name_parts.append(mapping_df[["product_name", "goods_id"]].copy())

    if name_parts:
        name_map = pd.concat(name_parts, ignore_index=True)
        name_map["product_name"] = normalize_text_key(name_map["product_name"])
        name_map["goods_id"] = normalize_text_key(name_map["goods_id"])
        name_map = name_map[(name_map["product_name"] != "") & (name_map["goods_id"] != "")].drop_duplicates()
        name_map = name_map.drop_duplicates(subset=["product_name"], keep="first")
        enriched = enriched.merge(name_map.rename(columns={"product_name": "product_name_key", "goods_id": "goods_id_by_name"}), on="product_name_key", how="left")
    else:
        enriched["goods_id_by_name"] = ""

    if not mapping_df.empty and {"sku", "goods_id"}.issubset(mapping_df.columns):
        sku_map = mapping_df[["sku", "goods_id"]].copy()
        sku_map["sku"] = normalize_text_key(sku_map["sku"])
        sku_map["goods_id"] = normalize_text_key(sku_map["goods_id"])
        sku_map = sku_map[(sku_map["sku"] != "") & (sku_map["goods_id"] != "")].drop_duplicates()
        sku_map = sku_map.drop_duplicates(subset=["sku"], keep="first")
        enriched = enriched.merge(sku_map.rename(columns={"sku": "frontend_sku", "goods_id": "goods_id_by_sku"}), on="frontend_sku", how="left")
    else:
        enriched["goods_id_by_sku"] = ""

    enriched["goods_id"] = normalize_text_key(enriched.get("goods_id_by_sku")).where(
        normalize_text_key(enriched.get("goods_id_by_sku")) != "",
        normalize_text_key(enriched.get("goods_id_by_name")),
    )
    enriched["display_sku"] = enriched["frontend_sku"].where(enriched["frontend_sku"] != "", enriched["goods_id"])

    stats = {
        "total_rows": int(len(enriched)),
        "mapped_rows": int((enriched["goods_id"] != "").sum()),
        "mapped_ratio": float((enriched["goods_id"] != "").mean()) if len(enriched) else 0.0,
        "sku_hit_rows": int((normalize_text_key(enriched.get("goods_id_by_sku")) != "").sum()),
        "name_hit_rows": int((normalize_text_key(enriched.get("goods_id_by_name")) != "").sum()),
    }
    return enriched, stats

def extract_goods_ids(raw_text: str) -> list[str]:
    if not raw_text:
        return []
    candidates = re.findall(r"\d{6,}", str(raw_text))
    seen = set()
    ordered = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def render_metric_card(title: str, value: str, group_class: str, delta_text: str | None = None, good: bool | None = None, tooltip_html: str | None = None) -> None:
    delta_html = ''
    if delta_text:
        status_class = 'neutral'
        if good is True:
            status_class = 'good'
        elif good is False:
            status_class = 'bad'
        delta_html = f'<div class="metric-delta {status_class}">{delta_text}</div>'
    tooltip_block = ''
    if tooltip_html:
        tooltip_block = f'<div class="metric-tooltip">{tooltip_html}</div>'
    html = (
        f'<div class="metric-card {group_class}">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{delta_html}'
        f'{tooltip_block}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


st.markdown(
    """
<style>
.metric-card {
    position: relative;
    border-radius: 16px;
    padding: 16px 18px;
    min-height: 132px;
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    background: #ffffff;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10); }
.metric-card.flow { background: linear-gradient(180deg, #eef6ff 0%, #ffffff 100%); }
.metric-card.conv { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%); }
.metric-card.sales { background: linear-gradient(180deg, #eefbf3 0%, #ffffff 100%); }
.metric-title {
    color: #334155;
    font-size: 0.96rem;
    margin-bottom: 14px;
    font-weight: 600;
}
.metric-value {
    color: #0f172a;
    font-size: 1.95rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 10px;
}
.metric-delta {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 600;
}
.metric-delta.good { color: #15803d; background: #dcfce7; }
.metric-delta.bad { color: #b91c1c; background: #fee2e2; }
.metric-delta.neutral { color: #475569; background: #e2e8f0; }
.goods-id-help { font-size: 0.85rem; color: #64748b; margin-top: 6px; }
.metric-tooltip {
    display: none;
    position: absolute;
    left: 12px;
    right: 12px;
    bottom: calc(100% + 10px);
    z-index: 999;
    padding: 12px 14px;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.96);
    color: #f8fafc;
    font-size: 0.82rem;
    line-height: 1.5;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.24);
}
.metric-tooltip b { color: #ffffff; }
.metric-card:hover .metric-tooltip { display: block; }

</style>
""",
    unsafe_allow_html=True,
)
st.caption("基于原项目按需求文档升级：Goods ID 作为匹配键，SKU 作为显示标准；支持批量上传多店铺、多日期销售/流量/商品信息文件。")

with st.sidebar:
    st.header("数据上传")
    sales_files = st.file_uploader("批量上传销售表", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    traffic_files = st.file_uploader("批量上传流量表", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    mapping_files = st.file_uploader("批量上传商品信息表 / SKU映射表", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    frontend_order_file = st.file_uploader("上传前端价订单导出表", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
    all_data_files = st.file_uploader("或一次性混合上传（自动识别销售/流量）", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    conversion_basis = st.selectbox("订单口径", ["订单商品数", "买家数", "下单件数"], index=["订单商品数", "买家数", "下单件数"].index(DEFAULT_CONVERSION_BASIS))
    industry_ctr = st.slider("CTR参考值", min_value=0.0, max_value=0.30, value=DEFAULT_INDUSTRY_CTR, step=0.005, format="%.3f")
    industry_conversion = st.slider("转化率参考值", min_value=0.0, max_value=0.50, value=DEFAULT_INDUSTRY_CONVERSION, step=0.005, format="%.3f")
    st.markdown("---")
    st.markdown("**说明**")
    st.write("- 支持多店铺、多日期批量上传")
    st.write("- 自动按字段识别销售表和流量表")
    st.write("- Goods ID 统一匹配，SKU 优先展示，未匹配回退 Goods ID")

if not sales_files and not traffic_files and not all_data_files:
    st.info("请先上传销售表和流量表；商品信息表可选但建议上传。")
    st.stop()

def _collect_inputs(files) -> list[tuple[str, bytes]]:
    return [(f.name, f.getvalue()) for f in files] if files else []

sales_inputs = _collect_inputs(sales_files)
traffic_inputs = _collect_inputs(traffic_files)
mapping_inputs = _collect_inputs(mapping_files)
mixed_inputs = _collect_inputs(all_data_files)
frontend_order_input = (frontend_order_file.name, frontend_order_file.getvalue()) if frontend_order_file else None

try:
    sales_df, traffic_df, mapping_df, messages, unknown_files = process_inputs(
        sales_inputs, traffic_inputs, mapping_inputs, mixed_inputs
    )
    if unknown_files:
        st.warning("以下文件未识别：" + "、".join(unknown_files[:10]) + (" ..." if len(unknown_files) > 10 else ""))
except Exception as exc:
    st.error(f"数据读取或清洗失败：{exc}")
    st.stop()

for msg in messages:
    st.success(msg)

frontend_order_df = pd.DataFrame()
frontend_match_stats = {"total_rows": 0, "mapped_rows": 0, "mapped_ratio": 0.0, "sku_hit_rows": 0, "name_hit_rows": 0}
if frontend_order_input:
    try:
        frontend_order_df = load_frontend_order_df(frontend_order_input[0], frontend_order_input[1])
        frontend_order_df, frontend_match_stats = enrich_frontend_order_df(frontend_order_df, sales_df, traffic_df, mapping_df)
        st.success(f"前端价订单表载入成功（支持 CSV / Excel），清洗后共 {len(frontend_order_df):,} 行")
        st.caption(f"前端订单映射成功率：{frontend_match_stats['mapped_ratio']:.1%}（SKU直连 {frontend_match_stats['sku_hit_rows']:,} 行，商品名映射 {frontend_match_stats['name_hit_rows']:,} 行）")
    except Exception as exc:
        st.warning(f"前端价订单表读取失败，本次将不展示前端价格销量走势图：{exc}")
        frontend_order_df = pd.DataFrame()

try:
    detail_df, match_summary, match_details = compute_base_datasets(sales_df, traffic_df, mapping_df, conversion_basis)
except Exception as exc:
    st.error(f"数据合并失败：{exc}")
    st.stop()

if detail_df.empty:
    st.warning("清洗后没有可用数据，请检查日期、Goods ID 或文件类型。")
    st.stop()

min_date = detail_df["date"].min().date()
max_date = detail_df["date"].max().date()
default_start = max(min_date, (detail_df["date"].max() - pd.Timedelta(days=6)).date())

st.markdown("## 模块 1：筛选与核心指标区")
with st.container(border=True):
    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns([1.15, 1.05, 1.8, 1.0, 1.1])
    with filter_col1:
        valid_stores = sorted({s for s in detail_df["store"].dropna().astype(str).tolist() if s.strip() and s.strip().casefold() not in {"0", "0.0", "nan", "none", "null", "<na>", "未分类店铺"}})
        store_options = ["全部店铺"] + valid_stores
        selected_store = st.selectbox("店铺", store_options, index=0)
    with filter_col2:
        product_mode = st.selectbox("产品筛选", ["全部产品", "按 Goods ID", "按 SKU"], index=0)
    with filter_col3:
        base_df = detail_df if selected_store == "全部店铺" else detail_df[detail_df["store"] == selected_store]
        product_tag_df = build_tag_snapshot(base_df) if not base_df.empty else pd.DataFrame()
        tag_priority = {"大爆款": 1, "爆款": 2, "旺款": 3, "上升趋势品": 4, "新品": 5, "常规款": 6, "滞制品": 7}

        def _pick_main_tag(tag_text: str) -> str:
            raw = str(tag_text or "").strip()
            if not raw:
                return "未分类"
            candidates = []
            if "大爆款" in raw:
                candidates.append("大爆款")
            if "爆款" in raw and "大爆款" not in raw:
                candidates.append("爆款")
            if "旺款" in raw:
                candidates.append("旺款")
            if "上升趋势品" in raw or "↑" in raw:
                candidates.append("上升趋势品")
            if "新品" in raw:
                candidates.append("新品")
            if "常规款" in raw:
                candidates.append("常规款")
            if "滞制品" in raw:
                candidates.append("滞制品")
            return sorted(set(candidates), key=lambda x: tag_priority.get(x, 99))[0] if candidates else raw.replace("↑", "").strip() or "未分类"

        goods_id_options = sorted(base_df["goods_id"].dropna().astype(str).loc[lambda s: s.str.strip() != ""].unique().tolist())
        frontend_sku_goods_df = pd.DataFrame(columns=["display_sku", "goods_id"])
        if not frontend_order_df.empty and {"display_sku", "goods_id"}.issubset(frontend_order_df.columns):
            frontend_sku_goods_df = frontend_order_df[["display_sku", "goods_id"]].copy()
            frontend_sku_goods_df["display_sku"] = frontend_sku_goods_df["display_sku"].astype(str).str.strip()
            frontend_sku_goods_df["goods_id"] = frontend_sku_goods_df["goods_id"].astype(str).str.strip()
            frontend_sku_goods_df = frontend_sku_goods_df[(frontend_sku_goods_df["display_sku"] != "") & (frontend_sku_goods_df["goods_id"] != "")]
            if selected_store != "全部店铺":
                base_goods_ids = set(base_df["goods_id"].astype(str).unique().tolist())
                frontend_sku_goods_df = frontend_sku_goods_df[frontend_sku_goods_df["goods_id"].isin(base_goods_ids)]
            frontend_sku_goods_df = frontend_sku_goods_df.drop_duplicates()

        fallback_sku_goods_df = base_df[["display_sku", "goods_id"]].copy()
        fallback_sku_goods_df["display_sku"] = fallback_sku_goods_df["display_sku"].astype(str).str.strip()
        fallback_sku_goods_df["goods_id"] = fallback_sku_goods_df["goods_id"].astype(str).str.strip()
        fallback_sku_goods_df = fallback_sku_goods_df[(fallback_sku_goods_df["display_sku"] != "") & (fallback_sku_goods_df["goods_id"] != "")].drop_duplicates()

        sku_goods_df = frontend_sku_goods_df if not frontend_sku_goods_df.empty else fallback_sku_goods_df
        sku_options_all = sorted(sku_goods_df["display_sku"].astype(str).unique().tolist()) if not sku_goods_df.empty else []

        product_tag_df = product_tag_df.copy() if not product_tag_df.empty else pd.DataFrame()
        if not product_tag_df.empty:
            product_tag_df["main_tag"] = product_tag_df["tag_short"].apply(_pick_main_tag)
        goods_tag_map = product_tag_df.drop_duplicates("goods_id").set_index("goods_id")["main_tag"].to_dict() if not product_tag_df.empty else {}
        sku_goods_map = sku_goods_df.drop_duplicates(subset=["display_sku"], keep="first").set_index("display_sku")["goods_id"].to_dict() if not sku_goods_df.empty else {}

        if product_mode == "按 Goods ID":
            goods_id_label_map = {key: f"{key} ｜ {goods_tag_map.get(key, '未分类')}" for key in goods_id_options}
            selected_goods_id = st.selectbox(
                "选择 Goods ID",
                ["全部Goods ID"] + goods_id_options,
                index=0,
                format_func=lambda x: x if x == "全部Goods ID" else goods_id_label_map.get(x, x),
            )
            selected_sku = "全部SKU"
        elif product_mode == "按 SKU":
            sub_col1, sub_col2 = st.columns(2)
            goods_id_label_map = {key: f"{key} ｜ {goods_tag_map.get(key, '未分类')}" for key in goods_id_options}
            with sub_col1:
                selected_goods_id = st.selectbox(
                    "联动 Goods ID（可选）",
                    ["全部Goods ID"] + goods_id_options,
                    index=0,
                    format_func=lambda x: x if x == "全部Goods ID" else goods_id_label_map.get(x, x),
                )
            if selected_goods_id != "全部Goods ID" and not sku_goods_df.empty:
                linked_sku_options = sorted(
                    sku_goods_df.loc[sku_goods_df["goods_id"] == str(selected_goods_id), "display_sku"].astype(str).unique().tolist()
                )
            else:
                linked_sku_options = sku_options_all
            sku_label_map = {
                key: f"{key} ｜ {goods_tag_map.get(sku_goods_map.get(key, ''), '未分类')} ｜ Goods {sku_goods_map.get(key, '-') }"
                for key in linked_sku_options
            }
            with sub_col2:
                selected_sku = st.selectbox(
                    "选择 SKU",
                    ["全部SKU"] + linked_sku_options,
                    index=0,
                    format_func=lambda x: x if x == "全部SKU" else sku_label_map.get(x, x),
                )
        else:
            st.selectbox("选择 Goods ID / SKU", ["全部产品"], index=0, disabled=True)
            selected_goods_id = "全部Goods ID"
            selected_sku = "全部SKU"
    with filter_col4:
        selected_dates = st.date_input("日期范围", value=(default_start, max_date), min_value=min_date, max_value=max_date)
    quick_tag_options = ["全部标签", "核心爆款", "上升趋势品", "滞制品", "大爆款", "爆款", "旺款", "常规款", "新品"]
    with filter_col5:
        selected_tag = st.selectbox("快捷标签", quick_tag_options, index=0)

    extra_col1, extra_col2, extra_col3 = st.columns([1.9, 0.55, 2.0])
    if "goods_id_bulk_input" not in st.session_state:
        st.session_state["goods_id_bulk_input"] = ""

    def clear_goods_id_bulk_input():
        st.session_state["goods_id_bulk_input"] = ""

    def fill_goods_id_bulk_example():
        st.session_state["goods_id_bulk_input"] = "商品平台活动信息\n商品 ID\n603182235263112\n602642076000852\n602642076000852"

    with extra_col1:
        goods_id_input = st.text_area(
            "Goods ID 批量搜索",
            key="goods_id_bulk_input",
            height=108,
            placeholder="支持直接粘贴原始内容；自动提取数字型 Goods ID，忽略标题文字。",
        ).strip()
        st.markdown('<div class="goods-id-help">支持换行、逗号、空格、分号，复制表格原文也可自动提取。</div>', unsafe_allow_html=True)
    with extra_col2:
        st.write("")
        st.write("")
        st.button("清空输入", use_container_width=True, on_click=clear_goods_id_bulk_input)
        st.button("粘贴示例", use_container_width=True, on_click=fill_goods_id_bulk_example)
    with extra_col3:
        st.caption("单品筛选支持按 Goods ID 或按 SKU 两种模式。SKU 模式新增 Goods ID 联动下拉：可先选 Goods ID，再只看该 Goods 下的 SKU；价格销量走势图按真实 SKU 过滤。")

raw_goods_ids = extract_goods_ids(goods_id_input)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = min_date, max_date

filtered_df = detail_df[(detail_df["date"].dt.date >= start_date) & (detail_df["date"].dt.date <= end_date)].copy()
if selected_store != "全部店铺":
    filtered_df = filtered_df[filtered_df["store"] == selected_store]
selected_product = "全部产品"
if product_mode == "按 Goods ID":
    selected_product = selected_goods_id
    if selected_goods_id != "全部Goods ID":
        filtered_df = filtered_df[filtered_df["goods_id"].astype(str) == str(selected_goods_id)]
elif product_mode == "按 SKU":
    selected_product = selected_sku
    selected_goods_ids: list[str] = []
    if selected_goods_id != "全部Goods ID":
        selected_goods_ids = [str(selected_goods_id)]
        filtered_df = filtered_df[filtered_df["goods_id"].astype(str) == str(selected_goods_id)]
    elif not frontend_order_df.empty and {"display_sku", "goods_id"}.issubset(frontend_order_df.columns) and selected_sku != "全部SKU":
        selected_goods_ids = frontend_order_df.loc[
            frontend_order_df["display_sku"].astype(str) == str(selected_sku),
            "goods_id",
        ].astype(str).dropna().unique().tolist()
        if selected_goods_ids:
            filtered_df = filtered_df[filtered_df["goods_id"].astype(str).isin(selected_goods_ids)]
        else:
            filtered_df = filtered_df.iloc[0:0].copy()
    if selected_sku != "全部SKU":
        if selected_goods_ids:
            sku_filtered_df = filtered_df[filtered_df["display_sku"].astype(str) == str(selected_sku)]
            if not sku_filtered_df.empty:
                filtered_df = sku_filtered_df
        elif frontend_order_df.empty:
            filtered_df = filtered_df[filtered_df["display_sku"].astype(str) == str(selected_sku)]

match_scope_df = filtered_df.copy()
available_goods_ids = set(match_scope_df["goods_id"].astype(str).dropna().tolist())
matched_goods_ids = [gid for gid in raw_goods_ids if gid in available_goods_ids]
unmatched_goods_ids = [gid for gid in raw_goods_ids if gid not in available_goods_ids]
if matched_goods_ids:
    filtered_df = filtered_df[filtered_df["goods_id"].astype(str).isin(matched_goods_ids)]
elif raw_goods_ids:
    filtered_df = filtered_df.iloc[0:0].copy()

tag_snapshot = build_tag_snapshot(filtered_df if not filtered_df.empty else detail_df)
if selected_tag != "全部标签" and not filtered_df.empty and not tag_snapshot.empty:
    if selected_tag == "上升趋势品":
        keep_ids = set(tag_snapshot.loc[tag_snapshot["trend_up"], "goods_id"])
    elif selected_tag == "核心爆款":
        keep_ids = set(tag_snapshot.loc[tag_snapshot["core_tag"].isin(["大爆款", "爆款", "旺款"]), "goods_id"])
    else:
        keep_ids = set(tag_snapshot.loc[tag_snapshot["display_tag"].str.contains(selected_tag, na=False), "goods_id"])
    filtered_df = filtered_df[filtered_df["goods_id"].isin(keep_ids)]

if raw_goods_ids:
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.info(f"已识别 Goods ID：{len(raw_goods_ids)} 个")
    summary_col2.success(f"命中：{len(matched_goods_ids)} 个")
    summary_col3.error(f"未命中：{len(unmatched_goods_ids)} 个")
    if unmatched_goods_ids:
        with st.expander("查看未命中的 Goods ID"):
            st.code("\n".join(unmatched_goods_ids), language="text")

if filtered_df.empty:
    st.warning("当前筛选条件下没有数据。")
    st.stop()

frontend_filtered_df = pd.DataFrame()
frontend_scope_text = "全部商品"
if not frontend_order_df.empty:
    frontend_filtered_df = frontend_order_df[(frontend_order_df["date"].dt.date >= start_date) & (frontend_order_df["date"].dt.date <= end_date)].copy()
    if selected_store != "全部店铺" and "goods_id" in frontend_filtered_df.columns:
        scoped_goods_ids = set(filtered_df["goods_id"].astype(str).unique().tolist())
        frontend_filtered_df = frontend_filtered_df[frontend_filtered_df["goods_id"].astype(str).isin(scoped_goods_ids)]
    if product_mode == "按 Goods ID" and selected_goods_id != "全部Goods ID":
        frontend_scope_text = f"Goods ID：{selected_goods_id}"
        frontend_filtered_df = frontend_filtered_df[frontend_filtered_df["goods_id"].astype(str) == str(selected_goods_id)]
    elif product_mode == "按 SKU":
        if selected_goods_id != "全部Goods ID":
            frontend_filtered_df = frontend_filtered_df[frontend_filtered_df["goods_id"].astype(str) == str(selected_goods_id)]
            frontend_scope_text = f"Goods ID：{selected_goods_id}"
        if selected_sku != "全部SKU":
            frontend_filtered_df = frontend_filtered_df[frontend_filtered_df["display_sku"].astype(str) == str(selected_sku)]
            frontend_scope_text = f"{frontend_scope_text} / SKU：{selected_sku}" if frontend_scope_text != "全部商品" else f"SKU：{selected_sku}"

daily_df = build_daily_dataset(filtered_df)
daily_detail_df = build_daily_detail_dataset(filtered_df)
tag_snapshot = build_tag_snapshot(filtered_df)
top20, abnormal_sku, unmatched_sku = build_sku_tables(filtered_df)
today_alerts, history_alerts, tag_alerts, actions = build_alerts(filtered_df, tag_snapshot, daily_df)

is_single_goods_selected = product_mode == "按 Goods ID" and selected_goods_id != "全部Goods ID"
is_single_sku_selected = product_mode == "按 SKU" and selected_sku != "全部SKU"
show_frontend_chart = is_single_goods_selected or is_single_sku_selected

summary_impressions = filtered_df["impressions"].sum()
summary_clicks = filtered_df["clicks"].sum()
summary_orders = filtered_df["orders"].sum()
summary_sales = filtered_df["sales_amount"].sum()
summary_units = filtered_df["units_ordered"].sum()
summary_ctr = safe_divide(summary_clicks, summary_impressions)
summary_conversion = safe_divide(summary_orders, summary_clicks)
days_count = max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1, 1)
summary_avg_units = summary_units / days_count
summary_avg_sales = summary_sales / days_count
summary_avg_sales_per_order = safe_divide(summary_sales, summary_orders)
summary_avg_units_per_order = safe_divide(summary_units, summary_orders)

recent7_daily = daily_df.sort_values("date").tail(min(7, len(daily_df))).copy()
recent7_days_count = max(len(recent7_daily), 1)
recent7_metrics = {
    "总曝光量": recent7_daily["impressions"].mean() if not recent7_daily.empty else 0,
    "总点击量": recent7_daily["clicks"].mean() if not recent7_daily.empty else 0,
    "整体 CTR": recent7_daily["ctr"].mean() if not recent7_daily.empty else 0,
    "总订单数": recent7_daily["orders"].mean() if not recent7_daily.empty else 0,
    "整体转化率": recent7_daily["conversion_rate"].mean() if not recent7_daily.empty else 0,
    "总销售额": recent7_daily["sales_amount"].mean() if not recent7_daily.empty else 0,
    "总销量": recent7_daily["units_ordered"].mean() if not recent7_daily.empty else 0,
    "每单销售额": recent7_daily["avg_sales_per_order"].mean() if not recent7_daily.empty else 0,
    "每单销售量": recent7_daily["avg_units_per_order"].mean() if not recent7_daily.empty else 0,
    "日均销售额": recent7_daily["sales_amount"].sum() / recent7_days_count if not recent7_daily.empty else 0,
    "日均销量": recent7_daily["units_ordered"].sum() / recent7_days_count if not recent7_daily.empty else 0,
}

single_goods_id = filtered_df["goods_id"].astype(str).nunique() == 1
inventory_qty_single = filtered_df["inventory_qty"].max() if (single_goods_id and "inventory_qty" in filtered_df.columns) else 0
library_sales_ratio = safe_divide(inventory_qty_single, summary_avg_units) if single_goods_id else 0
recent7_metrics["库销比"] = safe_divide(inventory_qty_single, recent7_metrics["日均销量"]) if single_goods_id else 0

def build_metric_tooltip(definition: str, formula: str, recent_value: str) -> str:
    return f"<b>指标定义</b><br>{definition}<br><br><b>计算逻辑</b><br>{formula}<br><br><b>近7天均值</b><br>{recent_value}"

metric_tooltips = {
    "总曝光量": build_metric_tooltip("当前筛选范围内累计曝光次数。", "筛选范围内按天汇总 impressions 后求和。", f"{recent7_metrics['总曝光量']:,.0f} /日"),
    "总点击量": build_metric_tooltip("当前筛选范围内累计点击次数。", "筛选范围内按天汇总 clicks 后求和。", f"{recent7_metrics['总点击量']:,.0f} /日"),
    "整体 CTR": build_metric_tooltip("点击效率，衡量曝光到点击的转化表现。", "总点击量 / 总曝光量 × 100%", format_pct(recent7_metrics['整体 CTR'])),
    "总订单数": build_metric_tooltip("当前筛选范围内累计订单数。", "筛选范围内按天汇总 orders 后求和。", f"{recent7_metrics['总订单数']:,.1f} /日"),
    "整体转化率": build_metric_tooltip("点击到下单的整体转化效率。", "总订单数 / 总点击量 × 100%", format_pct(recent7_metrics['整体转化率'])),
    "总销售额": build_metric_tooltip("当前筛选范围内累计销售额。", "筛选范围内按天汇总 sales_amount 后求和。", f"MX${recent7_metrics['总销售额']:,.2f} /日"),
    "总销量": build_metric_tooltip("当前筛选范围内累计销量。", "筛选范围内按天汇总 units_ordered 后求和。", f"{recent7_metrics['总销量']:,.1f} /日"),
    "每单销售额": build_metric_tooltip("平均每笔订单贡献的销售额。", "总销售额 / 总订单数", f"MX${recent7_metrics['每单销售额']:,.2f}"),
    "每单销售量": build_metric_tooltip("平均每笔订单售出的件数。", "总销量 / 总订单数", f"{recent7_metrics['每单销售量']:,.2f}"),
    "日均销售额": build_metric_tooltip("当前筛选周期内平均每天的销售额。", "总销售额 / 天数", f"MX${recent7_metrics['日均销售额']:,.2f}"),
    "日均销量": build_metric_tooltip("当前筛选周期内平均每天的销量。", "总销量 / 天数", f"{recent7_metrics['日均销量']:,.2f}"),
    "库销比": build_metric_tooltip("单个 Goods ID 当前库存相对日均销量的覆盖倍数，值越高说明库存覆盖天数越长。", "当前库存 / 日均销量", f"{recent7_metrics['库销比']:,.2f}"),
}

metric_row1 = st.columns(5)
with metric_row1[0]:
    render_metric_card("总曝光量", f"{summary_impressions:,.0f}", "flow", tooltip_html=metric_tooltips["总曝光量"])
with metric_row1[1]:
    render_metric_card("总点击量", f"{summary_clicks:,.0f}", "flow", tooltip_html=metric_tooltips["总点击量"])
with metric_row1[2]:
    render_metric_card("整体 CTR", format_pct(summary_ctr), "flow", f"参考 {format_pct(industry_ctr)}", summary_ctr >= industry_ctr, tooltip_html=metric_tooltips["整体 CTR"])
with metric_row1[3]:
    render_metric_card("总订单数", f"{summary_orders:,.0f}", "conv", tooltip_html=metric_tooltips["总订单数"])
with metric_row1[4]:
    render_metric_card("整体转化率", format_pct(summary_conversion), "conv", f"参考 {format_pct(industry_conversion)}", summary_conversion >= industry_conversion, tooltip_html=metric_tooltips["整体转化率"])

metric_row2 = st.columns(5)
with metric_row2[0]:
    render_metric_card("总销售额", f"MX${summary_sales:,.2f}", "sales", tooltip_html=metric_tooltips["总销售额"])
with metric_row2[1]:
    render_metric_card("总销量", f"{summary_units:,.0f}", "sales", tooltip_html=metric_tooltips["总销量"])
with metric_row2[2]:
    render_metric_card("每单销售额", f"MX${summary_avg_sales_per_order:,.2f}", "sales", tooltip_html=metric_tooltips["每单销售额"])
with metric_row2[3]:
    render_metric_card("每单销售量", f"{summary_avg_units_per_order:,.2f}", "sales", tooltip_html=metric_tooltips["每单销售量"])
with metric_row2[4]:
    render_metric_card("日均销售额", f"MX${summary_avg_sales:,.2f}", "sales", tooltip_html=metric_tooltips["日均销售额"])

metric_row3 = st.columns(5)
with metric_row3[0]:
    render_metric_card("日均销量", f"{summary_avg_units:,.2f}", "sales", tooltip_html=metric_tooltips["日均销量"])
with metric_row3[1]:
    if single_goods_id:
        render_metric_card("库销比", f"{library_sales_ratio:,.2f}", "sales", tooltip_html=metric_tooltips["库销比"])

st.markdown("## 模块 2：每日趋势可视化区")
with st.container(border=True):
    chart1, chart2 = st.columns(2)
    with chart1:
        st.plotly_chart(make_impressions_ctr_chart(daily_df, ctr_target=industry_ctr), use_container_width=True)
    with chart2:
        st.plotly_chart(make_clicks_conversion_chart(daily_df, conversion_target=industry_conversion), use_container_width=True)

    if show_frontend_chart:
        if not frontend_filtered_df.empty:
            frontend_daily_df = build_frontend_daily_dataset(frontend_filtered_df)
            if not frontend_daily_df.empty:
                st.plotly_chart(make_frontend_price_sales_chart(frontend_daily_df), use_container_width=True)
                st.caption(f"价格销量走势图口径：{frontend_scope_text}。仅在选中单个 Goods ID 或单个 SKU 后显示。前端价 = 当日 Retail price (tax excl.) 总和 ÷ 当日 quantity purchased 总和 × 1.16；销量 = 当日 quantity purchased 汇总。")
            else:
                st.info("当前单品条件下没有可展示的前端订单数据。")
        else:
            st.info("当前单品条件下没有可展示的前端订单数据。")
    else:
        st.info("请选择单个 Goods ID 或单个 SKU 后，再查看前端价格与销量走势图。")

st.markdown("## 模块 3：每日数据明细区")
with st.container(border=True):
    sort_field_map = {
        "日期": "date",
        "Goods ID": "goods_id",
        "SKU": "display_sku",
        "曝光量": "impressions",
        "点击量": "clicks",
        "CTR": "ctr",
        "订单数": "orders",
        "转化率": "conversion_rate",
        "销售额": "sales_amount",
        "销量": "units_ordered",
        "买家数": "buyers",
        "每单销售额": "avg_sales_per_order",
        "每单销售量": "avg_units_per_order",
        "客均购买量": "avg_units_per_buyer",
        "日均销量": "avg_daily_units",
        "日均销售额": "avg_daily_sales",
    }
    sort_col1, sort_col2, sort_col3 = st.columns([1.1, 1.0, 1.8])
    with sort_col1:
        sort_field_label = st.selectbox("默认排序字段", list(sort_field_map.keys()), index=0)
    with sort_col2:
        sort_order_label = st.selectbox("默认排序方式", ["降序", "升序"], index=0)
    with sort_col3:
        st.caption("表格支持直接点击列头二次排序；下方配置用于初始化默认排序。异常行会对 CTR / 转化率 / 订单数 / 异常原因做红字提示。")
    ascending = sort_order_label == "升序"
    sort_columns = [sort_field_map[sort_field_label], "date", "goods_id", "display_sku"]
    ascending_list = [ascending] + [False if c == "date" else True for c in sort_columns[1:]]
    table_df = daily_detail_df.sort_values(sort_columns, ascending=ascending_list).copy()
    show_df = table_df.copy()
    show_df["日期"] = show_df["date"].dt.strftime("%Y-%m-%d")
    show_df["Goods ID"] = show_df["goods_id"].astype(str)
    show_df["SKU"] = show_df["display_sku"].astype(str)
    show_df["曝光量"] = show_df["impressions"].map(lambda x: f"{x:,.0f}")
    show_df["点击量"] = show_df["clicks"].map(lambda x: f"{x:,.0f}")
    show_df["CTR"] = show_df["ctr"].apply(format_pct)
    show_df["订单数"] = show_df["orders"].map(lambda x: f"{x:,.0f}")
    show_df["转化率"] = show_df["conversion_rate"].apply(format_pct)
    show_df["销售额"] = show_df["sales_amount"].map(lambda x: f"MX${x:,.2f}")
    show_df["销量"] = show_df["units_ordered"].map(lambda x: f"{x:,.0f}")
    show_df["买家数"] = show_df["buyers"].map(lambda x: f"{x:,.0f}")
    show_df["每单销售额"] = show_df["avg_sales_per_order"].map(lambda x: f"MX${x:,.2f}")
    show_df["每单销售量"] = show_df["avg_units_per_order"].map(lambda x: f"{x:,.2f}")
    show_df["客均购买量"] = show_df["avg_units_per_buyer"].map(lambda x: f"{x:,.2f}")
    show_df["日均销量"] = show_df["avg_daily_units"].map(lambda x: f"{x:,.2f}")
    show_df["日均销售额"] = show_df["avg_daily_sales"].map(lambda x: f"MX${x:,.2f}")
    show_df = show_df.rename(columns={"anomaly_reason": "异常原因"})
    show_df = show_df[["日期", "Goods ID", "SKU", "曝光量", "点击量", "CTR", "订单数", "转化率", "销售额", "销量", "买家数", "每单销售额", "每单销售量", "客均购买量", "日均销量", "日均销售额", "异常原因"]]
    st.caption("SKU 优先显示；若商品信息表缺少 SKU，则自动回退显示 Goods ID。导出文件额外附带字段说明页。")
    st.dataframe(style_daily_detail_table(show_df), use_container_width=True)

    st.download_button(
        label="导出当前筛选结果 Excel",
        data=build_export_file(table_df),
        file_name="营销数据看板_每日明细增强版.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("## 模块 4：SKU 流量销售联动分析区")
with st.container(border=True):
    sku_col1, sku_col2, sku_col3 = st.columns(3)
    with sku_col1:
        st.markdown("**SKU 综合表现 TOP20**")
        top_cols = pick_existing_columns(top20, ["store", "display_sku", "goods_id", "impressions", "clicks", "ctr", "orders", "sales_amount", "units_ordered", "avg_sales_per_order"])
        top_show = top20[top_cols].copy() if not top20.empty else pd.DataFrame(columns=top_cols)
        if "ctr" in top_show.columns:
            top_show["ctr"] = top_show["ctr"].apply(format_pct)
        top_show = top_show.rename(columns={"store": "店铺", "display_sku": "SKU", "goods_id": "Goods ID", "impressions": "曝光量", "clicks": "点击量", "ctr": "CTR", "orders": "订单数", "sales_amount": "销售额", "units_ordered": "销量", "avg_sales_per_order": "每单销售额"})
        st.dataframe(top_show, use_container_width=True, hide_index=True)
    with sku_col2:
        st.markdown("**异常 SKU 榜**")
        abnormal_cols = pick_existing_columns(abnormal_sku, ["abnormal_type", "store", "display_sku", "goods_id", "impressions", "clicks", "ctr", "orders", "conversion_rate", "units_ordered", "sales_amount", "inventory_qty"])
        abnormal_show = abnormal_sku[abnormal_cols].copy() if not abnormal_sku.empty else pd.DataFrame(columns=abnormal_cols)
        for pct_col in ["ctr", "conversion_rate"]:
            if pct_col in abnormal_show.columns:
                abnormal_show[pct_col] = abnormal_show[pct_col].apply(format_pct)
        abnormal_show = abnormal_show.rename(columns={"abnormal_type": "异常类型", "store": "店铺", "display_sku": "SKU", "goods_id": "Goods ID", "impressions": "曝光量", "clicks": "点击量", "ctr": "CTR", "orders": "订单数", "conversion_rate": "转化率", "units_ordered": "销量", "sales_amount": "销售额", "inventory_qty": "库存余量"})
        st.dataframe(abnormal_show, use_container_width=True, hide_index=True)
    with sku_col3:
        st.markdown("**未联动 SKU 明细**")
        unmatched_cols = pick_existing_columns(unmatched_sku, ["unmatched_type", "store", "display_sku", "goods_id", "impressions", "clicks", "ctr", "orders", "sales_amount", "units_ordered", "inventory_qty"])
        unmatched_show = unmatched_sku[unmatched_cols].copy() if not unmatched_sku.empty else pd.DataFrame(columns=unmatched_cols)
        if "ctr" in unmatched_show.columns:
            unmatched_show["ctr"] = unmatched_show["ctr"].apply(format_pct)
        unmatched_show = unmatched_show.rename(columns={"unmatched_type": "类型", "store": "店铺", "display_sku": "SKU", "goods_id": "Goods ID", "impressions": "曝光量", "clicks": "点击量", "ctr": "CTR", "orders": "订单数", "sales_amount": "销售额", "units_ordered": "销量", "inventory_qty": "库存余量"})
        st.dataframe(unmatched_show, use_container_width=True, hide_index=True)

st.markdown("## 模块 5：异常提示与行动指引区")
with st.container(border=True):
    analysis_texts = build_analysis_text(daily_df, industry_ctr, industry_conversion)
    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        st.markdown("**今日异常总览**")
        if today_alerts:
            for item in today_alerts:
                st.warning(item)
        else:
            st.info("今日未识别到明显异常。")
        st.markdown("**标签维度预警**")
        if tag_alerts:
            for item in tag_alerts[:6]:
                st.error(item)
        else:
            st.success("当前没有明显的标签预警。")
    with alert_col2:
        st.markdown("**历史异常复盘**")
        if history_alerts:
            for item in history_alerts[:7]:
                st.write(f"- {item}")
        else:
            st.write("近7天无明显历史异常。")
        st.markdown("**运营行动建议**")
        all_actions = analysis_texts + actions
        if all_actions:
            for item in all_actions[:6]:
                st.write(f"- {item}")
        else:
            st.write("当前无需额外动作建议。")

with st.expander("Goods ID 匹配检查"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("销售表Goods ID数", match_summary["销售表Goods ID数"])
    c2.metric("流量表Goods ID数", match_summary["流量表Goods ID数"])
    c3.metric("销售表未映射数", match_summary["销售表未映射数"])
    c4.metric("流量表未映射数", match_summary["流量表未映射数"])
    d1, d2 = st.columns(2)
    d1.metric("销售有但流量没有", match_summary["销售有但流量没有"])
    d2.metric("流量有但销售没有", match_summary["流量有但销售没有"])
    if match_details["traffic_not_in_sales"]:
        st.markdown("**流量表有但销售表没有的 Goods ID**")
        st.dataframe(pd.DataFrame({"Goods ID": match_details["traffic_not_in_sales"]}), use_container_width=True, hide_index=True)
    if match_details["sales_not_in_traffic"]:
        st.markdown("**销售表有但流量表没有的 Goods ID**")
        st.dataframe(pd.DataFrame({"Goods ID": match_details["sales_not_in_traffic"]}), use_container_width=True, hide_index=True)

with st.expander("字段说明"):
    st.markdown("""
- 匹配键统一使用 **Goods ID**
- 展示字段统一使用 **SKU**，未匹配时回退显示 Goods ID
- 产品筛选仅显示 **SKU / Goods ID**，不使用 **Product name / Goods Name**
- 每日明细已去掉“每点击销售额、每千曝光销售额”，改为销量、日均销量、日均销售额、每单销售量、每单销售额
""")
