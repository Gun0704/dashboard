from __future__ import annotations

import io
import pandas as pd


def format_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def pick_existing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def style_daily_detail_table(show_df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def _highlight_row(row: pd.Series) -> list[str]:
        styles = ["" for _ in row.index]
        reason = str(row.get("异常原因", "") or "").strip()
        abnormal = bool(reason)
        for idx, col in enumerate(row.index):
            if col in {"CTR", "转化率", "订单数", "异常原因"} and abnormal:
                styles[idx] = "color: #b91c1c; font-weight: 700;"
        return styles

    styler = show_df.style.hide(axis="index")
    styler = styler.apply(_highlight_row, axis=1)
    styler = styler.set_properties(**{"white-space": "nowrap"}, subset=[c for c in ["日期", "Goods ID", "SKU"] if c in show_df.columns])
    styler = styler.set_properties(**{"white-space": "normal"}, subset=[c for c in ["异常原因"] if c in show_df.columns])
    return styler


def _format_detail_export_df(export_df: pd.DataFrame) -> pd.DataFrame:
    export_df = export_df.copy()
    export_df["日期"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")
    if "goods_id" in export_df.columns:
        export_df["Goods ID"] = export_df["goods_id"].astype(str)
    if "display_sku" in export_df.columns:
        export_df["SKU"] = export_df["display_sku"].astype(str)
    export_df["曝光量"] = export_df["impressions"].map(lambda x: f"{x:,.0f}")
    export_df["点击量"] = export_df["clicks"].map(lambda x: f"{x:,.0f}")
    export_df["CTR"] = export_df["ctr"].apply(format_pct)
    export_df["订单数"] = export_df["orders"].map(lambda x: f"{x:,.0f}")
    export_df["转化率"] = export_df["conversion_rate"].apply(format_pct)
    export_df["销售额"] = export_df["sales_amount"].map(lambda x: f"MX${x:,.2f}")
    export_df["销量"] = export_df["units_ordered"].map(lambda x: f"{x:,.0f}")
    export_df["买家数"] = export_df["buyers"].map(lambda x: f"{x:,.0f}")
    export_df["日均销量"] = export_df["avg_daily_units"].map(lambda x: f"{x:,.2f}")
    export_df["日均销售额"] = export_df["avg_daily_sales"].map(lambda x: f"MX${x:,.2f}")
    export_df["每单销售额"] = export_df["avg_sales_per_order"].map(lambda x: f"MX${x:,.2f}")
    export_df["每单销售量"] = export_df["avg_units_per_order"].map(lambda x: f"{x:,.2f}")
    export_df["客均购买量"] = export_df["avg_units_per_buyer"].map(lambda x: f"{x:,.2f}")
    export_df = export_df.rename(columns={"anomaly_reason": "异常原因"})
    desired = [
        "日期", "Goods ID", "SKU", "曝光量", "点击量", "CTR", "订单数", "转化率",
        "销售额", "销量", "买家数", "每单销售额", "每单销售量", "客均购买量",
        "日均销量", "日均销售额", "异常原因"
    ]
    desired = [c for c in desired if c in export_df.columns]
    return export_df[desired]


def build_export_file(df: pd.DataFrame) -> bytes:
    export_df = _format_detail_export_df(df)
    desc_df = pd.DataFrame({
        "列组": [
            "基础信息", "基础信息", "基础信息", "流量指标", "流量指标", "流量指标",
            "转化指标", "转化指标", "销售指标", "销售指标", "销售指标",
            "销售指标", "销售指标", "销售指标", "销售指标", "销售指标", "说明"
        ],
        "字段": [
            "日期", "Goods ID", "SKU", "曝光量", "点击量", "CTR", "订单数", "转化率", "销售额", "销量", "买家数",
            "每单销售额", "每单销售量", "客均购买量", "日均销量", "日均销售额", "异常原因"
        ],
        "格式要求": [
            "YYYY-MM-DD", "文本", "文本", "整数", "整数", "保留2位小数，带%", "整数", "保留2位小数，带%",
            "MX$，保留2位小数", "整数", "整数", "MX$，保留2位小数", "保留2位小数",
            "保留2位小数", "保留2位小数", "MX$，保留2位小数", "文本"
        ],
        "说明": [
            "具体交易日期", "商品唯一标识，作为匹配键", "SKU；若缺失则回退显示 Goods ID", "当日累计产品曝光次数", "当日累计产品点击次数", "点击量 / 曝光量 × 100%",
            "按当前订单口径汇总的订单数", "订单数 / 点击量 × 100%", "当日 Base price sales 合计", "当日 Units ordered 合计", "当日累计 Buyers",
            "销售额 / 订单数", "销量 / 订单数", "销量 / 买家数", "筛选期间累计销量 / 天数", "筛选期间累计销售额 / 天数", "异常识别命中后的原因说明"
        ],
        "数据来源": [
            "Sales/Traffic", "Mapping/Sales/Traffic", "Mapping", "Traffic", "Traffic", "Traffic",
            "Sales", "Sales+Traffic", "Sales", "Sales", "Sales", "Sales", "Sales", "Sales", "Sales", "Sales", "规则引擎"
        ],
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False, sheet_name="每日明细")
        desc_df.to_excel(writer, index=False, sheet_name="字段说明")

        workbook = writer.book
        detail_ws = writer.sheets["每日明细"]
        desc_ws = writer.sheets["字段说明"]

        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "align": "center", "valign": "vcenter"})
        abnormal_fmt = workbook.add_format({"font_color": "#B91C1C", "bold": True})
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        for idx, col in enumerate(export_df.columns):
            detail_ws.write(0, idx, col, header_fmt)
            width = max(12, min(28, max(len(str(col)) + 2, int(export_df[col].astype(str).str.len().quantile(0.9)) if not export_df.empty else 12)))
            detail_ws.set_column(idx, idx, width)

        if "异常原因" in export_df.columns:
            abnormal_col = export_df.columns.get_loc("异常原因")
            detail_ws.set_column(abnormal_col, abnormal_col, 32, wrap_fmt)
            letter = chr(65 + abnormal_col)
            detail_ws.conditional_format(1, 0, len(export_df), len(export_df.columns) - 1, {
                "type": "formula",
                "criteria": f'=LEN(${letter}2)>0',
                "format": abnormal_fmt,
            })

        for idx, col in enumerate(desc_df.columns):
            desc_ws.write(0, idx, col, header_fmt)
            desc_ws.set_column(idx, idx, 22 if col != "说明" else 34)

    return output.getvalue()
