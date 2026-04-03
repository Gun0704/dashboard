from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import CTR_DROP_THRESHOLD, CTR_VS_7D_THRESHOLD, IMPRESSIONS_DROP_THRESHOLD, SALES_DROP_THRESHOLD, TAG_THRESHOLDS

def safe_divide(numerator, denominator):
    denom = denominator.replace(0, np.nan) if isinstance(denominator, pd.Series) else (np.nan if denominator == 0 else denominator)
    result = numerator / denom
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan).fillna(0)
    if pd.isna(result) or np.isinf(result):
        return 0
    return result

def build_daily_dataset(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=["date", "impressions", "clicks", "orders", "sales_amount", "units_ordered", "ctr", "conversion_rate"])
    daily = detail_df.groupby("date", as_index=False)[["impressions", "clicks", "orders", "sales_amount", "units_ordered"]].sum()
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["ctr"] = safe_divide(daily["clicks"], daily["impressions"])
    daily["conversion_rate"] = safe_divide(daily["orders"], daily["clicks"])
    daily["avg_sales_per_order"] = safe_divide(daily["sales_amount"], daily["orders"])
    daily["avg_units_per_order"] = safe_divide(daily["units_ordered"], daily["orders"])
    days_count = np.arange(1, len(daily) + 1)
    daily["avg_daily_units"] = daily["units_ordered"].cumsum() / days_count
    daily["avg_daily_sales"] = daily["sales_amount"].cumsum() / days_count
    return attach_anomalies(daily)



def build_daily_detail_dataset(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=[
            "date", "goods_id", "display_sku", "impressions", "clicks", "orders",
            "sales_amount", "units_ordered", "buyers", "ctr", "conversion_rate",
            "avg_sales_per_order", "avg_units_per_order", "avg_units_per_buyer", "avg_daily_units", "avg_daily_sales",
            "anomaly_reason", "anomaly_level", "is_anomaly"
        ])
    detail = detail_df.groupby(["date", "goods_id", "display_sku"], as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        orders=("orders", "sum"),
        sales_amount=("sales_amount", "sum"),
        units_ordered=("units_ordered", "sum"),
        buyers=("buyers", "sum"),
    )
    detail = detail.sort_values(["date", "goods_id", "display_sku"]).reset_index(drop=True)
    detail["ctr"] = safe_divide(detail["clicks"], detail["impressions"])
    detail["conversion_rate"] = safe_divide(detail["orders"], detail["clicks"])
    detail["avg_sales_per_order"] = safe_divide(detail["sales_amount"], detail["orders"])
    detail["avg_units_per_order"] = safe_divide(detail["units_ordered"], detail["orders"])
    detail["avg_units_per_buyer"] = safe_divide(detail["units_ordered"], detail["buyers"])
    days_count = max(detail_df["date"].nunique(), 1)
    detail["avg_daily_units"] = detail["units_ordered"] / days_count
    detail["avg_daily_sales"] = detail["sales_amount"] / days_count
    detail = attach_anomalies(detail)
    return detail

def attach_anomalies(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        daily = daily.copy()
        daily["is_anomaly"] = False
        daily["anomaly_reason"] = ""
        daily["anomaly_level"] = "normal"
        return daily
    daily = daily.sort_values("date").reset_index(drop=True).copy()
    ctr_7d_mean = daily["ctr"].rolling(7, min_periods=1).mean().shift(1)
    prev_ctr = daily["ctr"].shift(1)
    prev_impressions = daily["impressions"].shift(1)
    prev_sales = daily["sales_amount"].shift(1)
    reasons, flags, levels = [], [], []
    for idx, row in daily.iterrows():
        row_reasons = []
        level = "normal"
        if idx > 0 and prev_ctr.iloc[idx] > 0 and row["ctr"] < prev_ctr.iloc[idx] * (1 - CTR_DROP_THRESHOLD):
            row_reasons.append(f"CTR较前日下降超过{int(CTR_DROP_THRESHOLD * 100)}%")
            level = "high"
        if idx > 0 and prev_impressions.iloc[idx] > 0 and row["impressions"] < prev_impressions.iloc[idx] * (1 - IMPRESSIONS_DROP_THRESHOLD):
            row_reasons.append(f"曝光较前日下降超过{int(IMPRESSIONS_DROP_THRESHOLD * 100)}%")
            level = "high"
        if idx > 0 and prev_sales.iloc[idx] > 0 and row["sales_amount"] < prev_sales.iloc[idx] * (1 - SALES_DROP_THRESHOLD):
            row_reasons.append(f"销售额较前日下降超过{int(SALES_DROP_THRESHOLD * 100)}%")
            level = "high"
        if row["clicks"] > 0 and row["orders"] == 0:
            row_reasons.append("当日有点击但订单数为0")
            level = "high"
        if ctr_7d_mean.iloc[idx] > 0 and row["ctr"] < ctr_7d_mean.iloc[idx] * CTR_VS_7D_THRESHOLD:
            row_reasons.append(f"CTR低于近7天均值{int(CTR_VS_7D_THRESHOLD * 100)}%")
            level = "medium" if level == "normal" else level
        reasons.append("；".join(row_reasons))
        flags.append(bool(row_reasons))
        levels.append(level)
    daily["is_anomaly"] = flags
    daily["anomaly_reason"] = reasons
    daily["anomaly_level"] = levels
    return daily

def _calc_core_tag(avg_impressions: float, avg_orders: float, age_days: float | None) -> str:
    if age_days is not None and age_days < 14:
        return "新品"
    for label, imp_threshold, order_threshold in TAG_THRESHOLDS:
        if label == "大爆款":
            if avg_impressions >= imp_threshold and avg_orders > 50:
                return label
        elif avg_impressions >= imp_threshold and avg_orders >= order_threshold:
            return label
    return "滞制品"

def build_tag_snapshot(detail_df: pd.DataFrame, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    base = detail_df.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.floor("D")
    if as_of_date is None:
        as_of_date = base["date"].max()
    recent = base[(base["date"] <= as_of_date) & (base["date"] > as_of_date - pd.Timedelta(days=7))]
    prev = base[(base["date"] <= as_of_date - pd.Timedelta(days=7)) & (base["date"] > as_of_date - pd.Timedelta(days=14))]
    store_ctr = recent.groupby("store")["orders"].sum() / recent.groupby("store")["clicks"].sum().replace(0, np.nan)
    store_ctr = store_ctr.fillna(0.02)

    grouped_recent = recent.groupby("goods_id", as_index=False).agg(
        store=("store", "first"),
        display_sku=("display_sku", "first"),
        product_name=("product_name", "first"),
        inventory_qty=("inventory_qty", "max"),
        date_created=("date_created", "min"),
        impressions_7d=("impressions", "sum"),
        clicks_7d=("clicks", "sum"),
        orders_7d=("orders", "sum"),
        sales_7d=("sales_amount", "sum"),
        units_7d=("units_ordered", "sum"),
        zero_order_days_7d=("orders", lambda s: int((s == 0).sum())),
        zero_impression_days_7d=("impressions", lambda s: int((s == 0).sum())),
    )
    grouped_prev = prev.groupby("goods_id", as_index=False).agg(
        impressions_prev_7d=("impressions", "sum"),
        orders_prev_7d=("orders", "sum"),
        zero_order_days_prev_7d=("orders", lambda s: int((s == 0).sum())),
    )
    tag_df = grouped_recent.merge(grouped_prev, on="goods_id", how="left").fillna(0)
    tag_df["avg_impressions_7d"] = tag_df["impressions_7d"] / 7
    tag_df["avg_orders_7d"] = tag_df["orders_7d"] / 7
    tag_df["ctr"] = safe_divide(tag_df["clicks_7d"], tag_df["impressions_7d"])
    tag_df["conversion_rate"] = safe_divide(tag_df["orders_7d"], tag_df["clicks_7d"])
    tag_df["avg_sales_7d"] = tag_df["sales_7d"] / 7
    tag_df["avg_units_7d"] = tag_df["units_7d"] / 7
    tag_df["orders_growth"] = safe_divide(tag_df["avg_orders_7d"] - (tag_df["orders_prev_7d"] / 7), (tag_df["orders_prev_7d"] / 7).replace(0, np.nan))
    tag_df["impressions_growth"] = safe_divide(tag_df["avg_impressions_7d"] - (tag_df["impressions_prev_7d"] / 7), (tag_df["impressions_prev_7d"] / 7).replace(0, np.nan))
    tag_df["age_days"] = np.where(tag_df["date_created"].notna(), (as_of_date - pd.to_datetime(tag_df["date_created"])).dt.days, np.nan)
    tag_df["core_tag"] = tag_df.apply(lambda r: _calc_core_tag(r["avg_impressions_7d"], r["avg_orders_7d"], None if pd.isna(r["age_days"]) else float(r["age_days"])), axis=1)
    tag_df["core_tag"] = np.where((tag_df["age_days"] < 14) & tag_df["age_days"].notna(), "新品", tag_df["core_tag"])
    tag_df["is_stagnant"] = (
        (tag_df["avg_impressions_7d"] < 1000) |
        (tag_df["avg_orders_7d"] < 1) |
        ((tag_df["zero_order_days_7d"] >= 7) | (tag_df["zero_impression_days_7d"] >= 7)) |
        ((tag_df["age_days"] < 14) & tag_df["age_days"].notna() & (tag_df["age_days"] >= 7) & (tag_df["orders_7d"] <= 0))
    )
    tag_df["trend_up"] = (
        (tag_df["orders_growth"] >= 0.20) &
        (tag_df["impressions_growth"] >= 0.15) &
        (tag_df["zero_order_days_7d"] < 2) &
        (tag_df["conversion_rate"] >= tag_df["store"].map(store_ctr).fillna(0.02))
    )
    tag_df["display_tag"] = tag_df["core_tag"]
    tag_df["display_tag"] = np.where((tag_df["core_tag"] == "新品") & tag_df["is_stagnant"], "新品 + 滞制品", tag_df["display_tag"])
    tag_df["display_tag"] = np.where((tag_df["core_tag"] != "新品") & tag_df["is_stagnant"], "滞制品", tag_df["display_tag"])
    tag_df["tag_short"] = tag_df["display_tag"] + np.where(tag_df["trend_up"], "↑", "")
    return tag_df

def build_sku_tables(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if detail_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sku_detail = detail_df.groupby(["store", "goods_id", "display_sku"], as_index=False).agg(
        product_name=("product_name", "first"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        orders=("orders", "sum"),
        sales_amount=("sales_amount", "sum"),
        units_ordered=("units_ordered", "sum"),
        inventory_qty=("inventory_qty", "max"),
        sales_rows=("has_sales", "sum"),
        traffic_rows=("has_traffic", "sum"),
    )
    sku_detail["ctr"] = safe_divide(sku_detail["clicks"], sku_detail["impressions"])
    sku_detail["conversion_rate"] = safe_divide(sku_detail["orders"], sku_detail["clicks"])
    sku_detail["avg_sales_per_order"] = safe_divide(sku_detail["sales_amount"], sku_detail["orders"])
    sku_detail["avg_units_per_order"] = safe_divide(sku_detail["units_ordered"], sku_detail["orders"])

    top20 = sku_detail.sort_values(["sales_amount", "units_ordered", "orders"], ascending=[False, False, False]).head(20).copy()

    abnormal1 = sku_detail[(sku_detail["impressions"] >= 5000) & (sku_detail["conversion_rate"] < 0.02)].copy()
    abnormal1["abnormal_type"] = "高曝光低转化"
    abnormal2 = sku_detail[(sku_detail["conversion_rate"] >= 0.03) & (sku_detail["units_ordered"] < 10)].copy()
    abnormal2["abnormal_type"] = "高转化低销量"
    abnormal = pd.concat([abnormal1, abnormal2], ignore_index=True)
    abnormal = abnormal.sort_values(["abnormal_type", "impressions", "sales_amount"], ascending=[True, False, False]).drop_duplicates(subset=["goods_id", "abnormal_type"])

    unmatched = sku_detail[(sku_detail["sales_rows"] == 0) | (sku_detail["traffic_rows"] == 0)].copy()
    unmatched["unmatched_type"] = np.where((unmatched["traffic_rows"] > 0) & (unmatched["sales_rows"] == 0), "流量有但销售无", "销售有但流量无")
    unmatched = unmatched.sort_values(["unmatched_type", "impressions", "sales_amount"], ascending=[True, False, False])
    return top20, abnormal, unmatched

def build_alerts(detail_df: pd.DataFrame, tag_df: pd.DataFrame, daily_df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    today_alerts, history_alerts, tag_alerts, actions = [], [], [], []
    if not daily_df.empty:
        latest = daily_df.iloc[-1]
        if latest["is_anomaly"]:
            today_alerts.append(f"{latest['date'].strftime('%Y-%m-%d')} 出现异常：{latest['anomaly_reason']}")
        recent_anoms = daily_df[daily_df["is_anomaly"]].tail(7)
        for _, row in recent_anoms.iterrows():
            history_alerts.append(f"{row['date'].strftime('%m-%d')}：{row['anomaly_reason']}")
    if not tag_df.empty:
        explosion = tag_df[(tag_df["core_tag"].isin(["大爆款", "爆款"])) & (tag_df["orders_growth"] <= -0.20)]
        for _, row in explosion.head(5).iterrows():
            tag_alerts.append(f"{row['display_sku']} 爆款预警：近7天日均单量 {row['avg_orders_7d']:.1f} 单。")
        new_bad = tag_df[(tag_df["age_days"] < 14) & tag_df["age_days"].notna() & (tag_df["age_days"] >= 7) & (tag_df["orders_7d"] <= 0)]
        for _, row in new_bad.head(5).iterrows():
            tag_alerts.append(f"{row['display_sku']} 新品预警：上架 {int(row['age_days'])} 天仍未破零。")
        stagnant = tag_df[(tag_df["display_tag"].str.contains("滞制品", na=False))]
        for _, row in stagnant.head(5).iterrows():
            tag_alerts.append(f"{row['display_sku']} 滞制品预警：近7天日均曝光 {row['avg_impressions_7d']:.0f}，日均单量 {row['avg_orders_7d']:.1f}。")
        trend_bad = tag_df[tag_df["trend_up"] & (tag_df["orders_7d"] <= 0)]
        for _, row in trend_bad.head(5).iterrows():
            tag_alerts.append(f"{row['display_sku']} 上升趋势品预警：连续零单，建议加大承接优化。")
        high_exp_low_conv = tag_df[(tag_df["impressions_7d"] >= 5000) & (tag_df["conversion_rate"] < 0.02)]
        if not high_exp_low_conv.empty:
            row = high_exp_low_conv.sort_values("impressions_7d", ascending=False).iloc[0]
            actions.append(f"重点提升：{row['display_sku']} 近7天曝光 {row['impressions_7d']:.0f} 但转化率仅 {row['conversion_rate']:.2%}，建议优化详情页、评价与活动承接。")
        if not new_bad.empty:
            row = new_bad.iloc[0]
            actions.append(f"紧急优化：{row['display_sku']} 上架 {int(row['age_days'])} 天仍未破零，建议立即检查主图、标题和首单激励。")
        trend_good = tag_df[tag_df["trend_up"]]
        if not trend_good.empty:
            row = trend_good.sort_values("orders_growth", ascending=False).iloc[0]
            actions.append(f"长期培养：{row['display_sku']} 呈上升趋势，建议加大优质流量投放并扩展关联SKU。")
    return today_alerts, history_alerts, tag_alerts, actions

def build_analysis_text(daily: pd.DataFrame, industry_ctr: float, industry_conversion: float) -> List[str]:
    texts: List[str] = []
    if daily.empty:
        return texts
    overall_ctr = safe_divide(daily["clicks"].sum(), daily["impressions"].sum())
    overall_conversion = safe_divide(daily["orders"].sum(), daily["clicks"].sum())
    if overall_ctr < industry_ctr:
        texts.append(f"整体 CTR 为 {overall_ctr:.2%}，低于参考值 {industry_ctr:.2%}，建议优先排查主图、标题与投放词包。")
    if overall_conversion < industry_conversion:
        texts.append(f"整体转化率为 {overall_conversion:.2%}，低于参考值 {industry_conversion:.2%}，建议重点优化详情页承接、价格与评价。")
    anomalies = daily[daily["is_anomaly"]]
    if not anomalies.empty:
        first = anomalies.iloc[-1]
        texts.append(f"{first['date'].strftime('%Y-%m-%d')} 出现异常：{first['anomaly_reason']}。")
    return texts