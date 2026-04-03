from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

def _date_tickvals(daily: pd.DataFrame):
    return daily["date"]

def make_dual_axis_trend_chart(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["impressions"], name="曝光量",
        mode="lines+markers", line=dict(color="#1f4e79", width=3), marker=dict(size=8), yaxis="y1",
        hovertemplate="%{x|%m-%d}<br>曝光量=%{y:,.0f}<br>点击量=%{customdata[0]:,.0f}<br>CTR=%{customdata[1]:.2%}<extra></extra>",
        customdata=daily[["clicks", "ctr"]].values
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["clicks"], name="点击量",
        mode="lines+markers", line=dict(color="#5dade2", width=3), marker=dict(size=8), yaxis="y2",
        hovertemplate="%{x|%m-%d}<br>点击量=%{y:,.0f}<br>曝光量=%{customdata[0]:,.0f}<br>CTR=%{customdata[1]:.2%}<extra></extra>",
        customdata=daily[["impressions", "ctr"]].values
    ))
    anomalies = daily[daily["is_anomaly"]]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies["clicks"], name="异常点", mode="markers",
            marker=dict(size=12, color="red"), yaxis="y2", text=anomalies["anomaly_reason"],
            hovertemplate="%{x|%m-%d}<br>%{text}<extra></extra>",
        ))
    fig.update_layout(
        height=420, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="", tickformat="%m-%d", tickfont=dict(size=13)),
        yaxis=dict(title=dict(text="曝光量", font=dict(size=13)), tickfont=dict(size=12)),
        yaxis2=dict(title=dict(text="点击量", font=dict(size=13)), overlaying="y", side="right", tickfont=dict(size=12)),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    return fig

def make_orders_ctr_chart(daily: pd.DataFrame, ctr_target: float = 0.03) -> go.Figure:
    order_target = daily["orders"].tail(7).mean() if not daily.empty else 0
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["orders"], name="订单数", marker_color="#2e8b57", yaxis="y1",
        hovertemplate="%{x|%m-%d}<br>订单数=%{y:,.0f}<br>转化率=%{customdata[0]:.2%}<extra></extra>",
        customdata=daily[["conversion_rate"]].values
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ctr"], name="CTR", mode="lines+markers",
        line=dict(color="#c0392b", width=3), marker=dict(size=8), yaxis="y2",
        hovertemplate="%{x|%m-%d}<br>CTR=%{y:.2%}<br>近7天CTR均值=%{customdata[0]:.2%}<extra></extra>",
        customdata=daily["ctr"].rolling(7, min_periods=1).mean().values.reshape(-1, 1)
    ))
    fig.add_hline(y=order_target, line_color="#2e8b57", line_dash="dash", yref="y", annotation_text="订单目标线")
    fig.add_hline(y=ctr_target, line_color="#c0392b", line_dash="dash", yref="y2", annotation_text="CTR目标线")
    anomalies = daily[daily["is_anomaly"]]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies["date"], y=anomalies["ctr"], name="异常点", mode="markers", marker=dict(size=12, color="red"), yaxis="y2", text=anomalies["anomaly_reason"], hovertemplate="%{x|%m-%d}<br>%{text}<extra></extra>"))
    fig.update_layout(
        height=420, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="", tickformat="%m-%d", tickfont=dict(size=13)),
        yaxis=dict(title=dict(text="订单数", font=dict(size=13)), tickfont=dict(size=12)),
        yaxis2=dict(title=dict(text="CTR", font=dict(size=13)), overlaying="y", side="right", tickformat=".0%", tickfont=dict(size=12)),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    return fig

def make_sales_units_chart(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["sales_amount"], name="销售额",
        mode="lines+markers", line=dict(color="#196f3d", width=3), marker=dict(size=8), yaxis="y1",
        hovertemplate="%{x|%m-%d}<br>销售额=%{y:,.2f}<br>销量=%{customdata[0]:,.0f}<br>每单销售额=%{customdata[1]:,.2f}<br>每单销售量=%{customdata[2]:,.2f}<extra></extra>",
        customdata=daily[["units_ordered", "avg_sales_per_order", "avg_units_per_order"]].values
    ))
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["units_ordered"], name="销量", marker_color="#7dcea0", yaxis="y2",
        hovertemplate="%{x|%m-%d}<br>销量=%{y:,.0f}<br>销售额=%{customdata[0]:,.2f}<extra></extra>",
        customdata=daily[["sales_amount"]].values
    ))
    anomalies = daily[daily["is_anomaly"]]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies["date"], y=anomalies["sales_amount"], name="异常点", mode="markers", marker=dict(size=12, color="red"), yaxis="y1", text=anomalies["anomaly_reason"], hovertemplate="%{x|%m-%d}<br>%{text}<extra></extra>"))
    fig.update_layout(
        height=420, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="", tickformat="%m-%d", tickfont=dict(size=13)),
        yaxis=dict(title=dict(text="销售额", font=dict(size=13)), tickfont=dict(size=12)),
        yaxis2=dict(title=dict(text="销量", font=dict(size=13)), overlaying="y", side="right", tickfont=dict(size=12)),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    return fig