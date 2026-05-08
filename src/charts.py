from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _add_anomaly_trace(fig: go.Figure, anomalies: pd.DataFrame, y: pd.Series, yaxis: str) -> None:
    if anomalies.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=anomalies["date"],
            y=y.loc[anomalies.index],
            name="异常点",
            mode="markers",
            marker=dict(size=12, color="red"),
            yaxis=yaxis,
            text=anomalies["anomaly_reason"],
            hovertemplate="%{x|%m-%d}<br>%{text}<extra></extra>",
        )
    )


def make_impressions_ctr_chart(daily: pd.DataFrame, ctr_target: float = 0.03) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["impressions"],
            name="曝光量",
            marker_color="#1f77b4",
            opacity=0.75,
            yaxis="y1",
            hovertemplate="%{x|%m-%d}<br>曝光量=%{y:,.0f}<br>点击率=%{customdata[0]:.2%}<br>点击量=%{customdata[1]:,.0f}<extra></extra>",
            customdata=daily[["ctr", "clicks"]].values,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["ctr"],
            name="点击率",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=8),
            yaxis="y2",
            hovertemplate="%{x|%m-%d}<br>点击率=%{y:.2%}<br>曝光量=%{customdata[0]:,.0f}<br>点击量=%{customdata[1]:,.0f}<extra></extra>",
            customdata=daily[["impressions", "clicks"]].values,
        )
    )
    if ctr_target is not None:
        fig.add_hline(y=ctr_target, line_color="#e74c3c", line_dash="dash", yref="y2", annotation_text="CTR目标线")
    anomalies = daily[daily["is_anomaly"]]
    _add_anomaly_trace(fig, anomalies, daily["ctr"], "y2")
    fig.update_layout(
        title="曝光量和点击率",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(title="", tickformat="%m-%d", tickfont=dict(size=13)),
        yaxis=dict(title=dict(text="曝光量", font=dict(size=13)), tickfont=dict(size=12)),
        yaxis2=dict(title=dict(text="点击率", font=dict(size=13)), overlaying="y", side="right", tickformat=".0%", tickfont=dict(size=12)),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    return fig


def make_clicks_conversion_chart(daily: pd.DataFrame, conversion_target: float = 0.0) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["clicks"],
            name="点击量",
            marker_color="#5dade2",
            opacity=0.8,
            yaxis="y1",
            hovertemplate="%{x|%m-%d}<br>点击量=%{y:,.0f}<br>转化率=%{customdata[0]:.2%}<br>订单数=%{customdata[1]:,.0f}<extra></extra>",
            customdata=daily[["conversion_rate", "orders"]].values,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["conversion_rate"],
            name="转化率",
            mode="lines+markers",
            line=dict(color="#27ae60", width=3),
            marker=dict(size=8),
            yaxis="y2",
            hovertemplate="%{x|%m-%d}<br>转化率=%{y:.2%}<br>点击量=%{customdata[0]:,.0f}<br>订单数=%{customdata[1]:,.0f}<extra></extra>",
            customdata=daily[["clicks", "orders"]].values,
        )
    )
    if conversion_target is not None:
        fig.add_hline(y=conversion_target, line_color="#27ae60", line_dash="dash", yref="y2", annotation_text="转化率目标线")
    anomalies = daily[daily["is_anomaly"]]
    _add_anomaly_trace(fig, anomalies, daily["conversion_rate"], "y2")
    fig.update_layout(
        title="点击量和转化率",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(title="", tickformat="%m-%d", tickfont=dict(size=13)),
        yaxis=dict(title=dict(text="点击量", font=dict(size=13)), tickfont=dict(size=12)),
        yaxis2=dict(title=dict(text="转化率", font=dict(size=13)), overlaying="y", side="right", tickformat=".0%", tickfont=dict(size=12)),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    return fig


def make_frontend_price_sales_chart(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["quantity_purchased"],
            name="销量",
            marker_color="#d2b4de",
            opacity=0.6,
            yaxis="y2",
            hovertemplate="%{x|%m-%d}<br>销量=%{y:,.0f}<br>前端价=%{customdata[0]:,.2f}<extra></extra>",
            customdata=daily[["frontend_price"]].values,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["frontend_price"],
            name="前端价",
            mode="lines+markers",
            line=dict(color="#8e44ad", width=4),
            marker=dict(size=10, color="#8e44ad", line=dict(color="white", width=2)),
            yaxis="y1",
            hovertemplate="%{x|%m-%d}<br>前端价=%{y:,.2f}<br>销量=%{customdata[0]:,.0f}<extra></extra>",
            customdata=daily[["quantity_purchased"]].values,
        )
    )
    fig.update_layout(
        title="前端价格销量走势图",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(title="", tickformat="%m-%d", tickfont=dict(size=13)),
        yaxis=dict(title=dict(text="前端价", font=dict(size=13)), tickfont=dict(size=12)),
        yaxis2=dict(title=dict(text="销量", font=dict(size=13)), overlaying="y", side="right", tickfont=dict(size=12)),
        legend=dict(orientation="h", y=1.08, traceorder="reversed"),
        hovermode="x unified",
    )
    return fig
