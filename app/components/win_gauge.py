import plotly.graph_objects as go

def render_win_gauge(batting_team: str, bowling_team: str, win_prob: float) -> go.Figure:
    """
    Renders an interactive, polished dual-color gauge/donut
    showing chasing vs defending win probabilities.
    """
    chasing_pct = round(win_prob * 100, 1)
    defending_pct = round((1.0 - win_prob) * 100, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=chasing_pct,
        number={"suffix": "%", "font": {"size": 42, "color": "#10B981" if chasing_pct >= 50 else "#EF4444"}},
        title={"text": f"<b>{batting_team}</b><br><span style='font-size:0.85em;color:#9CA3AF'>Chasing Win Probability</span>", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4B5563"},
            "bar": {"color": "#10B981" if chasing_pct >= 50 else "#EF4444"},
            "bgcolor": "#1F2937",
            "borderwidth": 1,
            "bordercolor": "#374151",
            "steps": [
                {"range": [0, 50], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [50, 100], "color": "rgba(16, 185, 129, 0.15)"}
            ],
            "threshold": {
                "line": {"color": "#F59E0B", "width": 3},
                "thickness": 0.8,
                "value": 50
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F9FAFB", "family": "Inter, sans-serif"},
        height=280,
        margin={"t": 60, "b": 20, "l": 30, "r": 30}
    )

    return fig
