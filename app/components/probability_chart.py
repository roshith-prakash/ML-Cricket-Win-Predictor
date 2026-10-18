import pandas as pd
import plotly.graph_objects as go

def render_probability_timeline(
    df_timeline: pd.DataFrame,
    team1: str,
    team2: str,
    is_full_match: bool = False
) -> go.Figure:
    """
    Renders an interactive line chart tracking win probability ball-by-ball.
    Supports either 2nd-innings chase or full match (Innings 1 + Innings 2).
    """
    fig = go.Figure()

    # 50% benchmark reference line
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="50-50 Parity",
        annotation_position="bottom right",
        annotation_font_color="#F59E0B"
    )

    # In full match replay, win probability is tracked from Team 1's perspective (0 to 100%)
    y_pct = df_timeline["team1_win_prob"] * 100 if "team1_win_prob" in df_timeline.columns else df_timeline["win_prob"] * 100

    fig.add_trace(go.Scatter(
        x=list(range(len(df_timeline))),
        y=y_pct,
        mode="lines",
        line=dict(color="#3B82F6", width=3, shape="spline"),
        name=f"{team1} Win %",
        hovertemplate=(
            "<b>%{customdata[0]}</b> (Over %{customdata[1]})<br>"
            "<b>Score:</b> %{customdata[2]}<br>"
            "<b>Event:</b> %{customdata[3]}<br>"
            "<b>Win Prob:</b> %{y:.1f}%<br>"
            "<b>Context:</b> %{customdata[4]}<extra></extra>"
        ),
        customdata=df_timeline[["innings_str", "over_str", "score_str", "event", "context_str"]].values
    ))

    # Innings break indicator if full match
    if is_full_match and "innings" in df_timeline.columns:
        inn2_start = df_timeline[df_timeline["innings"] == 2]
        if not inn2_start.empty:
            break_idx = inn2_start.index[0]
            fig.add_vline(
                x=break_idx,
                line_dash="dot",
                line_color="#10B981",
                line_width=2,
                annotation_text="Innings Break (Target Set)",
                annotation_position="top left",
                annotation_font_color="#10B981"
            )

    # Mark Wickets
    wickets_df = df_timeline[df_timeline["event"].str.contains("WICKET", case=False, na=False)]
    if not wickets_df.empty:
        fig.add_trace(go.Scatter(
            x=wickets_df.index,
            y=y_pct.loc[wickets_df.index],
            mode="markers",
            marker=dict(color="#EF4444", size=9, symbol="x"),
            name="Wicket",
            hovertemplate="<b>WICKET!</b> %{customdata[0]}<br>Score: %{customdata[1]}<extra></extra>",
            customdata=wickets_df[["event", "score_str"]].values
        ))

    # Mark Sixes
    sixes_df = df_timeline[df_timeline["event"].str.contains("SIX", case=False, na=False)]
    if not sixes_df.empty:
        fig.add_trace(go.Scatter(
            x=sixes_df.index,
            y=y_pct.loc[sixes_df.index],
            mode="markers",
            marker=dict(color="#10B981", size=8, symbol="star"),
            name="Six",
            hovertemplate="<b>SIX!</b> %{customdata[0]}<br>Score: %{customdata[1]}<extra></extra>",
            customdata=sixes_df[["event", "score_str"]].values
        ))

    fig.update_layout(
        title=f"Live Win Probability Progression — <b>{team1}</b> vs <b>{team2}</b>",
        xaxis=dict(
            title="Total Deliveries Bowled",
            showgrid=True,
            gridcolor="#374151",
            zeroline=False
        ),
        yaxis=dict(
            title=f"{team1} Win Probability (%)",
            range=[-2, 102],
            showgrid=True,
            gridcolor="#374151",
            zeroline=False
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font=dict(color="#F9FAFB", family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        margin=dict(t=50, b=40, l=50, r=30)
    )

    return fig
