# streamlit run trust_streamlit.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import openpyxl
import kaleido
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import scipy.stats as stats

dta = pd.read_excel('final_synthetic_data.xlsx')

# Rename columns for aesthetic
dta = dta.rename(columns = {'trust_current_advisor': 'Trust Rating',
               'delegate_current_advisor': 'Delegate Rating',
               'how_often_recommend_current': 'Recommendation Rating',
               'how_often_talk_current': 'How often do you talk to your advisor',
               'Communicate with you:_how_often_behavior_time_current': 'How often advisor communicates with you',
               'Helps you make progress toward your financial goals:_how_often_behavior_time_current': 'How often advisor helps make progress to goals',
               'Show that they understand your financial goals:_how_often_behavior_time_current': 'How often advisor shows they understand your goals',
               'Keeps their word:_how_often_behavior_chance_current': 'How often advisor keeps their word',
               'Acts in your best interest:_how_often_behavior_chance_current':'How often advisor acts in your best interest',
               'Listen to your point of view, suggestions, and needs and act accordingly:_how_often_behavior_chance_current': 'How often advisor listens to your point of view, suggestions, needs, and acts accordingly',
               'Show that they care about your future:_how_often_behavior_chance_current': 'How often advisor shows they care about your future',
               'Make efforts to treat all customers equally:_how_often_behavior_chance_current': 'How ofen advisor makes efforts to treat all customers equally',
               'Take the time to explain their decisions to you:_how_often_behavior_chance_current': 'How often advisor takes the time to explain their decisions to you',
               'Competently handle your requests:_how_often_behavior_chance_current':'How often advisor competently handles your requests',
               'Show that they are knowledgeable:_how_often_behavior_chance_current': 'How often advisor shows they are knowledgeable',
               'Show they have same concerns as you:_how_often_behavior_chance_current': 'How often advisor shows they have the same concerns as you',
               'Show they have the same values as you:_how_often_behavior_chance_current': 'How often advisor shows they have the same values as you',
               'Act as you would have acted:_how_often_behavior_chance_current': 'How often advisor acts as you would have acted',
               'Treat you with respect:_how_often_behavior_chance_current': 'How often advisor treats you with respect',
               'Show consideration in their dealings with you:_how_often_behavior_chance_current': 'How often advisor shows consideration in their dealings with you',
               'Provide advice that is suitable for you and your circumstances:_how_often_behavior_chance_current': 'How often advisor provides advice that is suitable for you'})

dta_pl = pl.DataFrame(dta)
# ----------------------------.


# Functions

def single_bar_plot(df, var, title_content, label_neg, label_pos):
    import plotly.express as px

    count_df = df[var].value_counts()
    count_df_sum = len(df)
    count_df2 = count_df.div(count_df_sum, axis=0)

    ind = count_df2.index.values
    fig = px.bar(x=ind, y=count_df2)

    fig.update_traces(hovertemplate=f"{title_content}: %{{x}}<br>Proportion: %{{y:.3f}}<extra></extra>")

    # Axis labels, title, ticks, and x-axis range
    fig.update_layout(
        xaxis_title=f"n = {count_df_sum}",
        yaxis_title="Proportion",
        title=f"Distribution of {title_content}",
        xaxis=dict(
            tickmode='array',
            tickvals=ind,  # explicit ticks to match your 'ind'
            range=[0, 8]  # equivalent to plt.xlim([0, 8])
        )
    )

    # Top-left annotation (similar to annotate at (0,0) with offset (0,-20))
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0, y=0,  # top-left in the plotting area
        text=label_neg,
        showarrow=False,
        xanchor='left', yanchor='top',
        yshift=-20  # pixel offset down
    )

    # Top-right annotation (similar to offset to the right, then -20 vertically)
    fig.add_annotation(
        xref='paper', yref='paper',
        x=1, y=0,  # top-right in the plotting area
        text=label_pos,
        showarrow=False,
        xanchor='right', yanchor='top',
        xshift=-20,  # small pixel offset left
        yshift=-20  # pixel offset down
    )

    # Return the figure if you're in a function
    return fig


def box_plot(df, xvar, yvar, xlabel, ylabel, annotate_x, annotate_y, title_content):
    import numpy as np
    import plotly.express as px

    fig = go.Figure()
    fig.add_trace(go.Box(y=df[yvar], x=df[xvar],
                         fillcolor='rgb(26, 80, 155)'))
    fig.update_layout(title={ 'text':title_content,
                              'y': 0.9,
                              'x': .6})

    # Axis labels, title, ticks, and x-axis range
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        title=title_content,
    )

    # X annotation
    fig.add_annotation(
        xref='paper', yref='paper',
        x=1, y=0,  # top-left in the plotting area
        text=annotate_x,
        showarrow=False,
        xanchor='left', yanchor='top',
        xshift=-60,
        yshift=-20  # pixel offset down
    )

    # Y annotation
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0, y=1,  # top-right in the plotting area
        text=annotate_y,
        showarrow=False,
        xanchor='right', yanchor='top',
        xshift=0,  # small pixel offset left
        yshift=25  # pixel offset down
    )

    return fig



def scatter_plot(df, xvar, yvar,  ylabel, annotate_x, annotate_y, title_content):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from scipy import stats

    # jitter for plotting
    df['x_jittered'] = df[xvar].astype(float) + np.random.uniform(-0.4, 0.4, size=len(df[xvar]))
    df['y_jittered'] = df[yvar].astype(float) + np.random.uniform(-0.4, 0.4, size=len(df[yvar]))

    xaxis_label = xvar.split(":", 1)[0]

    x = pd.to_numeric(df[xvar], errors='coerce')
    y = pd.to_numeric(df[yvar], errors='coerce')
    mask = x.notna() & y.notna()
    try:
        r, p = stats.pearsonr(x[mask], y[mask])
    except Exception:
        r, p = (float('nan'), float('nan'))

    # determine effect-size label using absolute r
    abs_r = abs(r) if not np.isnan(r) else np.nan
    if np.isnan(abs_r):
        effect_text = "Effect: N/A"
    elif abs_r <= 0.3:
        effect_text = "Small Effect"
    elif abs_r < 0.5:
        effect_text = "Medium Effect"
    else:
        effect_text = "Large Effect"

    # pass original values and advisor as custom_data; plot uses jittered coords
    fig = px.scatter(
        df,
        x='x_jittered',
        y='y_jittered',
        trendline='ols',
        custom_data=[xvar, yvar, 'advisor']
    )

    fig.update_layout(
        xaxis_title=xaxis_label,
        yaxis_title=ylabel,
        title=title_content,
        coloraxis_showscale=True,
    )

    # style trendline only
    fig.update_traces(line=dict(color='black', width=2), selector=dict(mode='lines'))

    # marker style and custom hover for markers only
    fig.update_traces(marker=dict(color='rgb(26, 80, 155)'), selector=dict(mode='markers'))
    fig.update_traces(
        hovertemplate=(
            "Behavior Frequency: %{customdata[0]:.0f}<br>"
            "Trust: %{customdata[1]:.0f}<br>"
            "Advisor: %{customdata[2]}<extra></extra>"
        ),
        selector=dict(mode='markers')
    )

    # annotations
    fig.add_annotation(xref='paper', yref='paper', x=1, y=0, text=annotate_x,
                       showarrow=False, xanchor='left', yanchor='top', xshift=-60, yshift=-20)
    fig.add_annotation(xref='paper', yref='paper', x=0, y=1, text=annotate_y,
                       showarrow=False, xanchor='right', yanchor='top', xshift=50, yshift=0)

    # top-right: correlation and p-value (above)
    fig.add_annotation(
        xref='paper', yref='paper', x=1, y=1,
        text=f"r = {r:.2f}",
        showarrow=False, xanchor='right', yanchor='top', xshift=0, yshift=20,
        font=dict(size=12)
    )

    # effect label directly below the correlation annotation
    fig.add_annotation(
        xref='paper', yref='paper', x=1, y=1,
        text=effect_text,
        showarrow=False, xanchor='right', yanchor='top', xshift=0, yshift=2,
        font=dict(size=12)
    )

    return fig


TEXT_WIDTH = 700
SMALLPLOT_WIDTH = 500
SMALLPLOT_HEIGHT = 500
MARK_SIZE = 70
DIRECTOR_MARK_SIZE = 150


rating_df = (
                dta_pl.select(pl.col('advisor'), pl.col('Delegate Rating'), pl.col('Trust Rating'),
                              pl.col('Unnamed: 0'))
                .filter(
                    pl.col('Delegate Rating').is_not_null() & pl.col('Trust Rating').is_not_null()
                )
                .with_columns(
                    delta=pl.col('Delegate Rating') / 7 - pl.col('Trust Rating') / 7,
                )
            ).sort(by="delta", descending=True)


rating_df.select(pl.exclude("delta")).head(20).sort(by="Trust Rating", descending=True)


def wide_centered_layout():
    with st.container(horizontal_alignment="center"):
        return st.container(
            width=2 * SMALLPLOT_WIDTH + 16, horizontal_alignment="center"
        )


# -----------------------------------------------------------------------------
# Draw app


with wide_centered_layout():
    with st.container(width=TEXT_WIDTH):
        st.title("What Makes Clients Trust Their Advisor?")

        st.space()

        """
        Trust is the cornerstone of the client advisor relationship and yet, little research has been done to 
        understand the drivers of trust and it's impact on an advisor's practice.
        
        This analysis builds off of [Morningstar's original research](https://assets.contentstack.io/v3/assets/blt4eb669caa7dc65b2/blt7bddca582721d669/how-to-build-trust-in-the-advisor-client-relationship.pdf?date=2025-02-07) on this topic by exploring similar data for a 
        fictional advisory group. The data used in this analysis is synthetic and does not represent any real clients or advisors. 
        However, the data was generated to mimic real world data.
        
        In practice, a tool such as this can be used as a coaching aid for advisors looking to improve their client 
        relationships by identifying behaviors that are most strongly associated with trust and delegation decisions for their clients.
        
        The tool can be customized to view the results for all advisors in the firm or for a specific advisor.
        
        """

        st.space()
        # advisor selector (include an "All" option)
        advisor_options = ["All Advisors"] + sorted(dta["advisor"].dropna().unique().tolist())
        selected_advisor = st.selectbox("Filter by Advisor", options=advisor_options, index=0)

        # build filtered pandas and polars frames used by the rest of the page
        if selected_advisor != "All Advisors":
            dta_filtered = dta[dta["advisor"] == selected_advisor].copy()
        else:
            dta_filtered = dta.copy()

        dta_pl_filtered = pl.DataFrame(dta_filtered)

        """
        ## Part I: How Trust Relates to Delegation Decisions

        It's clear that clients in the sample largely trust their advisor. This is good news but not particularly
        surprising given that these clients have chosen to work with the advisor in the first place.
        
        When looking at delegation decisions, we see a much wider distribution of ratings where some people choose to 
        not delegate financial decisions to their advisors at all, whereas others delegate whenever possible, and many 
        are somewhere in between these extremes. 
        
        Moreover, although we do so a strong correlation between trust and delegation decisions, we do not see a 1 to 1
        relationship. **In other words, delegation decisions may require more than just trust in one's advisor.**
        
        """


        with st.container(width=TEXT_WIDTH):



            cols = st.columns([0.6, 0.4])

            with cols[0]:
                st.subheader("")
                st.plotly_chart(single_bar_plot(dta_filtered, 'Trust Rating', 'Trust Rating',
                                                'Very Little Trust', 'Complete Trust'))

            with cols[1]:
                st.space("medium")

                st.plotly_chart(box_plot(dta_filtered, 'Delegate Rating', 'Trust Rating',
                                         'Delegation<br>Rating', 'Trust Rating',
                                         'Complete<br>Delegation', 'Complete<br>Trust',
                                         'Understanding Relationship<br>Between Delegation and Trust'))


            help_text = (
                "This is calculated based on the delta between the Trust and Delegation Ratings."
            )

            st.space()

            rating_df = (
                dta_pl_filtered.select(pl.col('advisor').alias("Advisor"), pl.col('Delegate Rating').alias("Delegation Rating"),
                              pl.col('Trust Rating').alias("Trust Rating"),
                              pl.col('Unnamed: 0').alias("Client ID"))
                .filter(
                    pl.col('Delegation Rating').is_not_null() & pl.col('Trust Rating').is_not_null()
                )
                .with_columns(
                    delta=pl.col('Delegation Rating') / 7 - pl.col('Trust Rating') / 7,
                )
            ).sort(by="delta", descending=True)

            cols = st.columns(2, border=True)
            with cols[0]:
                st.subheader(
                    "Clients with low trust ratings but high delegation ratings",
                    help=help_text,
                )

                st.dataframe(
                    rating_df.select(pl.exclude("delta"))
                    .head(5)
                    .sort(by="Delegation Rating", descending=True),
                    height="stretch",
                )

            with cols[1]:
                st.subheader(
                    "Clients with high trust ratings but low delegation ratings",
                    help=help_text,
                )

                st.dataframe(
                    rating_df.select(pl.exclude("delta"))
                    .tail(5)
                    .sort(by='Trust Rating', descending=True),
                    height="stretch",
                )

            # -----------------------------------------------------------------------------
            # Part 2

            st.space("large")

            with st.container(width=TEXT_WIDTH):
                """
                ## Part II: How Frequency of Advisor Behaviors Are Associated with Trust
                Financial advisors perform a vast range of services for their clients, but, for the purposes of this research
                we narrowed our focus to 18 attributes/behaviors collected from a variety of sources.
                
                The following displays the frequency with which clients report experiencing each advisor behavior along with 
                their rating of trust in their advisor. The black line shows the linear trendline between the two variables.
                
                The correlation coefficient between the behavior frequency and trust rating is also displayed in the top right corner.
                
                Use the dropdown menu below to select a behavior to focus on. Your selection will update the graph and calculate
                the correlation between the behavior frequency and trust rating. 

                """

            numeric_cols_x = [
                'How often do you talk to your advisor', 'How often advisor communicates with you', 'How often advisor helps make progress to goals', 'How often advisor shows they understand your goals', 'How often advisor keeps their word', 'How often advisor acts in your best interest', 'How often advisor listens to your point of view, suggestions, needs, and acts accordingly', 'How often advisor shows they care about your future', 'How ofen advisor makes efforts to treat all customers equally', 'How often advisor takes the time to explain their decisions to you', 'How often advisor competently handles your requests', 'How often advisor shows they are knowledgeable', 'How often advisor shows they have the same concerns as you', 'How often advisor shows they have the same values as you', 'How often advisor acts as you would have acted', 'How often advisor treats you with respect', 'How often advisor shows consideration in their dealings with you', 'How often advisor provides advice that is suitable for you'
            ]

            numeric_cols_y = ['Trust Rating',
                              'Delegate Rating'
                              ]
            with st.container(width=TEXT_WIDTH):
                st.space()

                cols = st.columns(1)

                with cols[0]:
                    x_col = st.selectbox(
                        "X Axis (Behavior)", options=numeric_cols_x, index=4
                    )  # Default IMDB


                if not x_col:
                    st.info("Please select columns to visualize.")
                    st.stop()


                st.space()

                with st.container(height=SMALLPLOT_HEIGHT, border=False):
                    st.plotly_chart(scatter_plot(dta_filtered, x_col, 'Trust Rating',
                'Trust Rating',
                'Very<br>Frequently', 'Complete\nTrust',
                'Understanding Relationship Between Behaviors and Trust')


)
            numeric_cols = numeric_cols_x + numeric_cols_y
            r = pl.DataFrame(dta_filtered[numeric_cols].corr(method='pearson').reset_index(names="index"))
            # .loc[xvar,yvar]

            with st.container(width=TEXT_WIDTH):
                st.space()

                cols = st.columns(1)

                with cols[0]:
                    st.subheader(
                        "Trust: Top Correlated Behaviors",
                        help=help_text,
                    )

                    st.dataframe(
                        r.select(pl.col('Trust Rating').alias("Correlation with Trust"),
                                 pl.col('index').alias("Behavior")).filter(pl.col('Behavior')!= 'Trust Rating')
                        .sort(by='Correlation with Trust', descending=True)
                        .head(5),
                        height="stretch",
                    )



############################ ------------------
#########################---------------------------------

