import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="SWaT Аномалии", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color:#00f5d4;'>📊 Детекция аномалий: SWaT Dataset</h1>",
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    test_val = pd.read_excel("demo/SWaT_Dataset_Attack_v0_cropped.xlsx")
    test_val.columns = test_val.iloc[0, :]
    test_val.drop(0, inplace=True)
    test_val.columns = test_val.columns.str.replace(' ', '')

    test_val["Timestamp"] = pd.to_datetime(test_val["Timestamp"], dayfirst=True)

    attacks = pd.read_excel("demo/List_of_attacks_Final.xlsx")
    attacks = attacks[['Start Time', 'End Time', 'Attack Point']].dropna()

    attacks['Day'] = attacks['Start Time'].apply(lambda x: str(x).split()[0])
    attacks['End Time'] = pd.to_datetime(attacks['Day'] + ' ' + attacks['End Time'].astype(str))
    attacks['Start Time'] = pd.to_datetime(attacks['Start Time'], dayfirst=True)

    table_start = np.datetime64('2015-12-28T10:00:00')
    attacks['ind_st'] = (attacks['Start Time'] - table_start).dt.total_seconds().astype(int)
    attacks['ind_end'] = (attacks['End Time'] - table_start).dt.total_seconds().astype(int)

    attacks = attacks[:-5].drop(columns=["Day"])

    return test_val.reset_index(drop=True), attacks

test_val, attacks_list = load_data()

if "playing" not in st.session_state:
    st.session_state.playing = False
if "start_ind" not in st.session_state:
    st.session_state.start_ind = 0


# ind_start, ind_end = 0, 100000
# subset = test_val.iloc[ind_start:ind_end]
# timestamps = subset["Timestamp"]


features = st.multiselect(
    "Выберите фичи для отображения:",
    options=[col for col in test_val.columns if col not in ["Timestamp", "Normal/Attack"]][1:],
    default=[col for col in test_val.columns if col not in ["Timestamp", "Normal/Attack"]][1:6]
)

if "start_ind" not in st.session_state:
    st.session_state.start_ind = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

col1, col2 = st.columns([1, 1])
with col1:
    play_clicked = st.button("▶️ Play")
with col2:
    stop_clicked = st.button("⏹️ Stop")

if play_clicked:
    st.session_state.playing = True
if stop_clicked:
    st.session_state.playing = False


total_range = 100_000
step = 300
window_size = 6_000
max_ind = total_range - window_size

if not st.session_state.playing:
    st.session_state.start_ind = st.slider(
        "Выберите диапазон времени:",
        min_value=0,
        max_value=max_ind,
        value=st.session_state.start_ind,
        step=step
    )

plot_placeholder = st.empty()

def draw_plots(ind_start, ind_end):
    subset = test_val.iloc[ind_start:ind_end]
    timestamps = subset["Timestamp"]

    for feature in features:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=subset[feature],
            mode="lines",
            name=feature,
            line=dict(color="#00f5d4")
        ))

        for _, row in attacks_list.iterrows():
            a_st, a_end = row['ind_st'], row['ind_end']
            if a_st <= ind_end and a_end >= ind_start:
                x0 = max(a_st, ind_start) - ind_start
                x1 = min(a_end, ind_end) - ind_start
                x1 = min(x1, len(subset) - 1)
                fig.add_vrect(
                    x0=subset.iloc[x0]["Timestamp"],
                    x1=subset.iloc[x1]["Timestamp"],
                    fillcolor="red",
                    opacity=0.3,
                    line_width=0
                )

        fig.update_layout(
            title=feature,
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#1e1e1e",
            font_color="white",
            xaxis=dict(title="Time"),
            yaxis=dict(title=feature),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)


if st.session_state.playing:
    while st.session_state.playing:
        if st.session_state.start_ind >= max_ind:
            st.session_state.playing = False
            break

        ind_start = st.session_state.start_ind
        ind_end = ind_start + window_size

        with plot_placeholder.container():
            st.info("⏳ Воспроизведение в реальном времени...")
            draw_plots(ind_start, ind_end)

        st.session_state.start_ind += step
        st.rerun()
else:
    ind_start = st.session_state.start_ind
    ind_end = ind_start + window_size
    with plot_placeholder.container():
            draw_plots(ind_start, ind_end)