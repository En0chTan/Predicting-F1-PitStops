import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# APP CONFIGURATION & CONSTANTS
# ─────────────────────────────────────────────
"""
F1 Pit Stop Predictor - Streamlit Dashboard
Combines exploratory data analysis with machine learning models to predict pit stop decisions.
Models: Logistic Regression & Neural Network (MLP)
"""

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700&display=swap');

    .stApp { font-family: 'Titillium Web', sans-serif; }

    .main-header {
        font-size: 2.5rem; font-weight: 700; color: #ffffff;
        text-transform: uppercase; letter-spacing: 2px;
        border-left: 4px solid #e10600; padding-left: 1rem; margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem; color: #888888; letter-spacing: 1px; margin-bottom: 2rem;
        padding-left: 1.3rem;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #ffffff;
        text-transform: uppercase; letter-spacing: 1px;
        margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.4rem;
        border-bottom: 2px solid #e10600;
    }
    .info-banner {
        background: rgba(225, 6, 0, 0.08); border: 1px solid rgba(225,6,0,0.4);
        border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 0.8rem;
        font-size: 0.85rem; color: #cccccc;
    }
    [data-testid="stMetricValue"] { font-weight: 700; font-size: 1.6rem; }
    [data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: 1px; font-size: 0.75rem; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY LAYOUT DEFAULTS
# ─────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,26,46,0.8)',
    font=dict(family='Titillium Web', color='#ffffff'),
    title_font=dict(size=15),
    margin=dict(t=50, b=30, l=20, r=20),
    xaxis=dict(gridcolor='#2a2a4e'),
    yaxis=dict(gridcolor='#2a2a4e'),
)

COMPOUND_COLORS = {
    'SOFT': '#e10600',
    'MEDIUM': '#FFC107',
    'HARD': '#dddddd',
    'INTERMEDIATE': '#00BCD4',
    'WET': '#1565C0'
}

# ─────────────────────────────────────────────
# MOCK DATA (used when no file is uploaded)
# ─────────────────────────────────────────────
@st.cache_data
def generate_mock_data(n=3000):
    np.random.seed(42)
    drivers   = ['VER', 'HAM', 'LEC', 'SAI', 'PER', 'NOR', 'ALO', 'RUS', 'STR', 'OCO']
    compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    years     = [2022, 2023, 2024, 2025]
    races     = ['Bahrain Grand Prix', 'Monaco Grand Prix', 'British Grand Prix',
                 'Italian Grand Prix', 'Japanese Grand Prix', 'Dutch Grand Prix',
                 'Canadian Grand Prix', 'Miami Grand Prix']

    tyre_life       = np.random.randint(1, 50, n).astype(float)
    lap_number      = np.random.randint(1, 78, n)
    position        = np.random.randint(1, 21, n)
    stint           = np.random.randint(1, 5, n)
    race_progress   = np.round(lap_number / 78, 6)
    lap_time        = np.round(np.random.uniform(75, 110, n), 3)
    lap_time_delta  = np.round(np.random.uniform(-10, 10, n), 3)
    cum_deg         = np.round(np.cumsum(np.random.uniform(-2, 2, n)), 3)
    pos_change      = np.random.randint(-5, 6, n)

    compound_choice = np.random.choice(compounds, n, p=[0.30, 0.35, 0.25, 0.07, 0.03])

    pit_prob = (
        (tyre_life / 50) * 0.45 +
        (compound_choice == 'SOFT').astype(int) * 0.2 +
        (stint > 2).astype(int) * 0.1 +
        np.random.random(n) * 0.25
    )
    pit_stop     = (pit_prob > 0.55).astype(int)
    pit_next_lap = np.where(pit_stop == 1, 0, (pit_prob > 0.48).astype(int))

    return pd.DataFrame({
        'Driver': np.random.choice(drivers, n),
        'Compound': compound_choice,
        'Race': np.random.choice(races, n),
        'Year': np.random.choice(years, n),
        'PitStop': pit_stop,
        'LapNumber': lap_number,
        'Stint': stint,
        'TyreLife': tyre_life,
        'Position': position,
        'LapTime (s)': lap_time,
        'LapTime_Delta': lap_time_delta,
        'Cumulative_Degradation': cum_deg,
        'RaceProgress': race_progress,
        'Position_Change': pos_change,
        'PitNextLap': pit_next_lap,
    })

# ─────────────────────────────────────────────
# DATA VALIDATION & PREPROCESSING
# ─────────────────────────────────────────────
def validate_dataframe(df, required_cols):
    """
    Validate that a dataframe contains required columns and has data.
    
    Args:
        df (pd.DataFrame): The dataframe to validate
        required_cols (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "Dataset is empty."
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"
    
    if df.isnull().all().any():
        return False, "Some columns contain only null values."
    
    return True, ""

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏎️ Navigation")
    page = st.radio("Page", ["📊 EDA Dashboard", "🤖 Model Evaluation"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📂 Data Upload")
    train_file = st.file_uploader("Upload train.csv", type=["csv"])

    df = None
    data_status = None
    
    if train_file:
        try:
            df = pd.read_csv(train_file)
            required_cols = ['PitStop', 'TyreLife', 'Compound', 'Stint', 'Position', 
                           'LapNumber', 'Driver', 'Race', 'Year']
            is_valid, error_msg = validate_dataframe(df, required_cols)
            
            if is_valid:
                st.success(f"✓ Loaded {len(df):,} rows")
                data_status = "user_uploaded"
            else:
                st.error(f"✗ {error_msg}")
                df = None
                data_status = "invalid"
        except Exception as e:
            st.error(f"✗ Error reading file: {str(e)}")
            df = None
            data_status = "error"
    
    if df is None:
        df = generate_mock_data()
        st.info("Using synthetic mock data (3,000 samples)")
        data_status = "mock"

    st.markdown("---")
    st.markdown("### 🔽 Filters")

    sel_drivers   = st.multiselect("Driver",   sorted(df['Driver'].unique()),   default=sorted(df['Driver'].unique())[:8])
    sel_races     = st.multiselect("Race",     sorted(df['Race'].unique()),     default=sorted(df['Race'].unique()))
    sel_years     = st.multiselect("Year",     sorted(df['Year'].unique()),     default=sorted(df['Year'].unique()))
    sel_compounds = st.multiselect("Compound", sorted(df['Compound'].unique()), default=sorted(df['Compound'].unique()))
    stint_range   = st.slider("Stint Range",
                               int(df['Stint'].min()), int(df['Stint'].max()),
                               (int(df['Stint'].min()), int(df['Stint'].max())))

    st.markdown("---")
    data_source = "Real Data" if data_status == "user_uploaded" else "Mock Data"
    st.caption(f"Data: F1 Telemetry ({data_source}) | Models: LR & ANN")

# Apply filters (fall back to all if nothing selected)
fdf = df.copy()
if sel_drivers:   fdf = fdf[fdf['Driver'].isin(sel_drivers)]
if sel_races:     fdf = fdf[fdf['Race'].isin(sel_races)]
if sel_years:     fdf = fdf[fdf['Year'].isin(sel_years)]
if sel_compounds: fdf = fdf[fdf['Compound'].isin(sel_compounds)]
fdf = fdf[(fdf['Stint'] >= stint_range[0]) & (fdf['Stint'] <= stint_range[1])]

if fdf.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar filters.")
    st.stop()

# Label helper
fdf['PitStopLabel'] = fdf['PitStop'].map({0: 'No Pit Stop', 1: 'Pit Stop'})

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-header">F1 Pit Stop Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Analytics & Machine Learning Dashboard</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 1 — EDA DASHBOARD
# ══════════════════════════════════════════════
if page == "📊 EDA Dashboard":

    tabs = st.tabs(["🏁 Overview", "🛞 Tyre Analysis", "🏎️ Driver & Race",
                    "📍 Position & Stint", "⏱️ Lap Time & Degradation", "📅 Season Trends"])

    # ── TAB 1: OVERVIEW ──────────────────────────────────────
    with tabs[0]:
        st.markdown('<p class="section-title">Key Metrics</p>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Laps",      f"{len(fdf):,}")
        c2.metric("Pit Stop Rate",   f"{fdf['PitStop'].mean()*100:.1f}%")
        c3.metric("Unique Drivers",  fdf['Driver'].nunique())
        c4.metric("Unique Races",    fdf['Race'].nunique())
        c5.metric("Avg Tyre Life",   f"{fdf['TyreLife'].mean():.1f}")
        c6.metric("Avg Lap Time",    f"{fdf['LapTime (s)'].mean():.2f}s")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="info-banner">Overall share of laps where a pit stop occurred.</div>', unsafe_allow_html=True)
            pit_counts = fdf['PitStopLabel'].value_counts().reset_index()
            pit_counts.columns = ['Label', 'Count']
            fig = px.pie(pit_counts, values='Count', names='Label', hole=0.45,
                         color='Label',
                         color_discrete_map={'No Pit Stop': '#00C853', 'Pit Stop': '#D50000'},
                         title='Pit Stop Distribution')
            fig.update_layout(**LAYOUT, showlegend=True,
                              legend=dict(orientation='h', y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="info-banner">At which lap does a pit stop most commonly occur?</div>', unsafe_allow_html=True)
            lap_pit = fdf.groupby('LapNumber')['PitStop'].mean().reset_index()
            lap_pit.columns = ['LapNumber', 'PitRate']
            fig = px.area(lap_pit, x='LapNumber', y='PitRate',
                          title='Pit Stop Rate by Lap Number',
                          color_discrete_sequence=['#e10600'])
            fig.update_layout(**LAYOUT)
            fig.update_yaxes(tickformat='.0%', title='Pit Stop Rate')
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="info-banner">Which stint generates the most pit stops?</div>', unsafe_allow_html=True)
            stint_pit = fdf.groupby('Stint')['PitStop'].mean().reset_index()
            stint_pit.columns = ['Stint', 'PitRate']
            fig = px.bar(stint_pit, x='Stint', y='PitRate',
                         title='Pit Stop Rate by Stint',
                         color='PitRate', color_continuous_scale='Reds', text_auto='.1%')
            fig.update_layout(**LAYOUT, coloraxis_showscale=False)
            fig.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="info-banner">Comparing actual PitStop vs PitNextLap prediction in the data.</div>', unsafe_allow_html=True)
            if 'PitNextLap' in fdf.columns:
                compare = pd.DataFrame({
                    'Category': ['Pit Stop', 'Pit Next Lap'],
                    'Rate': [fdf['PitStop'].mean(), fdf['PitNextLap'].mean()]
                })
                fig = px.bar(compare, x='Category', y='Rate',
                             title='PitStop vs PitNextLap Rate',
                             color='Category',
                             color_discrete_map={'Pit Stop': '#e10600', 'Pit Next Lap': '#FFC107'},
                             text_auto='.1%')
                fig.update_layout(**LAYOUT, showlegend=False)
                fig.update_yaxes(tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: TYRE ANALYSIS ─────────────────────────────────
    with tabs[1]:
        st.markdown('<p class="section-title">Tyre Performance Analysis</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="info-banner">Share of laps run on each compound type.</div>', unsafe_allow_html=True)
            comp_counts = fdf['Compound'].value_counts().reset_index()
            comp_counts.columns = ['Compound', 'Count']
            fig = px.pie(comp_counts, values='Count', names='Compound',
                         title='Tyre Compound Usage',
                         color='Compound', color_discrete_map=COMPOUND_COLORS)
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="info-banner">How many laps does each compound typically survive?</div>', unsafe_allow_html=True)
            fig = px.box(fdf, x='Compound', y='TyreLife',
                         title='Tyre Life by Compound',
                         color='Compound', color_discrete_map=COMPOUND_COLORS)
            fig.update_layout(**LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="info-banner">Tyre life is the strongest predictor — how does it differ at pit decision?</div>', unsafe_allow_html=True)
            fig = px.box(fdf, x='PitStopLabel', y='TyreLife',
                         title='Tyre Life vs Pit Stop Decision',
                         color='PitStopLabel',
                         color_discrete_map={'No Pit Stop': '#00C853', 'Pit Stop': '#D50000'})
            fig.update_layout(**LAYOUT, showlegend=False)
            fig.update_xaxes(title='')
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="info-banner">Pit stop probability rises sharply as tyre age increases.</div>', unsafe_allow_html=True)
            tyre_prob = fdf.groupby('TyreLife')['PitStop'].mean().reset_index()
            tyre_prob.columns = ['TyreLife', 'Probability']
            fig = px.line(tyre_prob, x='TyreLife', y='Probability',
                          title='Pit Stop Probability by Tyre Life',
                          color_discrete_sequence=['#e10600'])
            fig.update_layout(**LAYOUT)
            fig.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="info-banner">Tyre compound usage has shifted across seasons — reflecting rule and strategy changes.</div>', unsafe_allow_html=True)
        compound_year = fdf.groupby(['Year', 'Compound']).size().reset_index(name='Count')
        fig = px.bar(compound_year, x='Year', y='Count', color='Compound',
                     barmode='stack', title='Tyre Compound Usage by Year',
                     color_discrete_map=COMPOUND_COLORS)
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: DRIVER & RACE ─────────────────────────────────
    with tabs[2]:
        st.markdown('<p class="section-title">Driver & Race Analysis</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="info-banner">Top 10 drivers by total pit stops made.</div>', unsafe_allow_html=True)
            drv_pits = (fdf.groupby('Driver')['PitStop'].sum()
                        .sort_values().tail(10).reset_index())
            drv_pits.columns = ['Driver', 'PitStops']
            fig = px.bar(drv_pits, x='PitStops', y='Driver', orientation='h',
                         title='Top 10 Drivers by Pit Stop Frequency',
                         color='PitStops', color_continuous_scale='Reds', text_auto=True)
            fig.update_layout(**LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="info-banner">Drivers who run longer on a set of tyres before pitting.</div>', unsafe_allow_html=True)
            drv_tyre = (fdf.groupby('Driver')['TyreLife'].mean()
                        .sort_values().tail(10).reset_index())
            drv_tyre.columns = ['Driver', 'AvgTyreLife']
            fig = px.bar(drv_tyre, x='AvgTyreLife', y='Driver', orientation='h',
                         title='Top 10 Drivers – Longest Avg Tyre Life',
                         color_discrete_sequence=['orange'], text_auto='.1f')
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="info-banner">Which compound does each driver prefer?</div>', unsafe_allow_html=True)
            drv_comp = pd.crosstab(fdf['Driver'], fdf['Compound'])
            fig = px.imshow(drv_comp, title='Driver vs Tyre Compound Heatmap',
                            color_continuous_scale='magma', aspect='auto')
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="info-banner">Which races have the highest pit stop rates?</div>', unsafe_allow_html=True)
            race_pit = (fdf.groupby('Race')['PitStop'].mean()
                        .sort_values().reset_index())
            race_pit.columns = ['Race', 'PitRate']
            fig = px.bar(race_pit, x='PitRate', y='Race', orientation='h',
                         title='Pit Stop Rate by Race',
                         color='PitRate', color_continuous_scale='RdYlGn_r',
                         text_auto='.1%')
            fig.update_layout(**LAYOUT, coloraxis_showscale=False)
            fig.update_xaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="info-banner">How has each driver\'s pit stop rate changed year over year?</div>', unsafe_allow_html=True)
        top5 = fdf['Driver'].value_counts().head(5).index.tolist()
        drv_year = (fdf[fdf['Driver'].isin(top5)]
                    .groupby(['Year', 'Driver'])['PitStop'].mean().reset_index())
        drv_year.columns = ['Year', 'Driver', 'PitRate']
        fig = px.line(drv_year, x='Year', y='PitRate', color='Driver',
                      title='Pit Stop Rate by Year (Top 5 Drivers)', markers=True)
        fig.update_layout(**LAYOUT)
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: POSITION & STINT ──────────────────────────────
    with tabs[3]:
        st.markdown('<p class="section-title">Position & Stint Dynamics</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="info-banner">Do drivers in lower positions pit more frequently?</div>', unsafe_allow_html=True)
            fig = px.box(fdf, x='PitStopLabel', y='Position',
                         title='Race Position vs Pit Stop',
                         color='PitStopLabel',
                         color_discrete_map={'No Pit Stop': '#00C853', 'Pit Stop': '#D50000'})
            fig.update_layout(**LAYOUT, showlegend=False)
            fig.update_yaxes(autorange='reversed')
            fig.update_xaxes(title='')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="info-banner">Which stint sees the most activity?</div>', unsafe_allow_html=True)
            fig = px.histogram(fdf, x='Stint', color='Stint',
                               title='Stint Distribution',
                               color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_layout(**LAYOUT, showlegend=False)
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="info-banner">Does running on older tyres hurt track position?</div>', unsafe_allow_html=True)
            fig = px.scatter(fdf, x='TyreLife', y='Position', color='Compound',
                             title='Tyre Life vs Race Position',
                             color_discrete_map=COMPOUND_COLORS, opacity=0.5)
            fig.update_layout(**LAYOUT)
            fig.update_yaxes(autorange='reversed')
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="info-banner">Position changes per stint — are early stints more dynamic?</div>', unsafe_allow_html=True)
            if 'Position_Change' in fdf.columns:
                fig = px.box(fdf, x='Stint', y='Position_Change',
                             title='Position Change Distribution by Stint',
                             color='Stint',
                             color_discrete_sequence=px.colors.sequential.Plasma)
                fig.update_layout(**LAYOUT, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="info-banner">Correlation between all numeric features and the pit stop target.</div>', unsafe_allow_html=True)
        num_cols = ['TyreLife', 'Position', 'Stint', 'LapNumber',
                    'RaceProgress', 'Position_Change', 'PitStop']
        num_cols = [c for c in num_cols if c in fdf.columns]
        corr = fdf[num_cols].corr()
        fig = px.imshow(corr, title='Feature Correlation Heatmap',
                        color_continuous_scale='RdBu', text_auto='.2f', aspect='auto')
        fig.update_layout(**{k: v for k, v in LAYOUT.items()
                             if k not in ['xaxis', 'yaxis']})
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: LAP TIME & DEGRADATION ───────────────────────
    with tabs[4]:
        st.markdown('<p class="section-title">Lap Time & Tyre Degradation</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="info-banner">Lap time distribution — are pit stop laps slower?</div>', unsafe_allow_html=True)
            lt_df = fdf[fdf['LapTime (s)'] < 200].copy()  # remove outlier safety car laps
            fig = px.box(lt_df, x='PitStopLabel', y='LapTime (s)',
                         title='Lap Time vs Pit Stop',
                         color='PitStopLabel',
                         color_discrete_map={'No Pit Stop': '#00C853', 'Pit Stop': '#D50000'})
            fig.update_layout(**LAYOUT, showlegend=False)
            fig.update_xaxes(title='')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="info-banner">Average lap time per compound — which is fastest?</div>', unsafe_allow_html=True)
            lt_comp = (lt_df.groupby('Compound')['LapTime (s)']
                       .mean().sort_values().reset_index())
            lt_comp.columns = ['Compound', 'AvgLapTime']
            fig = px.bar(lt_comp, x='Compound', y='AvgLapTime',
                         title='Average Lap Time by Compound',
                         color='Compound', color_discrete_map=COMPOUND_COLORS,
                         text_auto='.2f')
            fig.update_layout(**LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="info-banner">Lap time delta — positive means slower than previous lap.</div>', unsafe_allow_html=True)
            if 'LapTime_Delta' in fdf.columns:
                delta_df = fdf[fdf['LapTime_Delta'].between(-30, 30)]
                fig = px.histogram(delta_df, x='LapTime_Delta', nbins=40,
                                   title='Lap Time Delta Distribution',
                                   color_discrete_sequence=['#e10600'],
                                   marginal='violin')
                fig.update_layout(**LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="info-banner">Cumulative degradation grows with tyre age — a key pit stop trigger.</div>', unsafe_allow_html=True)
            if 'Cumulative_Degradation' in fdf.columns:
                deg_df = fdf.groupby('TyreLife')['Cumulative_Degradation'].mean().reset_index()
                fig = px.line(deg_df, x='TyreLife', y='Cumulative_Degradation',
                              title='Avg Cumulative Degradation vs Tyre Life',
                              color_discrete_sequence=['#FFC107'])
                fig.update_layout(**LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="info-banner">Does race progress (0=start, 1=end) influence how degradation accumulates?</div>', unsafe_allow_html=True)
        if 'RaceProgress' in fdf.columns and 'Cumulative_Degradation' in fdf.columns:
            sample = fdf.sample(min(1500, len(fdf)), random_state=1)
            fig = px.scatter(sample, x='RaceProgress', y='Cumulative_Degradation',
                             color='Compound', opacity=0.5,
                             title='Race Progress vs Cumulative Degradation',
                             color_discrete_map=COMPOUND_COLORS)
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 6: SEASON TRENDS ─────────────────────────────────
    with tabs[5]:
        st.markdown('<p class="section-title">Season & Race Trends</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="info-banner">Has the overall pit stop rate changed across seasons?</div>', unsafe_allow_html=True)
            yr_pit = fdf.groupby('Year')['PitStop'].mean().reset_index()
            yr_pit.columns = ['Year', 'PitRate']
            fig = px.line(yr_pit, x='Year', y='PitRate',
                          title='Pit Stop Rate Trend by Year',
                          markers=True, color_discrete_sequence=['#e10600'])
            fig.update_layout(**LAYOUT)
            fig.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="info-banner">How many laps are recorded per season?</div>', unsafe_allow_html=True)
            yr_counts = fdf['Year'].value_counts().sort_index().reset_index()
            yr_counts.columns = ['Year', 'Count']
            fig = px.bar(yr_counts, x='Year', y='Count',
                         title='Laps Recorded by Year',
                         color='Count', color_continuous_scale='plasma', text_auto=True)
            fig.update_layout(**LAYOUT, coloraxis_showscale=False)
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="info-banner">Average lap time trend — are cars getting faster each season?</div>', unsafe_allow_html=True)
        lt_yr = (fdf[fdf['LapTime (s)'] < 200]
                 .groupby('Year')['LapTime (s)'].mean().reset_index())
        fig = px.line(lt_yr, x='Year', y='LapTime (s)',
                      title='Average Lap Time by Year', markers=True,
                      color_discrete_sequence=['#00BCD4'])
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 — MODEL EVALUATION
# ══════════════════════════════════════════════
elif page == "🤖 Model Evaluation":

    st.markdown('<p class="section-title">Model Selection & Pit Stop Prediction</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-banner">Features used: TyreLife, Compound, Stint, Position, LapNumber, Driver, Race, Year &nbsp;|&nbsp; Target: PitStop &nbsp;|&nbsp; Split: 75/25</div>', unsafe_allow_html=True)

    # Model selection
    model_choice = st.radio("Select Model:", ["Logistic Regression", "Neural Network (ANN)"], horizontal=True)

    with st.spinner("Preparing data and training model..."):

        FEATURES = ['TyreLife', 'Compound', 'Stint', 'Position', 'LapNumber',
                    'Driver', 'Race', 'Year']
        # Add extra features if they exist in the dataset
        for col in ['RaceProgress', 'Cumulative_Degradation', 'LapTime_Delta',
                    'Position_Change']:
            if col in fdf.columns:
                FEATURES.append(col)

        TARGET = 'PitStop'
        model_df = fdf[FEATURES + [TARGET]].dropna().copy()

        # Encode categoricals
        le_dict = {}
        for col in ['Compound', 'Driver', 'Race']:
            if col in model_df.columns:
                le = LabelEncoder()
                model_df[col] = le.fit_transform(model_df[col].astype(str))
                le_dict[col] = le

        X = model_df[FEATURES]
        y = model_df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Scale features for better model performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_choice == "Logistic Regression":
            clf = LogisticRegression(solver="newton-cg", random_state=42, max_iter=1000)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        else:  # Neural Network
            clf = MLPClassifier(
                hidden_layer_sizes=(50,),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)

    # ── Metric Cards ─────────────────────────────────────────
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc:.2%}")
    m2.metric("Precision", f"{prec:.2%}")
    m3.metric("Recall",    f"{rec:.2%}")
    m4.metric("F1 Score",  f"{f1:.2%}")
    m5.metric("ROC AUC",   f"{auc:.4f}")

    st.markdown("---")

    # ── Confusion Matrix & ROC ────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="info-banner">Confusion matrix — how many pit/no-pit laps are correctly classified?</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True,
                        color_continuous_scale='Blues',
                        labels=dict(x='Predicted', y='Actual'),
                        x=['No Pit Stop', 'Pit Stop'],
                        y=['No Pit Stop', 'Pit Stop'],
                        title='Confusion Matrix')
        fig.update_layout(**{k: v for k, v in LAYOUT.items()
                             if k not in ['xaxis', 'yaxis', 'plot_bgcolor']},
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="info-banner">ROC curve — AUC closer to 1.0 means stronger predictions.</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                  name=f'ROC (AUC = {auc:.3f})',
                                  line=dict(color='#e10600', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  name='Random Classifier',
                                  line=dict(color='#888888', dash='dash')))
        fig.update_layout(title='ROC Curve',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          **{k: v for k, v in LAYOUT.items()
                             if k not in ['xaxis', 'yaxis']})
        fig.update_xaxes(range=[0, 1], gridcolor='#2a2a4e')
        fig.update_yaxes(range=[0, 1], gridcolor='#2a2a4e')
        st.plotly_chart(fig, use_container_width=True)

    # ── Classification Report ─────────────────────────────────
    st.markdown('<p class="section-title">Classification Report</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-banner">Per-class breakdown of precision, recall, and F1 score.</div>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred,
                                    target_names=['No Pit Stop', 'Pit Stop'],
                                    output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(
        report_df.style.background_gradient(cmap='RdYlGn',
                                             subset=['precision', 'recall', 'f1-score']),
        use_container_width=True
    )

    # ── Feature Importance ────────────────────────────────────
    st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)
    
    if model_choice == "Logistic Regression":
        st.markdown('<div class="info-banner">Absolute coefficient magnitude — larger value means stronger influence on prediction.</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': np.abs(clf.coef_[0])
        }).sort_values('Importance', ascending=True)

        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance (Absolute Coefficients)',
                     color='Importance', color_continuous_scale='Reds')
        fig.update_layout(**LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="info-banner">Neural networks do not have directly interpretable feature coefficients. Importance derived from permutation analysis or model structure.</div>', unsafe_allow_html=True)
        st.info("Feature importance for neural networks requires more advanced techniques like SHAP or permutation importance. Consider using the Logistic Regression model for interpretability.")
