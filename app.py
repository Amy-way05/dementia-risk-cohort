import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dementia Risk Cohort · NHANES",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design tokens ─────────────────────────────────────────────────────────────
NAV = "#1F3864"
RED = "#C0504D"
GRY = "#7F8C8D"
GLD = "#E8A838"

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #F4F6FA; }
  [data-testid="stSidebar"] { background: #1F3864; }
  [data-testid="stSidebar"] * { color: white !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stMultiSelect label { color: #AAC4E0 !important; }
  .metric-card {
    background: white; border-radius: 10px; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center;
  }
  .metric-val { font-size: 2.4rem; font-weight: 800; margin: 4px 0; }
  .metric-lbl { font-size: 0.78rem; color: #888; text-transform: uppercase;
                letter-spacing: 0.06em; }
  .metric-sub { font-size: 0.72rem; color: #aaa; margin-top: 2px; }
  .section-header {
    font-size: 0.72rem; font-weight: 700; color: #1F3864;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 2px solid #1F3864; padding-bottom: 4px;
    margin: 28px 0 16px 0;
  }
  .insight-box {
    background: white; border-left: 4px solid #1F3864;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    margin: 12px 0; font-size: 0.85rem; color: #333;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: white; border-radius: 8px 8px 0 0;
    padding: 10px 20px; font-weight: 600; color: #555;
  }
  .stTabs [aria-selected="true"] {
    background: #1F3864 !important; color: white !important;
  }
  h1 { color: #1F3864 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load & process data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        "../forecasting/data/nhanes_pooled.csv"
    )
    df = df.rename(columns={
        'RIDAGEYR':'age','RIAGENDR':'gender','RIDRETH3':'race',
        'INDFMPIR':'income_pir','recall_score':'recall_score',
        'impaired':'impaired','wave_year':'wave_year'
    })
    df = df.dropna(subset=['age','income_pir','recall_score']).copy()
    df['age'] = df['age'].astype(int)
    df['gender_label'] = df['gender'].map({1:'Male',2:'Female'})
    df['race_label'] = df['race'].map({
        1:'Mexican American',2:'Other Hispanic',
        3:'Non-Hispanic White',4:'Non-Hispanic Black',6:'Non-Hispanic Asian'
    }).fillna('Other')
    df['wave_label'] = df['wave_year'].map({2012:'2011-12',2014:'2013-14'})
    df['age_grp'] = pd.cut(df['age'],
        bins=[59,64,69,74,79,84,200],
        labels=['60-64','65-69','70-74','75-79','80-84','85+'])
    return df

df_full = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 🧠 Dementia Risk\n### NHANES Cohort Explorer")
st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

age_range = st.sidebar.slider("Age range", 60, 95, (60, 95))
waves = st.sidebar.multiselect("NHANES Wave",
    options=['2011-12','2013-14'], default=['2011-12','2013-14'])
genders = st.sidebar.multiselect("Sex",
    options=['Male','Female'], default=['Male','Female'])
income_range = st.sidebar.slider("Income-to-Poverty Ratio", 0.0, 5.0, (0.0, 5.0), 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown(
    "Real NHANES cognitive functioning data (CFQ module), "
    "2011-12 and 2013-14 cycles. n=3,126 adults aged 60+."
)
st.sidebar.markdown("*Amrutha Ravikumar · MPS Analytics · Northeastern Roux Institute*")

# Apply filters
df = df_full[
    (df_full['age'] >= age_range[0]) &
    (df_full['age'] <= age_range[1]) &
    (df_full['wave_label'].isin(waves)) &
    (df_full['gender_label'].isin(genders)) &
    (df_full['income_pir'] >= income_range[0]) &
    (df_full['income_pir'] <= income_range[1])
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Cognitive Impairment Risk in Older Adults")
st.markdown(
    "**Data source:** CDC NHANES Cognitive Functioning Questionnaire (CFQ), "
    "cycles 2011-12 and 2013-14 · Adults aged 60+ · "
    f"Filtered cohort: **{len(df):,}** participants"
)

# ── KPI row ───────────────────────────────────────────────────────────────────
if len(df) == 0:
    st.warning("No data matches current filters. Adjust the sidebar filters.")
    st.stop()

imp_rate = df['impaired'].mean() * 100
mean_age = df['age'].mean()
mean_recall = df['recall_score'].mean()
n_waves = df['wave_year'].nunique()

c1,c2,c3,c4 = st.columns(4)
for col, val, lbl, sub, color in [
    (c1, f"{len(df):,}",       "Cohort Size",        "Filtered participants",        NAV),
    (c2, f"{imp_rate:.1f}%",   "Impairment Rate",    "Recall score ≤ 4 threshold",   RED),
    (c3, f"{mean_age:.1f} yrs","Mean Age",           "Range 60–95",                  "#2E4057"),
    (c4, f"{mean_recall:.2f}", "Mean Recall Score",  "Out of 7 (lower = worse)",     GLD),
]:
    col.markdown(f"""<div class="metric-card">
        <div class="metric-lbl">{lbl}</div>
        <div class="metric-val" style="color:{color}">{val}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Cohort Profile",
    "📈  Prevalence Trends",
    "🔬  Predictive Model",
    "🧪  Experiment Design"
])

AXIS = dict(showgrid=True, gridcolor='#EEE', linecolor='#CCC',
            showline=True, zeroline=False,
            tickfont=dict(family='Arial',size=11,color='#666'),
            title_font=dict(family='Arial',size=12,color='#444'))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — COHORT PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Impairment by Demographics</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Age group impairment rate
    with col_a:
        ags = df.groupby('age_grp', observed=True).agg(
            n=('impaired','count'), imp=('impaired','mean')).reset_index()
        ags['pct'] = (ags.imp * 100).round(1)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(ags.age_grp), y=list(ags.pct),
            marker_color=[RED if v > imp_rate else NAV for v in ags.pct],
            text=[f"{v}%<br><span style='font-size:10px'>n={n}</span>"
                  for v,n in zip(ags.pct, ags.n)],
            textposition='outside', textfont=dict(size=10,family='Arial')))
        fig.add_hline(y=imp_rate, line_dash='dash', line_color=GRY,
                      annotation_text=f"Overall: {imp_rate:.1f}%",
                      annotation_position="right")
        fig.update_layout(
            title=dict(text="Impairment Rate by Age Group",
                       font=dict(color=NAV,size=14,family='Arial')),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(**AXIS, title="Age Group"),
            yaxis=dict(**AXIS, title="Impairment Rate (%)", range=[0, ags.pct.max()+12]),
            height=340, margin=dict(t=50,b=40,l=50,r=20),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Recall score distribution
    with col_b:
        fig = go.Figure()
        for imp_val, color, name in [(0,NAV,'Not Impaired'),(1,RED,'Cognitively Impaired')]:
            grp = df[df['impaired']==imp_val]['recall_score']
            fig.add_trace(go.Histogram(x=grp, name=name,
                marker_color=color, opacity=0.75,
                xbins=dict(start=0,end=7,size=1),
                histnorm='percent'))
        fig.add_vline(x=4.5, line_dash='dash', line_color='#333',
                      annotation_text="Threshold (≤4 = impaired)",
                      annotation_position="top left",
                      annotation_font=dict(size=10))
        fig.update_layout(
            title=dict(text="Recall Score Distribution",
                       font=dict(color=NAV,size=14,family='Arial')),
            barmode='overlay',
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(**AXIS, title="Recall Score (0-7)"),
            yaxis=dict(**AXIS, title="Percent of Group (%)"),
            height=340, margin=dict(t=50,b=40,l=50,r=20),
            legend=dict(x=0.02,y=0.97,bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#EEE',borderwidth=1))
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Sex breakdown
    with col_c:
        sex_df = df.groupby(['gender_label','impaired']).size().reset_index(name='n')
        sex_tot = sex_df.groupby('gender_label')['n'].transform('sum')
        sex_df['pct'] = (sex_df['n']/sex_tot*100).round(1)
        sex_imp = sex_df[sex_df['impaired']==1]
        fig = go.Figure(go.Bar(
            x=list(sex_imp.gender_label), y=list(sex_imp.pct),
            marker_color=[NAV, RED],
            text=[f"{v}%" for v in sex_imp.pct],
            textposition='outside', textfont=dict(size=12,family='Arial')))
        fig.update_layout(
            title=dict(text="Impairment Rate by Sex",
                       font=dict(color=NAV,size=14,family='Arial')),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(**AXIS, title="Sex"),
            yaxis=dict(**AXIS, title="Impairment Rate (%)", range=[0,55]),
            height=300, margin=dict(t=50,b=40,l=50,r=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Income vs recall scatter
    with col_d:
        samp = df.sample(min(600,len(df)), random_state=42)
        fig = go.Figure()
        for imp_val,color,name in [(0,NAV,'Not Impaired'),(1,RED,'Impaired')]:
            g = samp[samp['impaired']==imp_val]
            fig.add_trace(go.Scatter(
                x=list(g['income_pir']), y=list(g['recall_score']),
                mode='markers', name=name,
                marker=dict(color=color,size=4,opacity=0.35)))
        # trend line
        m,b,*_ = stats.linregress(samp['income_pir'].fillna(0), samp['recall_score'])
        xl = np.linspace(0,5,80)
        fig.add_trace(go.Scatter(x=xl, y=m*xl+b, mode='lines',
            line=dict(color='#333',width=2,dash='dot'),
            name=f'Trend (β={m:.2f})', showlegend=True))
        fig.update_layout(
            title=dict(text="Income vs Recall Score",
                       font=dict(color=NAV,size=14,family='Arial')),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(**AXIS, title="Income-to-Poverty Ratio"),
            yaxis=dict(**AXIS, title="Recall Score"),
            height=300, margin=dict(t=50,b=40,l=50,r=20),
            legend=dict(x=0.02,y=0.97,bgcolor='rgba(255,255,255,0.8)'))
        st.plotly_chart(fig, use_container_width=True)

    # Key insight
    hi_age = ags.loc[ags.pct.idxmax(), 'age_grp']
    hi_pct = ags.pct.max()
    st.markdown(f"""<div class="insight-box">
        <b>Key finding:</b> Impairment prevalence rises sharply with age —
        reaching <b>{hi_pct:.1f}%</b> in the <b>{hi_age}</b> group.
        Higher income-to-poverty ratios are associated with better recall scores
        (β={stats.linregress(df['income_pir'].fillna(0), df['recall_score'])[0]:.2f} per unit),
        consistent with socioeconomic gradients in cognitive health literature.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREVALENCE TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Multi-Wave NHANES Prevalence Analysis</div>',
                unsafe_allow_html=True)

    wave_stats = df_full.groupby('wave_year').agg(
        n=('impaired','count'), imp=('impaired','mean')).reset_index()
    wave_stats['pct'] = (wave_stats.imp*100).round(1)

    def bootstrap_ci(x, n_boot=2000, ci=95):
        boots = [np.mean(np.random.choice(x, size=len(x), replace=True))
                 for _ in range(n_boot)]
        lo = np.percentile(boots, (100-ci)/2)
        hi = np.percentile(boots, 100-(100-ci)/2)
        return lo*100, hi*100

    cis = []
    for wy in [2012,2014]:
        x = df_full[df_full.wave_year==wy]['impaired'].values
        lo,hi = bootstrap_ci(x)
        cis.append({'wave_year':wy,'lo':lo,'hi':hi})
    ci_df = pd.DataFrame(cis)
    wave_stats = wave_stats.merge(ci_df, on='wave_year')

    col_a, col_b = st.columns([2,1])
    with col_a:
        fig = go.Figure()
        # CI band
        fig.add_trace(go.Scatter(
            x=list(wave_stats.wave_year)+list(wave_stats.wave_year[::-1]),
            y=list(wave_stats.hi)+list(wave_stats.lo[::-1]),
            fill='toself', fillcolor='rgba(31,56,100,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI', showlegend=True))
        # Observed points
        fig.add_trace(go.Scatter(
            x=list(wave_stats.wave_year), y=list(wave_stats.pct),
            mode='markers+lines+text',
            text=[f"  {v}%<br>  n={n:,}" for v,n in
                  zip(wave_stats.pct, wave_stats.n)],
            textposition='top right',
            textfont=dict(color=NAV,size=12,family='Arial'),
            marker=dict(color=NAV,size=16,line=dict(color='white',width=2)),
            line=dict(color=NAV,width=3),
            name='Observed (NHANES)',
            error_y=dict(type='data',symmetric=False,
                         array=list(wave_stats.hi-wave_stats.pct),
                         arrayminus=list(wave_stats.pct-wave_stats.lo),
                         visible=True,color=NAV,thickness=1.5,width=10)))
        # Illustrative trend
        fig.add_trace(go.Scatter(
            x=[2014,2016], y=[23.4,14.4], mode='lines',
            name='Linear extrapolation (illustrative)',
            line=dict(color=GRY,width=1.5,dash='dot')))
        fig.add_vline(x=2014.3, line_dash='dash', line_color='#CCC', line_width=1)
        fig.add_annotation(x=2013,y=36,
            text="<b>8.8 pp decline</b><br>2012 → 2014",
            showarrow=False, bgcolor='white', bordercolor=NAV,
            borderwidth=1, borderpad=6,
            font=dict(size=12,color=NAV,family='Arial'))
        fig.add_annotation(x=2015.2,y=28,
            text="CFQ module<br>discontinued<br>after 2013-14",
            showarrow=True, ax=-40,ay=20,
            arrowcolor=GRY, arrowwidth=1,
            font=dict(size=10,color=GRY,family='Arial'))
        fig.update_layout(
            title=dict(
                text="Cognitive Impairment Prevalence · NHANES 2011-2014",
                font=dict(color=NAV,size=15,family='Arial')),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(**AXIS, title="NHANES Survey Year",
                       range=[2010.5,2017],
                       tickvals=[2012,2014,2016],
                       ticktext=['2011-12','2013-14','2015-16 (no data)']),
            yaxis=dict(**AXIS, title="Impairment Prevalence (%)",
                       range=[0,42]),
            height=420, margin=dict(t=60,b=50,l=60,r=30),
            legend=dict(x=0.65,y=0.95,bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='#EEE',borderwidth=1))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Wave Summary</div>',
                    unsafe_allow_html=True)
        for _,row in wave_stats.iterrows():
            wave_lbl = '2011-12' if row.wave_year==2012 else '2013-14'
            delta = ""
            if row.wave_year == 2014:
                prev = wave_stats[wave_stats.wave_year==2012]['pct'].values[0]
                chg = row.pct - prev
                delta = f"<span style='color:{RED};font-weight:700'>{chg:+.1f} pp</span>"
            st.markdown(f"""<div class="insight-box">
                <b>NHANES {wave_lbl}</b><br>
                n = {int(row.n):,} participants<br>
                Prevalence: <b style='color:{NAV}'>{row.pct:.1f}%</b> {delta}<br>
                95% CI: [{row.lo:.1f}%, {row.hi:.1f}%]
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="insight-box">
            <b>Statistical note</b><br>
            Bootstrap CIs computed with 2,000 resamples.
            The 8.8 pp decline is consistent with published
            literature on cohort-level improvements in cognitive
            health driven by rising educational attainment
            and better cardiovascular risk management.
        </div>""", unsafe_allow_html=True)

    # By subgroup
    st.markdown('<div class="section-header">Prevalence by Subgroup · Both Waves</div>',
                unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        rdf = df_full.groupby(['wave_year','age_grp'], observed=True).agg(
            pct=('impaired','mean')).reset_index()
        rdf['pct'] = (rdf.pct*100).round(1)
        rdf['wave'] = rdf.wave_year.map({2012:'2011-12',2014:'2013-14'})
        fig = px.line(rdf, x='age_grp', y='pct', color='wave',
            color_discrete_map={'2011-12':NAV,'2013-14':RED},
            markers=True, labels={'age_grp':'Age Group','pct':'Impairment (%)','wave':'Wave'})
        fig.update_traces(line_width=2.5, marker_size=8)
        fig.update_layout(
            title=dict(text="Age-Stratified Trends",
                       font=dict(color=NAV,size=14,family='Arial')),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(**AXIS), yaxis=dict(**AXIS,range=[0,70]),
            height=320, margin=dict(t=50,b=40,l=50,r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        sdf = df_full.groupby(['wave_year','gender_label']).agg(
            pct=('impaired','mean')).reset_index()
        sdf['pct'] = (sdf.pct*100).round(1)
        sdf['wave'] = sdf.wave_year.map({2012:'2011-12',2014:'2013-14'})
        fig = px.bar(sdf, x='gender_label', y='pct', color='wave', barmode='group',
            color_discrete_map={'2011-12':NAV,'2013-14':RED},
            labels={'gender_label':'Sex','pct':'Impairment (%)','wave':'Wave'})
        fig.update_layout(
            title=dict(text="Sex-Stratified Trends",
                       font=dict(color=NAV,size=14,family='Arial')),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(**AXIS), yaxis=dict(**AXIS,range=[0,55]),
            height=320, margin=dict(t=50,b=40,l=50,r=20))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Logistic Regression · Predictors of Cognitive Impairment</div>',
                unsafe_allow_html=True)

    model_df = df[['impaired','age','gender','income_pir','wave_year']].dropna().copy()
    model_df['male'] = (model_df['gender']==1).astype(int)
    model_df['wave_2014'] = (model_df['wave_year']==2014).astype(int)

    if len(model_df) < 50:
        st.warning("Not enough data for model. Adjust filters.")
    else:
        model = smf.logit(
            'impaired ~ age + male + income_pir + wave_2014',
            data=model_df).fit(disp=0)

        res = pd.DataFrame({
            'OR': np.exp(model.params),
            'lo': np.exp(model.conf_int()[0]),
            'hi': np.exp(model.conf_int()[1]),
            'p':  model.pvalues
        }).drop('Intercept')

        col_a, col_b = st.columns([3,2])

        with col_a:
            lbls = {'age':'Age (per year)','male':'Male sex',
                    'income_pir':'Income-to-Poverty Ratio','wave_2014':'NHANES 2013-14 wave'}
            yl = [lbls.get(i,i) for i in res.index]
            fc = [RED if p<0.05 else GRY for p in res.p]
            sig = ['★ p<0.05' if p<0.05 else 'ns' for p in res.p]

            fig = go.Figure()
            fig.add_vline(x=1.0, line_dash='dash', line_color=RED,
                          line_width=1.5, annotation_text="OR = 1.0 (no effect)",
                          annotation_position="top right",
                          annotation_font=dict(size=10,color=RED))
            fig.add_trace(go.Scatter(
                x=list(res.OR), y=yl, mode='markers',
                error_x=dict(type='data', symmetric=False,
                             array=list(res.hi-res.OR),
                             arrayminus=list(res.OR-res.lo),
                             color='#BBB', thickness=1.8, width=8),
                marker=dict(color=fc, size=12, symbol='square',
                            line=dict(color='white',width=1.5)),
                text=[f"OR={row.OR:.3f} [{row.lo:.3f}, {row.hi:.3f}] {s}"
                      for (_,row),s in zip(res.iterrows(),sig)],
                hoverinfo='text', showlegend=False))
            fig.update_layout(
                title=dict(text="Forest Plot · Odds Ratios (95% CI)",
                           font=dict(color=NAV,size=14,family='Arial')),
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis=dict(**AXIS, title="Odds Ratio"),
                yaxis=dict(**AXIS, title=""),
                height=340, margin=dict(t=50,b=40,l=160,r=30))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">Model Diagnostics</div>',
                        unsafe_allow_html=True)
            preds = model.predict(model_df)
            auc = roc_auc_score(model_df['impaired'], preds)
            pred_class = (preds >= 0.5).astype(int)
            cm = confusion_matrix(model_df['impaired'], pred_class)
            acc = (cm[0,0]+cm[1,1])/cm.sum()

            for lbl, val, color in [
                ("AUC-ROC",       f"{auc:.3f}",  NAV),
                ("Accuracy",      f"{acc:.1%}",  NAV),
                ("McFadden R²",   f"{model.prsquared:.3f}", "#27AE60"),
                ("N (model)",     f"{len(model_df):,}", GRY),
            ]:
                st.markdown(f"""<div class="metric-card" style="margin-bottom:10px">
                    <div class="metric-lbl">{lbl}</div>
                    <div class="metric-val" style="color:{color};font-size:1.8rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        # OR table
        st.markdown('<div class="section-header">Coefficient Table</div>',
                    unsafe_allow_html=True)
        tbl = res.copy()
        tbl.index = [lbls.get(i,i) for i in tbl.index]
        tbl['95% CI'] = tbl.apply(lambda r: f"[{r.lo:.3f}, {r.hi:.3f}]", axis=1)
        tbl['p-value'] = tbl['p'].apply(lambda p: f"{'<0.001' if p<0.001 else f'{p:.3f}'}")
        tbl['Significant'] = tbl['p'].apply(lambda p: '★' if p<0.05 else '')
        tbl = tbl[['OR','95% CI','p-value','Significant']].rename(columns={'OR':'Odds Ratio'})
        tbl['Odds Ratio'] = tbl['Odds Ratio'].round(3)
        st.dataframe(tbl, use_container_width=True)

        st.markdown(f"""<div class="insight-box">
            <b>Interpretation:</b> Age is the strongest predictor of cognitive impairment
            (OR={res.loc['age','OR']:.3f} per year, p{'<0.001' if res.loc['age','p']<0.001 else f"={res.loc['age','p']:.3f}"}).
            Higher income-to-poverty ratio is protective
            (OR={res.loc['income_pir','OR']:.3f}).
            Model AUC = {auc:.3f}, indicating {'good' if auc>0.7 else 'moderate'} discriminative ability.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPERIMENT DESIGN
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">A/B Test Simulator · Cognitive Health Intervention</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Simulate the statistical power of a randomized intervention study "
        "targeting cognitive impairment reduction in adults 60+."
    )

    col_a, col_b = st.columns([1,1])
    with col_a:
        baseline = st.slider("Baseline impairment rate (%)",
            10.0, 50.0, float(round(imp_rate,1)), 0.5,
            help="Current impairment rate in your cohort")
        mde = st.slider("Minimum detectable effect (pp)",
            1.0, 15.0, 5.0, 0.5,
            help="Smallest reduction you care about detecting")
        alpha = st.select_slider("Significance level (α)",
            options=[0.01, 0.05, 0.10], value=0.05)
        power = st.select_slider("Statistical power (1-β)",
            options=[0.70, 0.80, 0.90, 0.95], value=0.80)

    with col_b:
        # Sample size calculation
        p1 = baseline / 100
        p2 = p1 - (mde / 100)
        p_pool = (p1 + p2) / 2
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta  = stats.norm.ppf(power)
        n_per_arm = int(np.ceil(
            (z_alpha * np.sqrt(2*p_pool*(1-p_pool)) +
             z_beta  * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2 / (mde/100)**2
        ))
        n_total = n_per_arm * 2

        st.markdown(f"""<div class="metric-card">
            <div class="metric-lbl">Required Sample Size</div>
            <div class="metric-val" style="color:{NAV}">{n_total:,}</div>
            <div class="metric-sub">{n_per_arm:,} per arm · α={alpha} · power={power:.0%}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        avail = len(df)
        feasible = avail >= n_total
        color = "#27AE60" if feasible else RED
        icon = "✅" if feasible else "⚠️"
        st.markdown(f"""<div class="insight-box" style="border-color:{color}">
            {icon} <b>Feasibility check:</b><br>
            Filtered cohort: <b>{avail:,}</b> participants<br>
            Required: <b>{n_total:,}</b><br>
            {"<b style='color:#27AE60'>Study is feasible with current cohort.</b>" if feasible
             else f"<b style='color:{RED}'>Need {n_total-avail:,} more participants.</b>"}
        </div>""", unsafe_allow_html=True)

    # Power curve
    st.markdown('<div class="section-header">Power Curve</div>', unsafe_allow_html=True)
    n_range = np.arange(50, max(n_total*2, 1000), 20)
    powers = []
    for n in n_range:
        se = np.sqrt(p1*(1-p1)/n + p2*(1-p2)/n)
        z = abs(p1-p2)/se - z_alpha
        powers.append(stats.norm.cdf(z))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_range, y=[p*100 for p in powers],
        mode='lines', line=dict(color=NAV,width=2.5), name='Power (%)'))
    fig.add_hline(y=power*100, line_dash='dash', line_color=RED,
                  annotation_text=f"Target power {power:.0%}",
                  annotation_position="right")
    fig.add_vline(x=n_per_arm, line_dash='dash', line_color=GLD,
                  annotation_text=f"n={n_per_arm:,} per arm",
                  annotation_position="top right")
    fig.update_layout(
        title=dict(text=f"Statistical Power vs Sample Size per Arm  |  MDE={mde}pp",
                   font=dict(color=NAV,size=14,family='Arial')),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(**AXIS, title="Sample Size per Arm"),
        yaxis=dict(**AXIS, title="Power (%)", range=[0,105]),
        height=360, margin=dict(t=50,b=50,l=60,r=80),
        showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class="insight-box">
        <b>Study design summary:</b>
        To detect a <b>{mde} percentage-point</b> reduction in impairment
        (from {baseline:.1f}% to {baseline-mde:.1f}%) with {power:.0%} power
        at α={alpha}, you need <b>{n_per_arm:,} participants per arm</b>
        ({n_total:,} total). This is a two-sided z-test for proportions.
        The current filtered cohort (n={avail:,}) is
        {'sufficient' if feasible else 'insufficient'} for this design.
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#AAA;font-size:0.75rem'>"
    "Amrutha Ravikumar · MPS Analytics · Northeastern Roux Institute · 2026 · "
    "Data: CDC NHANES CFQ Module (public use) · "
    "<a href='https://github.com/Amy-way05/dementia-risk-cohort' "
    "style='color:#AAA'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
