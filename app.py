import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from io import BytesIO
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import base64
from datetime import datetime

# ------------------------------
# SAYFA YAPILANDIRMASI VE TEMA
# ------------------------------
st.set_page_config(page_title="CogniML Analyst Pro", layout="wide", initial_sidebar_state="expanded")

# Profesyonel koyu tema ve özel CSS
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #e6edf3; }
    .stButton>button { background-color: #238636; color: white; border-radius: 6px; font-weight: 500; }
    .stButton>button:hover { background-color: #2ea043; }
    .metric-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }
    .insight-box { background: #161b22; border-left: 4px solid #79c0ff; padding: 12px; border-radius: 4px; margin: 8px 0; }
    .section-header { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# Plotly global tema
pio.templates["cogniml"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        colorway=["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff", "#ff7b72", "#79c0ff", "#a5d6ff"],
    )
)
pio.templates.default = "cogniml"

# ------------------------------
# YARDIMCI FONKSİYONLAR
# ------------------------------
def clean_column_names(df):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^\w]", "_", regex=True)
                  .str.replace(r"_+", "_", regex=True).str.strip("_"))
    return df

def safe_to_numeric(series):
    return pd.to_numeric(series, errors='coerce').fillna(0)

def extract_dates(df, col):
    try:
        dt = pd.to_datetime(df[col], errors='coerce')
        if dt.notna().sum() > len(df) * 0.7:
            df[col + '_year'] = dt.dt.year
            df[col + '_month'] = dt.dt.month
            df[col + '_day'] = dt.dt.day
            df[col + '_dayofweek'] = dt.dt.dayofweek
            df.drop(columns=[col], inplace=True)
            return True
    except:
        pass
    return False

def data_quality_score(df):
    score = 100
    missing = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    score -= missing * 30
    dup = df.duplicated().sum() / df.shape[0]
    score -= dup * 20
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        skewness = df[num_cols].skew().abs().mean()
        score -= min(skewness * 5, 15)
        outliers = 0
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers += ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        outlier_ratio = outliers / (len(num_cols) * df.shape[0])
        score -= outlier_ratio * 20
    return max(0, min(100, round(score, 1)))

# ------------------------------
# SESSION STATE
# ------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ------------------------------
# ANA ARAYÜZ
# ------------------------------
st.title("🚀 CogniML Analyst Pro")
st.markdown("**Senior Seviye Veri Analizi ve Görselleştirme Platformu**")

with st.sidebar:
    st.header("📂 Veri Kaynağı")
    uploaded_file = st.file_uploader("CSV, Excel yükleyin", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df = clean_column_names(df)
            st.session_state.df = df
            st.success(f"✅ {uploaded_file.name} yüklendi")
        except Exception as e:
            st.error(f"Hata: {e}")

    st.header("🎨 Tema")
    theme_choice = st.radio("Tema seç", ["Koyu", "Aydınlık"], index=0)
    if theme_choice == "Aydınlık":
        st.session_state.theme = "light"
        pio.templates.default = "plotly_white"
    else:
        st.session_state.theme = "dark"
        pio.templates.default = "cogniml"

if st.session_state.df is None:
    st.info("👈 Lütfen bir dosya yükleyin.")
    st.stop()

df = st.session_state.df
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
all_cols = df.columns.tolist()

# ============ ÜST KARTLAR ============
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Satır", df.shape[0])
col2.metric("Sütun", df.shape[1])
col3.metric("Sayısal", len(num_cols))
col4.metric("Eksik %", f"{df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%")
quality = data_quality_score(df)
col5.metric("Kalite Skoru", f"{quality}/100")

# ============ ANA SEKMELER ============
tab_names = [
    "🧹 Temizlik", "📈 Görsel Keşif", "🧩 Pivot & Kodlama",
    "📋 Veri Sözlüğü", "🧠 İçgörü & Testler", "⚙️ Gelişmiş", "💾 Dışa Aktar"
]
tabs = st.tabs(tab_names)

# ----- TEMİZLİK -----
with tabs[0]:
    st.subheader("Akıllı Veri Temizleme")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🪄 Sayısal/Tarih Düzelt"):
            for col in all_cols:
                if df[col].dtype == object:
                    if df[col].str.contains(r'^-?[\d.,%\s€$]+$').all():
                        df[col] = df[col].str.replace('%','').str.replace('$','').str.replace('€','').str.replace(',','')
                        df[col] = safe_to_numeric(df[col])
                    elif df[col].nunique() > 50:
                        extract_dates(df, col)
            st.success("Düzeltildi!")
    with c2:
        if st.button("📊 Eksik Doldur"):
            for col in num_cols:
                df[col] = df[col].fillna(df[col].median())
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Bilinmiyor")
            st.success("Eksikler dolduruldu.")
    with c3:
        if st.button("🗑️ Tekrarları Sil"):
            before = len(df)
            df.drop_duplicates(inplace=True)
            st.success(f"{before - len(df)} tekrar silindi.")
    if st.button("💾 Değişiklikleri Kaydet"):
        st.session_state.df = df
        st.success("Kaydedildi!")

# ----- GÖRSEL KEŞİF -----
with tabs[1]:
    st.subheader("Profesyonel Grafikler")
    col_left, col_right = st.columns([1, 3])
    with col_left:
        chart_type = st.selectbox("Grafik tipi", ["Çubuk","Çizgi","Dağılım","Pasta","Histogram","Kutu","Isı Haritası","Zaman Serisi","3D Dağılım","Coğrafi Harita"])
        x_col = st.selectbox("X Ekseni", all_cols, key="xcol")
        y_col = st.selectbox("Y Ekseni (sayısal)", num_cols, key="ycol") if num_cols else None
        color_col = st.selectbox("Renk", ["Yok"] + all_cols, key="colorcol")
        facet_col = st.selectbox("Grupla (facet)", ["Yok"] + all_cols, key="facetcol")
    with col_right:
        plot_df = df.copy()
        if chart_type == "Isı Haritası" and color_col != "Yok":
            pivot = plot_df.pivot_table(index=x_col, columns=color_col, values=y_col, aggfunc='mean')
            fig = px.imshow(pivot, aspect="auto", color_continuous_scale='viridis')
        elif chart_type == "Zaman Serisi":
            if pd.api.types.is_datetime64_any_dtype(df[x_col]) or 'date' in x_col.lower():
                fig = px.line(plot_df.sort_values(x_col), x=x_col, y=y_col, color=None if color_col=="Yok" else color_col)
            else:
                st.warning("Zaman serisi için tarih sütunu seçin.")
                st.stop()
        elif chart_type == "3D Dağılım":
            z_col = st.selectbox("Z Ekseni", num_cols, key="zcol")
            fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col, color=None if color_col=="Yok" else color_col)
        elif chart_type == "Coğrafi Harita":
            if 'country' in [c.lower() for c in all_cols] or 'city' in [c.lower() for c in all_cols]:
                geo_col = st.selectbox("Konum sütunu", all_cols, key="geocol")
                fig = px.choropleth(plot_df, locations=geo_col, locationmode='country names',
                                    color=y_col, color_continuous_scale='viridis')
            else:
                st.warning("Coğrafi harita için ülke/şehir sütunu gerekli.")
                st.stop()
        else:
            fig = getattr(px, chart_type.lower().split()[0])(plot_df, x=x_col, y=y_col,
                                                             color=None if color_col=="Yok" else color_col,
                                                             facet_col=None if facet_col=="Yok" else facet_col)
        fig.update_layout(template=pio.templates.default, height=550)
        st.plotly_chart(fig, use_container_width=True)

# ----- PIVOT & KODLAMA -----
with tabs[2]:
    st.subheader("Pivot Tablo ve Kodlama")
    sub1, sub2 = st.tabs(["Pivot", "Kodlama"])
    with sub1:
        idx = st.selectbox("Satır", all_cols, key="pivot_idx")
        col_pivot = st.selectbox("Sütun", ["Yok"] + all_cols, key="pivot_col")
        val_pivot = st.selectbox("Değer", num_cols, key="pivot_val")
        agg = st.selectbox("Fonksiyon", ["mean","sum","count","min","max"])
        if st.button("Pivot Oluştur"):
            if col_pivot == "Yok":
                pivot = df.pivot_table(index=idx, values=val_pivot, aggfunc=agg)
            else:
                pivot = df.pivot_table(index=idx, columns=col_pivot, values=val_pivot, aggfunc=agg)
            st.dataframe(pivot)
    with sub2:
        enc_col = st.selectbox("Kodlanacak sütun", cat_cols, key="enc_col")
        method = st.radio("Yöntem", ["Label Encoding", "One-Hot Encoding"])
        if st.button("Kodlamayı Uygula"):
            if method == "Label Encoding":
                le = LabelEncoder()
                df[enc_col + "_encoded"] = le.fit_transform(df[enc_col].astype(str))
            else:
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                encoded = ohe.fit_transform(df[[enc_col]])
                df_ohe = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([enc_col]))
                df = pd.concat([df, df_ohe], axis=1)
            st.session_state.df = df
            st.success("Kodlama eklendi.")

# ----- VERİ SÖZLÜĞÜ -----
with tabs[3]:
    st.subheader("Veri Sözlüğü (Metadata)")
    meta = pd.DataFrame({
        "Sütun": all_cols,
        "Tip": df.dtypes,
        "Eksik": df.isnull().sum(),
        "Eksik %": (df.isnull().sum()/len(df)*100).round(1),
        "Unique": df.nunique(),
        "Min": [df[col].min() if col in num_cols else "-" for col in all_cols],
        "Max": [df[col].max() if col in num_cols else "-" for col in all_cols],
    })
    st.dataframe(meta, use_container_width=True)

# ----- İÇGÖRÜ & TESTLER -----
with tabs[4]:
    st.subheader("Otomatik İçgörü ve İstatistiksel Testler")
    insight, test = st.tabs(["İçgörü", "Testler"])
    with insight:
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if not missing_cols.empty:
            st.markdown(f"<div class='insight-box'>⚠️ Eksik değerli sütunlar: {', '.join(missing_cols.index[:5])}</div>", unsafe_allow_html=True)
        dup = df.duplicated().sum()
        if dup > 0:
            st.markdown(f"<div class='insight-box'>🔄 {dup} tekrar eden satır.</div>", unsafe_allow_html=True)
        if num_cols:
            skewed = [c for c in num_cols if abs(df[c].skew()) > 1]
            if skewed:
                st.markdown(f"<div class='insight-box'>📐 Çarpık sütunlar: {', '.join(skewed[:5])}</div>", unsafe_allow_html=True)
            corr = df[num_cols].corr()
            high_corr = [(i,j,corr.loc[i,j]) for i in corr.columns for j in corr.columns if i<j and abs(corr.loc[i,j])>0.7]
            if high_corr:
                st.markdown(f"<div class='insight-box'>🔗 Yüksek korelasyonlar: {'; '.join([f'{a}&{b}:{v:.2f}' for a,b,v in high_corr[:5]])}</div>", unsafe_allow_html=True)
    with test:
        test_type = st.selectbox("Test seç", ["T-Testi (bağımsız)", "ANOVA", "Ki-Kare"])
        if test_type == "T-Testi (bağımsız)":
            group_col = st.selectbox("Gruplandırma sütunu", cat_cols, key="ttest_group")
            value_col = st.selectbox("Sayısal sütun", num_cols, key="ttest_val")
            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                g1 = df[df[group_col]==groups[0]][value_col].dropna()
                g2 = df[df[group_col]==groups[1]][value_col].dropna()
                t_stat, p_val = stats.ttest_ind(g1, g2)
                st.write(f"t-istatistik: {t_stat:.4f}, p-değeri: {p_val:.4f}")
                st.write("Anlamlı fark var" if p_val<0.05 else "Anlamlı fark yok")
            else:
                st.warning("T-testi için tam 2 grup gerekli.")
        elif test_type == "ANOVA":
            group_col = st.selectbox("Gruplandırma", cat_cols, key="anova_group")
            value_col = st.selectbox("Değer", num_cols, key="anova_val")
            groups = [df[df[group_col]==g][value_col].dropna() for g in df[group_col].unique() if len(df[df[group_col]==g])>1]
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                st.write(f"F-istatistik: {f_stat:.4f}, p-değeri: {p_val:.4f}")
            else:
                st.warning("Yeterli grup yok.")

# ----- GELİŞMİŞ -----
with tabs[5]:
    st.subheader("Gelişmiş Araçlar")
    st.markdown("**Regex ile Sütun Bölme**")
    col_split = st.selectbox("Sütun", all_cols, key="split_col")
    pattern = st.text_input("Regex deseni", r"[\s,]+")
    if st.button("Böl"):
        df[col_split].str.split(pattern, expand=True).rename(columns=lambda x: f"{col_split}_{x}")
        st.success("Sütun bölündü (sonuç yeni sütunlarda)")
    st.markdown("**Dönüşüm Önerisi**")
    if num_cols:
        skew_col = st.selectbox("Çarpık sütun", num_cols, key="skew_col")
        if st.button("log1p dönüşümü uygula"):
            df[skew_col + "_log"] = np.log1p(df[skew_col])
            st.success("Log dönüşümü eklendi.")

# ----- DIŞA AKTAR -----
with tabs[6]:
    st.subheader("Dışa Aktarma ve Rapor")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 CSV İndir", csv, "temiz_veri.csv")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 Excel İndir", output.getvalue(), "temiz_veri.xlsx")

    # Basit HTML rapor
    if st.button("📄 Hızlı Rapor (HTML)"):
        html = f"<html><body><h1>Veri Raporu</h1>{df.head(5).to_html()}</body></html>"
        b64 = base64.b64encode(html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="rapor.html">Raporu indir</a>'
        st.markdown(href, unsafe_allow_html=True)
