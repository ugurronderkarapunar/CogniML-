import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="CogniML Analyst", layout="wide")
st.markdown("""
<style>
    .main { background-color: #0f1117; color: #c9d1e0; }
    .stButton>button { background-color: #4f8ef7; color: white; font-weight: bold; }
    .insight-box { background: #1a1d27; padding: 15px; border-left: 4px solid #4ff7a0; border-radius: 4px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ========== Yardımcı fonksiyonlar ==========
def clean_column_names(df):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^\w]", "_", regex=True)
                  .str.replace(r"_+", "_", regex=True).str.strip("_"))
    return df

def is_numeric_like(series):
    sample = series.dropna().head(10).astype(str).str.strip()
    return sample.str.contains(r'^-?[\d.,%\s€$]+$').all()

def clean_numeric_col(series):
    return series.astype(str).str.replace('%','').str.replace('$','').str.replace('€','').str.replace(',','').str.strip()

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

def generate_auto_insights(df):
    insights = []
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        insights.append(f"⚠️ Eksik değer içeren {len(missing_cols)} sütun var: {', '.join(missing_cols.index[:5])}")
    dup = df.duplicated().sum()
    if dup > 0:
        insights.append(f"🔄 {dup} tekrar eden satır bulunuyor.")
    num_cols = df.select_dtypes(include=np.number).columns
    outlier_cols = []
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        if ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).any():
            outlier_cols.append(col)
    if outlier_cols:
        insights.append(f"📈 Aykırı değerler barındıran sütunlar: {', '.join(outlier_cols[:5])}")
    skewed = []
    for col in num_cols:
        if abs(df[col].skew()) > 1:
            skewed.append(col)
    if skewed:
        insights.append(f"📐 Yüksek çarpıklık (>1) gösteren sütunlar: {', '.join(skewed[:5])}")
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.7:
                    high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        if high_corr:
            insights.append("🔗 Yüksek korelasyonlu değişken çiftleri:")
            for pair in high_corr[:5]:
                insights.append(f"   - {pair[0]} & {pair[1]}: {pair[2]:.2f}")
    return insights

# ========== Session State ==========
if "df" not in st.session_state:
    st.session_state.df = None

st.title("📊 CogniML Analyst – Otomatik İçgörülü Veri Analizi")

# ========== Veri Yükleme ==========
with st.sidebar:
    st.header("📂 Veri Yükle")
    uploaded_file = st.file_uploader("CSV veya Excel", type=["csv", "xlsx", "xls"])
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

if st.session_state.df is not None:
    df = st.session_state.df

    # ========== Üst bilgi kartları ==========
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Satır", df.shape[0])
    col2.metric("Sütun", df.shape[1])
    col3.metric("Sayısal Sütun", len(df.select_dtypes(include=np.number).columns))
    col4.metric("Eksik Hücre %", f"{df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%")

    # ========== Ana Sekmeler ==========
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧹 Temizlik", "📈 Görsel Keşif", "🧩 Pivot & Filtre", "📋 Tablo", "💾 İndir", "🧠 İçgörü"
    ])

    # ---------- TEMİZLİK ----------
    with tab1:
        st.subheader("Akıllı Veri Temizleme")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🪄 Sayısal/Tarih Kolonlarını Otomatik Düzelt"):
                for col in df.columns:
                    if df[col].dtype == object:
                        if is_numeric_like(df[col]):
                            df[col] = clean_numeric_col(df[col])
                            df[col] = safe_to_numeric(df[col])
                        elif df[col].nunique() > 50:
                            extract_dates(df, col)
                st.success("Sayısal ve tarih sütunları düzeltildi!")
        with c2:
            if st.button("📊 Eksik Değerleri Doldur (Medyan / Mod)"):
                num_cols = df.select_dtypes(include=np.number).columns
                cat_cols = df.select_dtypes(include='object').columns
                if len(num_cols) > 0:
                    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                if len(cat_cols) > 0:
                    for c in cat_cols:
                        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Bilinmiyor")
                st.success("Eksikler dolduruldu!")
        if df.duplicated().sum() > 0:
            if st.button(f"🗑️ {df.duplicated().sum()} Tekrar Eden Satırı Sil"):
                df.drop_duplicates(inplace=True)
                st.success("Tekrarlar silindi.")
        if st.button("💾 Temizlenmiş Hali Kaydet"):
            st.session_state.df = df
            st.success("Veri güncellendi!")

    # ---------- GÖRSEL KEŞİF ----------
    with tab2:
        st.subheader("📈 Etkileşimli Grafikler (Plotly)")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        if not numeric_cols:
            st.warning("Sayısal sütun bulunamadı. Lütfen önce 'Temizlik' sekmesinde veriyi düzeltin.")
        else:
            col_left, col_right = st.columns([1, 3])
            with col_left:
                chart_type = st.selectbox("Grafik Tipi", 
                                        ["Çubuk","Çizgi","Dağılım (Scatter)","Pasta","Histogram","Kutu","Alan","Isı Haritası"])
                x_col = st.selectbox("X Ekseni", all_cols)
                y_col = st.selectbox("Y Ekseni (sayısal)", numeric_cols)
                color_col = st.selectbox("Renklendir (opsiyonel)", ["Yok"] + all_cols)
                size_col = st.selectbox("Büyüklük (Scatter için)", ["Yok"] + numeric_cols)
                facet_col = st.selectbox("Gruplara Ayır (Facet)", ["Yok"] + all_cols)
                
                # Filtreler
                st.subheader("Filtreler")
                filter_conditions = []
                for col in all_cols:
                    if st.checkbox(f"Filtrele: {col}", key=f"filter_{col}"):
                        if df[col].dtype == object or df[col].nunique() < 30:
                            selected = st.multiselect(f"{col} değerleri", df[col].unique(), key=f"mf_{col}")
                            if selected:
                                filter_conditions.append((col, selected))
                        else:
                            min_v, max_v = float(df[col].min()), float(df[col].max())
                            range_vals = st.slider(f"{col} aralığı", min_v, max_v, (min_v, max_v), key=f"rg_{col}")
                            filter_conditions.append((col, range_vals))

            with col_right:
                plot_df = df.copy()
                for col, cond in filter_conditions:
                    if isinstance(cond, list):
                        plot_df = plot_df[plot_df[col].isin(cond)]
                    else:
                        plot_df = plot_df[(plot_df[col] >= cond[0]) & (plot_df[col] <= cond[1])]

                if plot_df.empty:
                    st.warning("Filtreler sonucu veri kalmadı.")
                else:
                    color = None if color_col == "Yok" else color_col
                    size = None if size_col == "Yok" else size_col
                    facet = None if facet_col == "Yok" else facet_col

                    try:
                        if chart_type == "Çubuk":
                            fig = px.bar(plot_df, x=x_col, y=y_col, color=color, facet_col=facet)
                        elif chart_type == "Çizgi":
                            fig = px.line(plot_df, x=x_col, y=y_col, color=color, facet_col=facet)
                        elif chart_type == "Dağılım (Scatter)":
                            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color, size=size, facet_col=facet)
                        elif chart_type == "Pasta":
                            fig = px.pie(plot_df, names=x_col, values=y_col, color=color)
                        elif chart_type == "Histogram":
                            fig = px.histogram(plot_df, x=x_col, color=color, facet_col=facet)
                        elif chart_type == "Kutu":
                            fig = px.box(plot_df, x=x_col, y=y_col, color=color, facet_col=facet)
                        elif chart_type == "Alan":
                            fig = px.area(plot_df, x=x_col, y=y_col, color=color, facet_col=facet)
                        elif chart_type == "Isı Haritası":
                            if color_col == "Yok":
                                st.error("Isı haritası için lütfen bir renk sütunu seçin.")
                                st.stop()
                            pivot = plot_df.pivot_table(index=x_col, columns=color_col, values=y_col, aggfunc='mean')
                            fig = px.imshow(pivot, title=f"{y_col} ısı haritası")
                        fig.update_layout(template="plotly_dark", height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Grafik oluşturulamadı: {e}")

    # ---------- PİVOT & FİLTRE ----------
    with tab3:
        st.subheader("🧩 Pivot Tablo & Gelişmiş Filtre")
        pivot_index = st.selectbox("Satır", all_cols, key="piv_idx")
        pivot_cols = st.selectbox("Sütun", ["Yok"] + all_cols, key="piv_col")
        pivot_vals = st.selectbox("Değer", numeric_cols, key="piv_val")
        pivot_agg = st.selectbox("Fonksiyon", ["mean","sum","count","min","max"], key="piv_agg")
        if st.button("Pivot Tablo Oluştur"):
            if pivot_cols == "Yok":
                piv = df.pivot_table(index=pivot_index, values=pivot_vals, aggfunc=pivot_agg)
            else:
                piv = df.pivot_table(index=pivot_index, columns=pivot_cols, values=pivot_vals, aggfunc=pivot_agg)
            st.dataframe(piv)

    # ---------- TABLO ----------
    with tab4:
        st.subheader("📋 Ham Veri Tablosu")
        st.dataframe(df, use_container_width=True)

    # ---------- İNDİR ----------
    with tab5:
        st.subheader("💾 Veriyi Dışa Aktar")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 CSV İndir", data=csv, file_name='temizlenmis_veri.csv')
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Veri')
        st.download_button("📥 Excel İndir", data=output.getvalue(), file_name='temizlenmis_veri.xlsx')

    # ---------- İÇGÖRÜ ----------
    with tab6:
        st.subheader("🧠 Otomatik İçgörü")
        insights = generate_auto_insights(df)
        if insights:
            for ins in insights:
                st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)
        else:
            st.success("Veride belirgin bir sorun bulunamadı.")

else:
    st.info("👈 Lütfen sol kenar çubuğundan bir dosya yükleyin.")
