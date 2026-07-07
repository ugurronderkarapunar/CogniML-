"""
CogniML Analyst – Senior-Level Interactive Data Analysis Tool
Streamlit uygulaması. Veri yükleme, temizleme, görsel keşif,
pivot tablolar, otomatik içgörü ve dışa aktarma sağlar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# ------------------------------
# SAYFA YAPILANDIRMASI
# ------------------------------
st.set_page_config(
    page_title="CogniML Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Koyu tema ve özel stiller
st.markdown("""
<style>
    .main { background-color: #0f1117; color: #c9d1e0; }
    .stButton>button {
        background-color: #4f8ef7; color: white; font-weight: bold;
        border-radius: 6px; padding: 8px 16px; transition: 0.2s;
    }
    .stButton>button:hover { background-color: #3a6fd8; }
    .metric-box {
        background: #1a1d27; border-radius: 10px; padding: 18px;
        border: 1px solid #2e3347; text-align: center;
    }
    .insight-box {
        background: #1a1d27; padding: 15px; border-left: 4px solid #4ff7a0;
        border-radius: 4px; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# YARDIMCI FONKSİYONLAR
# ------------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun adlarını küçük harf, alt çizgi formatına dönüştür."""
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def is_numeric_like(series: pd.Series) -> bool:
    """Bir serinin sayısal görünümlü olup olmadığını kontrol eder (% , $ vb. içerebilir)."""
    sample = series.dropna().head(10).astype(str).str.strip()
    return sample.str.contains(r'^-?[\d.,%\s€$]+$').all()

def clean_numeric_col(series: pd.Series) -> pd.Series:
    """Yüzde, para birimi gibi karakterleri temizler."""
    return series.astype(str).str.replace('%', '').str.replace('$', '').str.replace('€', '').str.replace(',', '').str.strip()

def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Bir seriyi güvenli şekilde sayısala çevirir, hataları 0 yapar."""
    return pd.to_numeric(series, errors='coerce').fillna(0)

def extract_date_features(df: pd.DataFrame, col: str) -> bool:
    """Tarih sütununu algılar ve yıl, ay, gün gibi özellikler çıkarır."""
    try:
        dt = pd.to_datetime(df[col], errors='coerce')
        if dt.notna().sum() > len(df) * 0.7:  # %70'den fazlası tarih ise
            df[col + '_year'] = dt.dt.year
            df[col + '_month'] = dt.dt.month
            df[col + '_day'] = dt.dt.day
            df[col + '_dayofweek'] = dt.dt.dayofweek
            df.drop(columns=[col], inplace=True)
            return True
    except Exception:
        pass
    return False

def generate_insights(df: pd.DataFrame) -> list:
    """Veri kümesi hakkında otomatik içgörüler üretir."""
    insights = []
    # Eksik değerler
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        insights.append(f"⚠️ Eksik değer içeren {len(missing_cols)} sütun var: {', '.join(missing_cols.index[:5])}")
    # Tekrar eden satırlar
    dup = df.duplicated().sum()
    if dup > 0:
        insights.append(f"🔄 {dup} tekrar eden satır bulunuyor.")
    # Sayısal sütunlar
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        insights.append("📉 Hiç sayısal sütun yok, grafikler sınırlı olacak.")
        return insights
    # Aykırı değerler (IQR)
    outlier_cols = []
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        if ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).any():
            outlier_cols.append(col)
    if outlier_cols:
        insights.append(f"📈 Aykırı değerler barındıran sütunlar: {', '.join(outlier_cols[:5])}")
    # Çarpıklık
    skewed = [col for col in num_cols if abs(df[col].skew()) > 1]
    if skewed:
        insights.append(f"📐 Yüksek çarpıklık (>1): {', '.join(skewed[:5])}")
    # Korelasyonlar
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    high_corr.append((corr.columns[i], corr.columns[j], val))
        if high_corr:
            insights.append("🔗 Yüksek korelasyonlu çiftler:")
            for a, b, v in high_corr[:5]:
                insights.append(f"   - {a} & {b}: {v:.2f}")
    return insights

# ------------------------------
# SESSION STATE BAŞLATMA
# ------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# ------------------------------
# ARAYÜZ BAŞLIĞI
# ------------------------------
st.title("📊 CogniML Analyst – Senior Veri Analizi Aracı")
st.markdown("Yükleyin, temizleyin, keşfedin ve içgörü kazanın.")

# ------------------------------
# SIDEBAR – VERİ YÜKLEME
# ------------------------------
with st.sidebar:
    st.header("📂 Veri Kaynağı")
    uploaded_file = st.file_uploader(
        "CSV veya Excel dosyası seçin",
        type=["csv", "xlsx", "xls"],
        help="Maksimum dosya boyutu: 200MB"
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df = clean_column_names(df)
            st.session_state.df = df
            st.success(f"✅ {uploaded_file.name} yüklendi")
        except Exception as e:
            st.error(f"❌ Dosya okunamadı: {e}")

# ------------------------------
# ANA İÇERİK
# ------------------------------
if st.session_state.df is None:
    st.info("👈 Başlamak için sol kenar çubuğundan bir veri dosyası yükleyin.")
    st.stop()

df = st.session_state.df

# === ÜST BİLGİ KARTLARI ===
st.markdown("### 📋 Veri Kümesi Özeti")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Satır", df.shape[0])
col2.metric("Sütun", df.shape[1])
col3.metric("Sayısal Sütun", len(df.select_dtypes(include=np.number).columns))
col4.metric("Eksik Hücre %", f"{df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%")

# === ANA SEKMELER ===
tabs = st.tabs([
    "🧹 Temizlik", "📈 Görsel Keşif", "🧩 Pivot Tablo", "📋 Ham Veri", "💾 Dışa Aktar", "🧠 İçgörü"
])

# =========================================================
# SEKMELERİN İÇERİĞİ
# =========================================================

# --- TEMİZLİK ---
with tabs[0]:
    st.subheader("Veri Temizleme Araçları")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🪄 Sayısal/Tarih Sütunlarını Otomatik Düzelt", help="% , $ , € işaretlerini temizler, tarihleri parçalar."):
            for col in df.columns:
                if df[col].dtype == object:
                    if is_numeric_like(df[col]):
                        df[col] = clean_numeric_col(df[col])
                        df[col] = safe_to_numeric(df[col])
                    elif df[col].nunique() > 50:
                        extract_date_features(df, col)
            st.success("Sayısal ve tarih sütunları düzeltildi!")
    with c2:
        if st.button("📊 Eksik Değerleri Doldur", help="Sayısal: medyan, Kategorik: mod ile doldurur."):
            num_cols = df.select_dtypes(include=np.number).columns
            cat_cols = df.select_dtypes(include='object').columns
            if len(num_cols) > 0:
                df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            if len(cat_cols) > 0:
                for c in cat_cols:
                    df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Bilinmiyor")
            st.success("Eksik değerler dolduruldu!")

    dup = df.duplicated().sum()
    if dup > 0:
        if st.button(f"🗑️ {dup} Tekrar Eden Satırı Sil"):
            df.drop_duplicates(inplace=True)
            st.success("Tekrarlar silindi.")

    if st.button("💾 Temizlenmiş Veriyi Kaydet (Session)", help="Yaptığınız değişiklikler oturum boyunca geçerli olur."):
        st.session_state.df = df
        st.success("Veri güncellendi!")

# --- GÖRSEL KEŞİF ---
with tabs[1]:
    st.subheader("Etkileşimli Grafikler")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    if not numeric_cols:
        st.warning("Sayısal sütun bulunamadı. Lütfen 'Temizlik' sekmesinde veriyi düzeltin.")
    else:
        # Sol panel – grafik ayarları
        with st.container():
            col_left, col_right = st.columns([1, 3])
            with col_left:
                chart_type = st.selectbox(
                    "Grafik Tipi",
                    ["Çubuk", "Çizgi", "Dağılım (Scatter)", "Pasta", "Histogram", "Kutu", "Alan", "Isı Haritası"]
                )
                x_col = st.selectbox("X Ekseni", all_cols)
                y_col = st.selectbox("Y Ekseni (sayısal)", numeric_cols)
                color_col = st.selectbox("Renklendir", ["Yok"] + all_cols)
                facet_col = st.selectbox("Gruplara Ayır (Facet)", ["Yok"] + all_cols)

                # Filtreler (dinamik)
                st.markdown("**Filtreler**")
                filters = []
                for col in all_cols:
                    if st.checkbox(f"Filtrele: {col}", key=f"flt_{col}"):
                        if pd.api.types.is_numeric_dtype(df[col]):
                            min_val, max_val = float(df[col].min()), float(df[col].max())
                            selected_range = st.slider(
                                f"{col} aralığı", min_val, max_val, (min_val, max_val), key=f"rng_{col}"
                            )
                            filters.append((col, "range", selected_range))
                        else:
                            unique_vals = df[col].dropna().unique()
                            selected_vals = st.multiselect(
                                f"{col} değerleri", unique_vals, key=f"mul_{col}"
                            )
                            if selected_vals:
                                filters.append((col, "multiselect", selected_vals))

            # Sağ panel – grafiğin çizilmesi
            with col_right:
                plot_df = df.copy()
                # Filtreleri uygula
                for col, ftype, fval in filters:
                    if ftype == "range":
                        plot_df = plot_df[(plot_df[col] >= fval[0]) & (plot_df[col] <= fval[1])]
                    elif ftype == "multiselect":
                        plot_df = plot_df[plot_df[col].isin(fval)]

                if plot_df.empty:
                    st.warning("Filtrelere uyan veri yok.")
                else:
                    color = None if color_col == "Yok" else color_col
                    facet = None if facet_col == "Yok" else facet_col
                    try:
                        if chart_type == "Çubuk":
                            fig = px.bar(plot_df, x=x_col, y=y_col, color=color, facet_col=facet,
                                         title=f"{y_col} - {x_col}")
                        elif chart_type == "Çizgi":
                            fig = px.line(plot_df, x=x_col, y=y_col, color=color, facet_col=facet)
                        elif chart_type == "Dağılım (Scatter)":
                            size_col = st.selectbox("Büyüklük (opsiyonel)", ["Yok"] + numeric_cols, key="size_col")
                            size = None if size_col == "Yok" else size_col
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
                                st.error("Isı haritası için 'Renklendir' seçeneğini doldurun.")
                                st.stop()
                            pivot = plot_df.pivot_table(index=x_col, columns=color_col, values=y_col, aggfunc='mean')
                            fig = px.imshow(pivot, title=f"{y_col} ısı haritası", aspect="auto")
                        fig.update_layout(template="plotly_dark", height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Grafik oluşturulurken hata: {e}")

# --- PİVOT TABLO ---
with tabs[2]:
    st.subheader("Pivot Tablo Oluştur")
    if not numeric_cols:
        st.warning("Pivot için en az bir sayısal sütun gerekli.")
    else:
        idx = st.selectbox("Satır", all_cols, key="piv_idx")
        cols = st.selectbox("Sütun", ["Yok"] + all_cols, key="piv_cols")
        vals = st.selectbox("Değer", numeric_cols, key="piv_vals")
        agg = st.selectbox("Fonksiyon", ["mean", "sum", "count", "min", "max"], key="piv_agg")
        if st.button("Pivot Tablo Oluştur"):
            try:
                if cols == "Yok":
                    piv = df.pivot_table(index=idx, values=vals, aggfunc=agg)
                else:
                    piv = df.pivot_table(index=idx, columns=cols, values=vals, aggfunc=agg)
                st.dataframe(piv)
            except Exception as e:
                st.error(f"Pivot hatası: {e}")

# --- HAM VERİ ---
with tabs[3]:
    st.subheader("Veri Tablosu")
    st.dataframe(df, use_container_width=True)

# --- DIŞA AKTAR ---
with tabs[4]:
    st.subheader("Veriyi İndir")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 CSV İndir", data=csv, file_name="temizlenmis_veri.csv", mime="text/csv")

    # Excel çıktısı
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Veri')
    st.download_button("📥 Excel İndir", data=output.getvalue(),
                       file_name="temizlenmis_veri.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- İÇGÖRÜ ---
with tabs[5]:
    st.subheader("Otomatik İçgörü Raporu")
    insights = generate_insights(df)
    if insights:
        for ins in insights:
            st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)
    else:
        st.success("Veriniz temiz görünüyor, belirgin bir sorun tespit edilmedi.")
