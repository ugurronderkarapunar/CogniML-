import streamlit as st
import pandas as pd
import numpy as np
from core.data_profiler import generate_profile
from core.preprocessor import preprocess_pipeline
from core.trainer import train_and_compare
from core.reporter import generate_report

st.set_page_config(page_title="CogniML", layout="wide")
st.markdown("""
<style>
    .main { background-color: #0f1117; color: #c9d1e0; }
    .stButton>button { background-color: #4f8ef7; color: white; font-weight: bold; border-radius: 6px; }
    .stButton>button:hover { background-color: #3a6fd8; }
</style>
""", unsafe_allow_html=True)

# Oturum durumu
for key, default in {
    "df_raw": None, "target": None, "task": None,
    "selected_models": [], "selected_features": [],
    "model": None, "best_model_name": "", "preprocessing_objects": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🧠 CogniML – The Cognitive Data Scientist")
st.markdown("Veri yükle → Profille → Modelleri eğit → Rapor al → Tahmin yap")

# Adım çubuğu
step = st.radio("Adım", ["📂 Veri Yükle", "🔍 Veri Profili", "🤖 Model Seç & Eğit", "📊 Sonuçlar & Rapor", "🔮 Tahmin"],
                horizontal=True, label_visibility="collapsed")

# ── Adım 1: Veri Yükleme ──
if step == "📂 Veri Yükle":
    st.header("📂 Veri Yükleme")
    f = st.file_uploader("CSV/Excel", type=["csv","xlsx","xls"])
    if f:
        try:
            if f.name.endswith('.csv'):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
            df.columns = (df.columns.str.strip().str.lower()
                          .str.replace(r"[^\w]","_",regex=True)
                          .str.replace(r"_+","_",regex=True).str.strip("_"))
            st.session_state.df_raw = df
            st.success(f"{f.name} yüklendi: {df.shape[0]:,} satır × {df.shape[1]} sütun")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Okuma hatası: {e}")

# ── Adım 2: Veri Profili (Meta-Öznitelikler) ──
elif step == "🔍 Veri Profili":
    st.header("🔍 Veri Profili & Meta-Öznitelikler")
    if st.session_state.df_raw is None:
        st.warning("Lütfen önce veri yükleyin.")
    else:
        df = st.session_state.df_raw
        target = st.selectbox("Hedef değişken", df.columns, key="target_select")
        if st.button("Profili Çıkar"):
            st.session_state.target = target
            nuniq = df[target].nunique()
            task = "classification" if df[target].dtype==object or nuniq<=15 else "regression"
            st.session_state.task = task
            profile = generate_profile(df, target)
            st.subheader("Meta-Öznitelikler")
            st.json(profile)
            st.info(f"Görev: **{task.upper()}** | Hedef benzersiz değer: {nuniq}")

# ── Adım 3: Model Seçimi ve Eğitim ──
elif step == "🤖 Model Seç & Eğit":
    st.header("🤖 Model Seçimi ve Eğitim")
    if st.session_state.df_raw is None:
        st.warning("Önce veri yükleyin.")
    else:
        df = st.session_state.df_raw
        target = st.session_state.target
        if target is None:
            st.warning("Lütfen 'Veri Profili' adımında hedef seçin.")
        else:
            models_available = {
                "Random Forest": "rf",
                "XGBoost": "xgb",
                "LightGBM": "lgbm",
                "Gradient Boosting": "gbm",
                "SVM": "svm",
                "Logistic Regression / Ridge": "linear",
                "Derin Öğrenme": "dl"
            }
            selected_names = st.multiselect("Modelleri seçin", list(models_available.keys()),
                                            default=["Random Forest","XGBoost"])
            if st.button("🚀 Modelleri Eğit"):
                model_codes = [models_available[n] for n in selected_names]
                st.session_state.selected_models = model_codes
                with st.spinner("Ön işleme ve eğitim sürüyor..."):
                    # Preprocessing
                    X, y, preproc = preprocess_pipeline(df, target, st.session_state.task)
                    st.session_state.preprocessing_objects = preproc
                    # Eğitim ve karşılaştırma
                    results, best_name, best_model = train_and_compare(X, y, st.session_state.task, model_codes)
                    st.session_state.model = best_model
                    st.session_state.best_model_name = best_name
                    st.session_state.results = results
                st.success("Eğitim tamamlandı! Sonuçları görmek için 'Sonuçlar & Rapor' adımına geçin.")

# ── Adım 4: Sonuçlar ve Rapor ──
elif step == "📊 Sonuçlar & Rapor":
    st.header("📊 Model Karşılaştırma ve Rapor")
    if "results" not in st.session_state:
        st.warning("Önce model eğitimi yapın.")
    else:
        results = st.session_state.results
        st.subheader("Model Performansları")
        st.table(results)

        best_name = st.session_state.best_model_name
        st.markdown(f"### 🏆 En iyi model: **{best_name.upper()}**")

        openai_key = st.text_input("OpenAI API Key (LLM raporu için)", type="password")
        if st.button("📄 Rapor Oluştur"):
            with st.spinner("Rapor hazırlanıyor..."):
                report = generate_report(results, best_name, st.session_state.task, openai_key)
                st.markdown(report, unsafe_allow_html=True)

# ── Adım 5: Tahmin ──
elif step == "🔮 Tahmin":
    st.header("🔮 Yeni Veriyle Tahmin")
    if st.session_state.model is None:
        st.warning("Önce model eğitimi yapın.")
    else:
        preproc = st.session_state.preprocessing_objects
        features = preproc["selected_features"]
        task = st.session_state.task
        st.write(f"Hedef: **{st.session_state.target}** | Model: **{st.session_state.best_model_name.upper()}**")
        input_data = {}
        for col in features:
            if col in preproc["cat_options"]:
                opts = preproc["cat_options"][col]
                input_data[col] = st.selectbox(col, opts)
            else:
                mn,mx,mean = preproc["num_stats"].get(col, (0,100,50))
                input_data[col] = st.number_input(col, value=float(mean), min_value=mn, max_value=mx)
        if st.button("Tahmin Et"):
            # Ön işleme (frekans kodlaması + scaling)
            row = {}
            for col in features:
                val = input_data[col]
                if col in preproc["encoders"]:
                    freq_dict = preproc["encoders"][col]
                    val = freq_dict.get(str(val), 0)
                row[col] = float(val)
            X_new = pd.DataFrame([row])[features]
            X_new = X_new.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_scaled = preproc["scaler"].transform(X_new)
            model = st.session_state.model
            if st.session_state.best_model_name != "dl":
                if task == "classification":
                    pred = model.predict(X_scaled)[0]
                    proba = model.predict_proba(X_scaled)[0]
                    if preproc["target_encoder"]:
                        pred_label = preproc["target_encoder"].inverse_transform([pred])[0]
                        labels = list(preproc["target_encoder"].classes_)
                    else:
                        pred_label = pred
                        labels = model.classes_
                    st.success(f"🎯 Tahmin: **{pred_label}**")
                    prob_df = pd.DataFrame({"Sınıf": labels, "Olasılık": proba})
                    st.bar_chart(prob_df.set_index("Sınıf"))
                else:
                    pred = model.predict(X_scaled)[0]
                    st.success(f"🎯 Tahmin: **{pred:.4f}**")
            else:
                pred_prob = model.predict(X_scaled, verbose=0)
                if task == "classification":
                    if pred_prob.shape[1] == 1:
                        pred = (pred_prob > 0.5).astype(int).flatten()[0]
                        proba = np.array([1-pred_prob[0][0], pred_prob[0][0]])
                        labels = [0,1] if preproc["target_encoder"] is None else list(preproc["target_encoder"].classes_)
                    else:
                        pred = np.argmax(pred_prob, axis=1)[0]
                        proba = pred_prob[0]
                        labels = list(range(len(proba))) if preproc["target_encoder"] is None else list(preproc["target_encoder"].classes_)
                    if preproc["target_encoder"]:
                        pred_label = preproc["target_encoder"].inverse_transform([pred])[0]
                    else:
                        pred_label = pred
                    st.success(f"🎯 Tahmin: **{pred_label}**")
                    prob_df = pd.DataFrame({"Sınıf": labels, "Olasılık": proba})
                    st.bar_chart(prob_df.set_index("Sınıf"))
                else:
                    pred = float(pred_prob.flatten()[0])
                    st.success(f"🎯 Tahmin: **{pred:.4f}**")
