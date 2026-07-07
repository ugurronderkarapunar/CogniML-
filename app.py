import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import openai
import matplotlib.pyplot as plt

# Sayfa ayarları
st.set_page_config(page_title="CogniML", layout="wide")
st.markdown("""
<style>
    .main { background-color: #0f1117; color: #c9d1e0; }
    .stButton>button { background-color: #4f8ef7; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Session state başlatma
for key, default in {
    "df_raw": None, "target": None, "task": None,
    "selected_models": [], "preproc": {}, "results": [],
    "best_model": None, "best_name": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ========== Yardımcı Fonksiyonlar ==========
def generate_profile(df, target):
    num = df.select_dtypes(include=np.number)
    cat = df.select_dtypes(include='object')
    n_rows, n_cols = df.shape
    return {
        "Satır": n_rows,
        "Sütun": n_cols,
        "Sayısal Sütun": len(num.columns),
        "Kategorik Sütun": len(cat.columns),
        "Hedef Benzersiz Değer": df[target].nunique(),
        "Eksik Hücre Oranı": round(df.isnull().sum().sum()/(n_rows*n_cols), 3),
        "Tekrar Eden Satır": df.duplicated().sum(),
        "Yüksek Kardinalite": [(c, df[c].nunique()) for c in cat.columns if df[c].nunique()>50]
    }

@st.cache_data
def preprocess_data(df, target, task):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    features = [c for c in df.columns if c != target]
    num_cols = df[features].select_dtypes(include=np.number).columns.tolist()
    cat_cols = df[features].select_dtypes(include="object").columns.tolist()

    # Outlier clipping
    if num_cols:
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        df[num_cols] = df[num_cols].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR, axis=1)

    # Skewness log1p
    for col in num_cols:
        if df[col].min() > 0 and abs(df[col].skew()) > 1:
            df[col] = np.log1p(df[col])

    # Eksik doldurma
    if num_cols:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="constant", fill_value="missing").fit_transform(df[cat_cols])

    # Frekans kodlama
    encoders = {}
    cat_options = {}
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)
        df[col].fillna(0, inplace=True)
        encoders[col] = freq.to_dict()
        cat_options[col] = list(freq.index)

    # Tüm feature'ları sayısala zorla
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    num_stats = {}
    for col in features:
        num_stats[col] = (float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    y = df[target].copy()
    target_encoder = None
    if task == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target)
        target_encoder = le
    else:
        y = y.astype(float)

    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Özellik önemi
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) if task=="classification" else RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    top_features = imp.head(15).index.tolist()

    preproc = {
        "selected_features": top_features,
        "encoders": encoders,
        "cat_options": cat_options,
        "num_stats": num_stats,
        "scaler": scaler,
        "target_encoder": target_encoder
    }
    # Sadece seçili özellikleri döndür
    selected_indices = [features.index(f) for f in top_features]
    return X_scaled[:, selected_indices], y, preproc

@st.cache_data
def train_models(X, y, task, model_codes):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []
    best_score = -np.inf
    best_model = None
    best_name = ""

    model_grids = {
        "classification": {
            "rf": (RandomForestClassifier(random_state=42, class_weight='balanced'),
                   {"n_estimators":[100,200],"max_depth":[None,10],"min_samples_split":[2,5]}),
            "xgb": (XGBClassifier(eval_metric='logloss',random_state=42),
                    {"n_estimators":[100,200],"max_depth":[3,6],"learning_rate":[0.01,0.1]}),
            "lgbm": (LGBMClassifier(random_state=42,verbose=-1),
                     {"n_estimators":[100,200],"max_depth":[-1,10],"learning_rate":[0.01,0.1]}),
            "gbm": (GradientBoostingClassifier(random_state=42),
                    {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]}),
            "svm": (SVC(probability=True,random_state=42),
                    {"C":[0.1,1,10],"gamma":["scale","auto"],"kernel":["rbf","linear"]}),
            "linear": (LogisticRegression(max_iter=1000,random_state=42),
                       {"C":[0.1,1,10],"solver":["lbfgs","liblinear"]})
        },
        "regression": {
            "rf": (RandomForestRegressor(random_state=42),
                   {"n_estimators":[100,200],"max_depth":[None,10],"min_samples_split":[2,5]}),
            "xgb": (XGBRegressor(random_state=42),
                    {"n_estimators":[100,200],"max_depth":[3,6],"learning_rate":[0.01,0.1]}),
            "lgbm": (LGBMRegressor(random_state=42,verbose=-1),
                     {"n_estimators":[100,200],"max_depth":[-1,10],"learning_rate":[0.01,0.1]}),
            "gbm": (GradientBoostingRegressor(random_state=42),
                    {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.01,0.1]}),
            "svm": (SVR(), {"C":[0.1,1,10],"gamma":["scale","auto"],"kernel":["rbf","linear"]}),
            "linear": (Ridge(random_state=42), {"alpha":[0.1,1,10]})
        }
    }
    grids = model_grids[task]
    scoring = 'roc_auc_ovr_weighted' if task=="classification" else 'r2'

    for code in model_codes:
        if code not in grids:
            continue
        model, param_dist = grids[code]
        search = RandomizedSearchCV(model, param_dist, n_iter=4, cv=3, scoring=scoring,
                                    random_state=42, n_jobs=-1, verbose=0)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        if task == "classification":
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, best.predict_proba(X_test)[:,1])
            else:
                auc = roc_auc_score(y_test, best.predict_proba(X_test), multi_class='ovr')
            results.append({"Model": code.upper(), "Test AUC": round(auc,4)})
            if auc > best_score:
                best_score, best_model, best_name = auc, best, code
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results.append({"Model": code.upper(), "Test R²": round(r2,4), "MAE": round(mae,4)})
            if r2 > best_score:
                best_score, best_model, best_name = r2, best, code

    if "dl" in model_codes:
        n_classes = len(np.unique(y_train)) if task=="classification" else 1
        input_dim = X_train.shape[1]
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"), layers.Dropout(0.3),
            layers.Dense(64, activation="relu"), layers.Dropout(0.2),
        ])
        if task=="classification":
            if n_classes==2:
                model.add(layers.Dense(1, activation="sigmoid"))
                loss="binary_crossentropy"; metrics=["accuracy"]
            else:
                model.add(layers.Dense(n_classes, activation="softmax"))
                loss="sparse_categorical_crossentropy"; metrics=["accuracy"]
        else:
            model.add(layers.Dense(1)); loss="mse"; metrics=["mae"]
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss=loss, metrics=metrics)
        early = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, callbacks=[early], verbose=0)
        y_pred_prob = model.predict(X_test, verbose=0)
        if task=="classification":
            if n_classes==2:
                auc = roc_auc_score(y_test, y_pred_prob)
            else:
                auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
            results.append({"Model": "DL", "Test AUC": round(auc,4)})
            if auc > best_score:
                best_score, best_model, best_name = auc, model, "dl"
        else:
            y_pred = y_pred_prob.flatten()
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results.append({"Model": "DL", "Test R²": round(r2,4), "MAE": round(mae,4)})
            if r2 > best_score:
                best_score, best_model, best_name = r2, model, "dl"

    return results, best_name, best_model

def generate_report(results, best_name, task, api_key=None):
    lines = []
    lines.append("## CogniML Otomatik Rapor\n")
    lines.append(f"**Görev:** {task.upper()}")
    lines.append(f"**En Başarılı Model:** {best_name.upper()}\n")
    lines.append("| Model | Metrik |")
    lines.append("|-------|--------|")
    for r in results:
        vals = " | ".join(f"{v}" for v in r.values())
        lines.append(f"| {vals} |")
    if api_key:
        openai.api_key = api_key
        prompt = f"Aşağıdaki ML sonuçlarını yönetici özeti olarak yorumla:\n{results}"
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}],
                temperature=0.3
            )
            lines.append("\n### 🤖 Yapay Zeka Yorumu\n")
            lines.append(resp.choices[0].message.content)
        except Exception as e:
            lines.append(f"\n*(LLM hatası: {e})*")
    else:
        lines.append("\n*(LLM yorumu için API anahtarı girin.)*")
    return "\n".join(lines)

# ========== ARAYÜZ ==========
st.title("🧠 CogniML – The Cognitive Data Scientist")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Veri Yükle", "🔍 Profil", "🤖 Eğitim", "📊 Sonuçlar", "🔮 Tahmin"
])

with tab1:
    st.header("Veri Yükleme")
    f = st.file_uploader("CSV veya Excel", type=["csv","xlsx","xls"])
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
            st.success(f"{f.name} yüklendi – {df.shape[0]} satır × {df.shape[1]} sütun")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Hata: {e}")

with tab2:
    st.header("Veri Profili")
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
            st.json(profile)
            st.success(f"Görev: **{task.upper()}**")

with tab3:
    st.header("Model Eğitimi")
    if st.session_state.df_raw is None or st.session_state.target is None:
        st.warning("Önce veri yükleyip profil adımında hedef seçin.")
    else:
        models = {
            "Random Forest": "rf",
            "XGBoost": "xgb",
            "LightGBM": "lgbm",
            "Gradient Boosting": "gbm",
            "SVM": "svm",
            "Logistic Regression / Ridge": "linear",
            "Derin Öğrenme": "dl"
        }
        selected_names = st.multiselect("Modelleri seçin", list(models.keys()), default=["Random Forest","XGBoost"])
        if st.button("🚀 Modelleri Eğit"):
            model_codes = [models[n] for n in selected_names]
            st.session_state.selected_models = model_codes
            with st.spinner("Veri işleniyor..."):
                X, y, preproc = preprocess_data(st.session_state.df_raw, st.session_state.target, st.session_state.task)
                st.session_state.preproc = preproc
            with st.spinner("Modeller eğitiliyor..."):
                results, best_name, best_model = train_models(X, y, st.session_state.task, model_codes)
                st.session_state.results = results
                st.session_state.best_model = best_model
                st.session_state.best_name = best_name
            st.success("Eğitim tamamlandı!")

with tab4:
    st.header("Sonuçlar & Rapor")
    if not st.session_state.results:
        st.warning("Henüz eğitim yapılmadı.")
    else:
        st.table(st.session_state.results)
        st.markdown(f"### 🏆 En iyi: **{st.session_state.best_name.upper()}**")
        api_key = st.text_input("OpenAI API Key (opsiyonel)", type="password")
        if st.button("Rapor Oluştur"):
            report = generate_report(st.session_state.results, st.session_state.best_name, st.session_state.task, api_key)
            st.markdown(report)

with tab5:
    st.header("Yeni Kayıt Tahmini")
    if st.session_state.best_model is None:
        st.warning("Önce model eğitin.")
    else:
        preproc = st.session_state.preproc
        features = preproc["selected_features"]
        st.write(f"**Hedef:** {st.session_state.target} | **Model:** {st.session_state.best_name.upper()}")
        input_data = {}
        cols = st.columns(2)
        for i, col in enumerate(features):
            if col in preproc["cat_options"]:
                opts = preproc["cat_options"][col]
                input_data[col] = cols[i%2].selectbox(col, opts, key=f"pred_{col}")
            else:
                mn, mx, mean = preproc["num_stats"].get(col, (0,100,50))
                input_data[col] = cols[i%2].number_input(col, value=float(mean), min_value=mn, max_value=mx, key=f"pred_{col}")

        if st.button("Tahmin Et"):
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
            model = st.session_state.best_model
            task = st.session_state.task
            if st.session_state.best_name != "dl":
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
