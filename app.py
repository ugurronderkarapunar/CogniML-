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
import openai

st.set_page_config(page_title="CogniML", layout="wide")
st.markdown("""<style>
    .main { background-color: #0f1117; color: #c9d1e0; }
    .stButton>button { background-color: #4f8ef7; color: white; font-weight: bold; }
    .step-box { border: 1px solid #4f8ef7; border-radius: 8px; padding: 20px; margin: 10px 0; }
</style>""", unsafe_allow_html=True)

# ========== Session State ==========
if "step" not in st.session_state:
    st.session_state.step = 1
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "target" not in st.session_state:
    st.session_state.target = None
if "task" not in st.session_state:
    st.session_state.task = None
if "high_card_cols" not in st.session_state:
    st.session_state.high_card_cols = []
if "dropped_cols" not in st.session_state:
    st.session_state.dropped_cols = []
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
if "preproc" not in st.session_state:
    st.session_state.preproc = {}
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []
if "X_final" not in st.session_state:
    st.session_state.X_final = None
if "y_final" not in st.session_state:
    st.session_state.y_final = None
if "results" not in st.session_state:
    st.session_state.results = []
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "best_name" not in st.session_state:
    st.session_state.best_name = ""

# ========== Yardımcı Fonksiyonlar ==========
def is_numeric_column(series):
    sample = series.dropna().head(10).astype(str).str.strip()
    return sample.str.contains(r'^-?[\d.,%\s€$]+$').all()

def clean_numeric_column(series):
    return series.astype(str).str.replace('%','').str.replace('$','').str.replace('€','').str.replace(',','').str.strip()

def safe_convert_to_numeric(series):
    return pd.to_numeric(series, errors='coerce').fillna(0)

def extract_date_features(df, col):
    try:
        dt = pd.to_datetime(df[col], errors='coerce')
        if dt.notna().sum() > len(df) * 0.7:
            df[col+'_year'] = dt.dt.year
            df[col+'_month'] = dt.dt.month
            df[col+'_day'] = dt.dt.day
            df[col+'_dayofweek'] = dt.dt.dayofweek
            df.drop(columns=[col], inplace=True)
            return True
    except:
        pass
    return False

def compute_feature_importance(X, y, task):
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) if task=="classification" else RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

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
        if code not in grids: continue
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

# ========== ADIMLAR ==========
st.title("🧠 CogniML – Karar Yetkili Sürüm")
st.progress(st.session_state.step / 7)

# ---------- ADIM 1: Veri Yükle ----------
if st.session_state.step == 1:
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.header("📂 Adım 1: Veri Yükleme")
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
                if st.button("Profil ve Hedef Seçimine Geç ➡️"):
                    st.session_state.step = 2
                    st.rerun()
            except Exception as e:
                st.error(f"Hata: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- ADIM 2: Profil & Hedef ----------
elif st.session_state.step == 2:
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.header("🔍 Adım 2: Veri Profili ve Hedef Seçimi")
        df = st.session_state.df_raw
        target = st.selectbox("Hedef değişken", df.columns)
        if st.button("Profili Çıkar ve Devam Et"):
            st.session_state.target = target
            nuniq = df[target].nunique()
            task = "classification" if df[target].dtype==object or nuniq<=15 else "regression"
            st.session_state.task = task
            # Yüksek kardinalite tespiti
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            high_card = [(c, df[c].nunique()) for c in cat_cols if c!=target and df[c].nunique()>50]
            st.session_state.high_card_cols = high_card
            st.session_state.step = 3
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- ADIM 3: Yüksek Kardinalite Temizliği ----------
elif st.session_state.step == 3:
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.header("🗑️ Adım 3: Yüksek Kardinaliteli Sütunları Temizle")
        high_card = st.session_state.high_card_cols
        if not high_card:
            st.success("Yüksek kardinaliteli sütun bulunamadı.")
            if st.button("Özellik Seçimine Geç ➡️"):
                st.session_state.step = 4
                st.rerun()
        else:
            st.warning("Aşağıdaki sütunlar çok fazla benzersiz değere sahip. Silmek istediklerinizi işaretleyin:")
            drop_list = []
            for col, nunique in high_card:
                if st.checkbox(f"{col} ({nunique} benzersiz)", value=False, key=f"drop_{col}"):
                    drop_list.append(col)
            if st.button("Seçilenleri Sil ve Devam Et ➡️"):
                st.session_state.dropped_cols = drop_list
                st.session_state.step = 4
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- ADIM 4: Özellik Seçimi ----------
elif st.session_state.step == 4:
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.header("🧩 Adım 4: Özellik Seçimi")
        # Veriyi temizle ve özellik önemini hesapla
        df = st.session_state.df_raw.copy()
        target = st.session_state.target
        drop_cols = st.session_state.dropped_cols
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True, errors='ignore')
        df.drop_duplicates(inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        features = [c for c in df.columns if c != target]
        # Otomatik sayısallaştırma (tarih, yüzde, para vb.)
        for col in features[:]:
            if col in df.columns and df[col].dtype == object:
                if is_numeric_column(df[col]):
                    df[col] = clean_numeric_column(df[col])
                    df[col] = safe_convert_to_numeric(df[col])
                elif df[col].dropna().nunique() > 50:
                    extract_date_features(df, col)  # tarih olabilir
        # Sayısal sütunları temizle
        num_cols = df[features].select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            Q1 = df[num_cols].quantile(0.25)
            Q3 = df[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            df[num_cols] = df[num_cols].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR, axis=1)
        # Çarpıklık
        for col in num_cols:
            if df[col].min() > 0 and abs(df[col].skew()) > 1:
                df[col] = np.log1p(df[col])
        # Eksik doldurma
        cat_cols = df[features].select_dtypes(include="object").columns.tolist()
        if num_cols:
            df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = SimpleImputer(strategy="constant", fill_value="missing").fit_transform(df[cat_cols])
        # Kategorikleri frekans kodla
        encoders = {}
        cat_options = {}
        for col in cat_cols:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq).fillna(0)
            encoders[col] = freq.to_dict()
            cat_options[col] = list(freq.index)
        # Tüm sütunları sayısala zorla
        for col in features:
            df[col] = safe_convert_to_numeric(df[col])
        # Hedef encode
        y = df[target].copy()
        target_encoder = None
        if st.session_state.task == "classification":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target)
            target_encoder = le
        else:
            y = safe_convert_to_numeric(y)
        X = df[[c for c in features if c in df.columns]].dropna(axis=1, how='all')
        # Ölçekleyip önem hesapla
        scaler_temp = StandardScaler()
        X_temp = scaler_temp.fit_transform(X)
        importance = compute_feature_importance(X_temp, y, st.session_state.task)
        st.session_state.feature_importance = importance
        st.session_state.cleaned_data = (X, y, encoders, cat_options, target_encoder)
        st.session_state.all_features = X.columns.tolist()

        st.write("**Özellik Önem Sıralaması:**")
        st.dataframe(importance.reset_index().rename(columns={"index":"Özellik", 0:"Önem"}))
        # Kullanıcıya seçim hakkı
        default_features = importance.head(15).index.tolist()
        selected = st.multiselect("Modelde kullanılacak özellikleri seçin (en az 1)", 
                                  options=X.columns.tolist(),
                                  default=default_features)
        if st.button("Seçilen Özelliklerle Devam Et ➡️"):
            if not selected:
                st.error("En az bir özellik seçmelisiniz!")
            else:
                st.session_state.selected_features = selected
                # Seçilen özelliklerle son X_scaled oluştur
                X_sel = X[selected]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_sel)
                st.session_state.X_final = X_scaled
                st.session_state.y_final = y
                st.session_state.preproc = {
                    "selected_features": selected,
                    "encoders": encoders,
                    "cat_options": cat_options,
                    "num_stats": {col: (float(X_sel[col].min()), float(X_sel[col].max()), float(X_sel[col].mean())) for col in selected},
                    "scaler": scaler,
                    "target_encoder": target_encoder,
                    "task": st.session_state.task
                }
                st.session_state.step = 5
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- ADIM 5: Model Eğitimi ----------
elif st.session_state.step == 5:
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.header("🤖 Adım 5: Model Seçimi ve Eğitim")
        models = {
            "Random Forest": "rf",
            "XGBoost": "xgb",
            "LightGBM": "lgbm",
            "Gradient Boosting": "gbm",
            "SVM": "svm",
            "Logistic Regression / Ridge": "linear"
        }
        selected_names = st.multiselect("Eğitilecek modeller", list(models.keys()), default=["Random Forest","XGBoost"])
        if st.button("🚀 Modelleri Eğit"):
            model_codes = [models[n] for n in selected_names]
            with st.spinner("Eğitim sürüyor..."):
                results, best_name, best_model = train_models(
                    st.session_state.X_final, st.session_state.y_final,
                    st.session_state.task, model_codes
                )
                st.session_state.results = results
                st.session_state.best_model = best_model
                st.session_state.best_name = best_name
            st.success("Eğitim tamamlandı!")
            st.session_state.step = 6
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- ADIM 6: Sonuçlar & Rapor ----------
elif st.session_state.step == 6:
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.header("📊 Adım 6: Sonuçlar ve Rapor")
        if not st.session_state.results:
            st.warning("Henüz sonuç yok.")
        else:
            st.table(st.session_state.results)
            st.markdown(f"### 🏆 En iyi model: **{st.session_state.best_name.upper()}**")
            api_key = st.text_input("OpenAI API Key (opsiyonel)", type="password")
            if st.button("📄 Rapor Oluştur"):
                report = generate_report(st.session_state.results, st.session_state.best_name, st.session_state.task, api_key)
                st.markdown(report)
            if st.button("Tahmin Aşamasına Geç ➡️"):
                st.session_state.step = 7
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- ADIM 7: Tahmin ----------
elif st.session_state.step == 7:
    with st.container():
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.header("🔮 Adım 7: Yeni Kayıt Tahmini")
        preproc = st.session_state.preproc
        features = preproc["selected_features"]
        task = st.session_state.task
        st.write(f"**Hedef:** {st.session_state.target} | **Model:** {st.session_state.best_name.upper()}")
        input_data = {}
        cols = st.columns(2)
        for i, col in enumerate(features):
            if col in preproc["cat_options"]:
                opts = preproc["cat_options"][col]
                input_data[col] = cols[i%2].selectbox(col, opts, key=f"pred_{col}")
            else:
                mn, mx, mean = preproc["num_stats"].get(col, (0,100,50))
                input_data[col] = cols[i%2].number_input(col, value=float(mean), min_value=float(mn), max_value=float(mx), key=f"pred_{col}")

        if st.button("Tahmin Et"):
            row = []
            for col in features:
                val = input_data[col]
                if col in preproc["encoders"]:
                    freq_dict = preproc["encoders"][col]
                    val = freq_dict.get(str(val), 0)
                row.append(float(val))
            X_new = pd.DataFrame([row], columns=features).apply(safe_convert_to_numeric)
            X_scaled = preproc["scaler"].transform(X_new)
            model = st.session_state.best_model

            if task == "classification":
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]

                # Etiket çözme (director gibi isimleri geri getir)
                if preproc.get("target_encoder") is not None:
                    pred_label = preproc["target_encoder"].inverse_transform([pred])[0]
                    labels = list(preproc["target_encoder"].classes_)
                elif hasattr(model, "classes_"):
                    labels = model.classes_
                    pred_label = labels[pred]
                else:
                    pred_label = pred
                    labels = [str(i) for i in range(len(proba))]

                st.success(f"🎯 Tahmin: **{pred_label}**")
                prob_df = pd.DataFrame({"Sınıf": labels, "Olasılık": proba})
                st.bar_chart(prob_df.set_index("Sınıf"))
            else:
                pred = model.predict(X_scaled)[0]
                st.success(f"🎯 Tahmin: **{pred:.4f}**")

        if st.button("🔄 Başa Dön"):
            st.session_state.step = 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
