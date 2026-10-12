# -----------------------------------------------------------------------
# Zameen.com Property Price Predictor
# One-file Streamlit app. Run with: streamlit run app.py
# -----------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, accuracy_score

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# -----------------------------------------------------------------------
# Helper: turn a PKR number into readable words like "2.5 Crore"
# -----------------------------------------------------------------------
def price_to_words(pkr):
    pkr = max(pkr, 0)
    if pkr >= 1_000_000_000:
        return f"{pkr / 1_000_000_000:.2f} Arab"
    elif pkr >= 10_000_000:
        return f"{pkr / 10_000_000:.2f} Crore"
    elif pkr >= 100_000:
        return f"{pkr / 100_000:.2f} Lakh"
    elif pkr >= 1_000:
        return f"{pkr / 1_000:.2f} Thousand"
    else:
        return f"{pkr:.0f} PKR"

def get_fit_verdict(train_score, test_score, is_reg):
    if is_reg:
        if train_score - test_score > 0.10: return "Overfit"
        if test_score < 0.60: return "Underfit"
        return "Good fit"
    else:
        if train_score - test_score > 0.05: return "Overfit"
        if test_score < 0.75: return "Underfit"
        return "Good fit"

# -----------------------------------------------------------------------
# Main cached function: clean data and train models
# -----------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    # --- Load Data ---
    data_path = os.path.join("Dataset", "zameenData (1).csv")
    df = pd.read_csv(data_path)
    
    # --- Cleaning (STRICTLY UNCHANGED) ---
    df = df.dropna(how="all")
    df = df.drop(columns=["Unnamed: 0", "agent", "name", "added_days"])
    df = df[df["pupose"].isin(["For Sale", "For Rent"])]
    df = df[~df["baths"].astype(str).str.contains("PKR")]
    
    df["bed_num"] = pd.to_numeric(df["bed"].replace("-", 0), errors="coerce")
    df["baths_num"] = pd.to_numeric(df["baths"].replace("-", 0), errors="coerce")
    
    df["price"] = df["price"].str.replace("PKR", "", regex=False)
    price_parts = df["price"].str.split(" ", n=1, expand=True)
    unit_map = {"Thousand": 1_000, "Lakh": 100_000, "Crore": 10_000_000, "Arab": 1_000_000_000}
    df["price_num"] = pd.to_numeric(price_parts[0], errors="coerce") * price_parts[1].map(unit_map)
    
    area_parts = df["area"].str.split(" ", n=1, expand=True)
    df = df[area_parts[1].isin(["Kanal", "Marla"])]
    
    area_parts = df["area"].str.split(" ", n=1, expand=True)
    area_num = pd.to_numeric(area_parts[0], errors="coerce")
    df["area_marla"] = np.where(area_parts[1] == "Kanal", area_num * 20, area_num)
    
    df["area_name"] = df["location"].str.split(",").str[0]
    top_areas = df["area_name"].value_counts().head(15).index
    df["area_name"] = np.where(df["area_name"].isin(top_areas), df["area_name"], "Other")
    
    df = df[df["pupose"] == "For Sale"]
    price_cut = df["price_num"].quantile(0.99)
    df = df[df["price_num"] < price_cut]
    df = df.dropna(subset=["price_num", "area_marla", "bed_num", "baths_num"])

    clean_df = df.copy()
    median_price = df["price_num"].median()
    land_types = sorted(df["land_type"].dropna().unique().tolist())
    area_names = sorted(df["area_name"].unique().tolist())

    df = pd.get_dummies(df, columns=["land_type", "area_name"], dtype=int)
    dummy_cols = [c for c in df.columns if c.startswith("land_type_") or c.startswith("area_name_")]
    feature_cols = dummy_cols + ["area_marla", "bed_num", "baths_num"]
    
    X = df[feature_cols]
    y = df["price_num"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- LOG PRICE FIX ---
    y_log_train = np.log1p(y_train)
    y_log_test = np.log1p(y_test)

    # ================= REGRESSION MODELS (Tuned) =================

    # 1. Linear Regression (Log price applied)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_log_train)
    lin_pred_train = np.expm1(lin_reg.predict(X_train_scaled))
    lin_pred = np.expm1(lin_reg.predict(X_test_scaled))
    lin_params = "No parameters"

    # 2. Polynomial Regression (degree 2) with Ridge
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    poly_reg = Ridge(alpha=100)
    poly_reg.fit(X_train_poly, y_log_train)
    poly_pred_train = np.expm1(poly_reg.predict(X_train_poly))
    poly_pred = np.expm1(poly_reg.predict(X_test_poly))
    poly_params = "{'alpha': 100}"

    # 3. KNN Regressor
    knn_reg = KNeighborsRegressor(n_neighbors=20, p=1, weights='distance')
    knn_reg.fit(X_train_scaled, y_log_train)
    knn_pred_train = np.expm1(knn_reg.predict(X_train_scaled))
    knn_pred = np.expm1(knn_reg.predict(X_test_scaled))
    knn_params = "{'n_neighbors': 20, 'p': 1, 'weights': 'distance'}"

    # 4. SVR (5000 rows)
    s_rows = np.random.RandomState(42).choice(len(X_train_scaled), 5000, replace=False)
    svr = SVR(C=100, epsilon=0.2, gamma=0.1, kernel='rbf')
    svr.fit(X_train_scaled[s_rows], y_log_train.to_numpy()[s_rows])
    svr_pred_train = np.expm1(svr.predict(X_train_scaled))
    svr_pred = np.expm1(svr.predict(X_test_scaled))
    svr_params = "{'C': 100, 'epsilon': 0.2, 'gamma': 0.1, 'kernel': 'rbf'}"

    # 5. Decision Tree Regressor
    tree_reg = DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, min_samples_split=10, random_state=42)
    tree_reg.fit(X_train_scaled, y_log_train)
    tree_pred_train = np.expm1(tree_reg.predict(X_train_scaled))
    tree_pred = np.expm1(tree_reg.predict(X_test_scaled))
    tree_params = "{'max_depth': 12, 'min_samples_leaf': 5, 'min_samples_split': 10, 'random_state': 42}"

    reg_scores_list = [
        ("Linear Regression", lin_pred_train, lin_pred, lin_params),
        ("Polynomial Regression", poly_pred_train, poly_pred, poly_params),
        ("KNN Regressor", knn_pred_train, knn_pred, knn_params),
        ("SVR (5000 rows)", svr_pred_train, svr_pred, svr_params),
        ("Decision Tree Regressor", tree_pred_train, tree_pred, tree_params),
    ]

    reg_scores = pd.DataFrame([
        {
            "Model": name,
            "Train R2": r2_score(y_train, p_tr),
            "Test R2": r2_score(y_test, p_te),
            "Fit": get_fit_verdict(r2_score(y_train, p_tr), r2_score(y_test, p_te), True),
            "Best Params": params
        } for name, p_tr, p_te, params in reg_scores_list
    ])


    # ================= CLASSIFICATION MODELS =================
    y_class = (df["price_num"] > median_price).astype(int)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    # 1. Logistic Regression
    log_reg = LogisticRegression(C=100, max_iter=2000, solver='lbfgs')
    log_reg.fit(X_train_c_scaled, y_train_c)
    log_pred_train = log_reg.predict(X_train_c_scaled)
    log_pred = log_reg.predict(X_test_c_scaled)
    log_params = "{'C': 100, 'max_iter': 2000, 'solver': 'lbfgs'}"

    # 2. KNN Classifier
    knn_clf = KNeighborsClassifier(n_neighbors=10, p=1, weights='distance')
    knn_clf.fit(X_train_c_scaled, y_train_c)
    knn_clf_pred_train = knn_clf.predict(X_train_c_scaled)
    knn_clf_pred = knn_clf.predict(X_test_c_scaled)
    knn_clf_params = "{'n_neighbors': 10, 'p': 1, 'weights': 'distance'}"

    # 3. SVC (5000 rows)
    s_rows_c = np.random.RandomState(42).choice(len(X_train_c_scaled), 5000, replace=False)
    svc = SVC(C=10, gamma=0.1, kernel='rbf')
    svc.fit(X_train_c_scaled[s_rows_c], y_train_c.to_numpy()[s_rows_c])
    svc_pred_train = svc.predict(X_train_c_scaled)
    svc_pred = svc.predict(X_test_c_scaled)
    svc_params = "{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}"

    # 4. Decision Tree Classifier
    tree_clf = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, min_samples_split=10, random_state=42)
    tree_clf.fit(X_train_c_scaled, y_train_c)
    tree_clf_pred_train = tree_clf.predict(X_train_c_scaled)
    tree_clf_pred = tree_clf.predict(X_test_c_scaled)
    tree_clf_params = "{'max_depth': 12, 'min_samples_leaf': 5, 'min_samples_split': 10, 'random_state': 42}"

    clf_scores_list = [
        ("Logistic Regression", log_pred_train, log_pred, log_params),
        ("KNN Classifier", knn_clf_pred_train, knn_clf_pred, knn_clf_params),
        ("SVC (5000 rows)", svc_pred_train, svc_pred, svc_params),
        ("Decision Tree Classifier", tree_clf_pred_train, tree_clf_pred, tree_clf_params),
    ]

    clf_scores = pd.DataFrame([
        {
            "Model": name,
            "Train Accuracy": accuracy_score(y_train_c, p_tr),
            "Test Accuracy": accuracy_score(y_test_c, p_te),
            "Fit": get_fit_verdict(accuracy_score(y_train_c, p_tr), accuracy_score(y_test_c, p_te), False),
            "Best Params": params
        } for name, p_tr, p_te, params in clf_scores_list
    ])


    bundle = {
        "scaler": scaler,
        "scaler_c": scaler_c,
        "poly": poly,
        "feature_cols": feature_cols,
        "lin_reg": lin_reg,
        "poly_reg": poly_reg,
        "knn_reg": knn_reg,
        "svr": svr,
        "tree_reg": tree_reg,
        "log_reg": log_reg,
        "knn_clf": knn_clf,
        "svc": svc,
        "tree_clf": tree_clf,
        "reg_scores": reg_scores,
        "clf_scores": clf_scores,
        "median_price": median_price,
        "land_types": land_types,
        "area_names": area_names,
        "clean_df": clean_df,
        "num_rows": len(clean_df),
        "num_features": len(feature_cols),
    }
    return bundle

# -----------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------
st.set_page_config(page_title="Zameen.com Property Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 Zameen.com Property Price Predictor")
st.write("Train 9 machine learning models on Zameen.com data to predict property prices and classify them as expensive or not.")

with st.spinner("Training models, please wait..."):
    bundle = load_and_train()

scaler = bundle["scaler"]
scaler_c = bundle["scaler_c"]
poly = bundle["poly"]
feature_cols = bundle["feature_cols"]
lin_reg = bundle["lin_reg"]
poly_reg = bundle["poly_reg"]
knn_reg = bundle["knn_reg"]
svr = bundle["svr"]
tree_reg = bundle["tree_reg"]
log_reg = bundle["log_reg"]
knn_clf = bundle["knn_clf"]
svc = bundle["svc"]
tree_clf = bundle["tree_clf"]
reg_scores = bundle["reg_scores"]
clf_scores = bundle["clf_scores"]
median_price = bundle["median_price"]
land_types = bundle["land_types"]
area_names = bundle["area_names"]
clean_df = bundle["clean_df"]
num_rows = bundle["num_rows"]
num_features = bundle["num_features"]

tab1, tab2, tab3 = st.tabs(["Predict", "Model Scores", "About the Data"])

# ===============================================================
# TAB 1: Predict
# ===============================================================
with tab1:
    st.subheader("Enter property details to get a price prediction")
    col1, col2 = st.columns(2)

    with col1:
        chosen_type = st.selectbox("Property type", options=land_types)
        chosen_area = st.selectbox("Area", options=area_names)
        chosen_area_marla = st.number_input("Area (Marla)", min_value=1.0, max_value=2000.0, value=10.0, step=1.0)

    with col2:
        chosen_beds = st.number_input("Bedrooms", min_value=0, max_value=30, value=3, step=1)
        chosen_baths = st.number_input("Bathrooms", min_value=0, max_value=30, value=2, step=1)

    if st.button("Predict price"):
        row = {col: 0 for col in feature_cols}
        land_col = "land_type_" + chosen_type
        area_col = "area_name_" + chosen_area
        if land_col in row:
            row[land_col] = 1
        if area_col in row:
            row[area_col] = 1

        row["area_marla"] = chosen_area_marla
        row["bed_num"] = chosen_beds
        row["baths_num"] = chosen_baths

        row_df = pd.DataFrame([row])
        row_scaled = scaler.transform(row_df)
        row_poly = poly.transform(row_scaled)
        row_scaled_c = scaler_c.transform(row_df)

        # Predict and expm1 back to real price
        p_lin = np.expm1(float(lin_reg.predict(row_scaled)[0]))
        p_poly = np.expm1(float(poly_reg.predict(row_poly)[0]))
        p_knn = np.expm1(float(knn_reg.predict(row_scaled)[0]))
        p_svr = np.expm1(float(svr.predict(row_scaled)[0]))
        p_tree = np.expm1(float(tree_reg.predict(row_scaled)[0]))

        p_lin = max(p_lin, 0)
        p_poly = max(p_poly, 0)
        p_knn = max(p_knn, 0)
        p_svr = max(p_svr, 0)
        p_tree = max(p_tree, 0)

        st.markdown("### Regression predictions")
        reg_preds = pd.DataFrame({
            "Model": ["Linear Regression", "Polynomial Regression", "KNN Regressor", "SVR (5000 rows)", "Decision Tree Regressor"],
            "Predicted Price": [price_to_words(p_lin), price_to_words(p_poly), price_to_words(p_knn), price_to_words(p_svr), price_to_words(p_tree)],
        })
        st.dataframe(reg_preds, use_container_width=True)

        avg_price = (p_lin + p_poly + p_knn + p_svr + p_tree) / 5
        st.metric(label="Average predicted price (all 5 models)", value=price_to_words(avg_price))

        c_log = int(log_reg.predict(row_scaled_c)[0])
        c_knn = int(knn_clf.predict(row_scaled_c)[0])
        c_svc = int(svc.predict(row_scaled_c)[0])
        c_tree = int(tree_clf.predict(row_scaled_c)[0])

        def label(v): return "Expensive" if v == 1 else "Not expensive"
        
        st.markdown("### Classification predictions")
        clf_preds = pd.DataFrame({
            "Model": ["Logistic Regression", "KNN Classifier", "SVC (5000 rows)", "Decision Tree Classifier"],
            "Verdict": [label(c_log), label(c_knn), label(c_svc), label(c_tree)],
        })
        st.dataframe(clf_preds, use_container_width=True)
        
        votes = c_log + c_knn + c_svc + c_tree
        final_label = "Expensive" if votes >= 2 else "Not expensive"
        st.markdown(f"**Final verdict (majority vote): {final_label}**")
        st.write(f"A property is considered expensive if its price is above the median price, which is **{price_to_words(median_price)}**.")

# ===============================================================
# TAB 2: Model Scores
# ===============================================================
with tab2:
    st.subheader("Regression model scores")
    for _, row in reg_scores.iterrows():
        with st.expander(f"{row['Model']} (Fit: {row['Fit']})"):
            st.write(f"**Train R2:** {row['Train R2']:.4f}")
            st.write(f"**Test R2:** {row['Test R2']:.4f}")
            st.write(f"**Best Parameters:** {row['Best Params']}")
            
    st.dataframe(reg_scores.drop(columns=["Best Params"]), use_container_width=True)
    st.bar_chart(reg_scores.set_index("Model")[["Train R2", "Test R2"]])

    st.divider()

    st.subheader("Classification model scores")
    for _, row in clf_scores.iterrows():
        with st.expander(f"{row['Model']} (Fit: {row['Fit']})"):
            st.write(f"**Train Accuracy:** {row['Train Accuracy']:.4f}")
            st.write(f"**Test Accuracy:** {row['Test Accuracy']:.4f}")
            st.write(f"**Best Parameters:** {row['Best Params']}")

    st.dataframe(clf_scores.drop(columns=["Best Params"]), use_container_width=True)
    st.bar_chart(clf_scores.set_index("Model")[["Train Accuracy", "Test Accuracy"]])

# ===============================================================
# TAB 3: About the Data
# ===============================================================
with tab3:
    st.subheader("Data summary")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows used for training", f"{num_rows:,}")
    col_b.metric("Number of features", str(num_features))
    col_c.metric("Median price", price_to_words(median_price))
    
    st.write("First 10 rows of the cleaned data:")
    display_cols = ["land_type", "area", "area_name", "bed_num", "baths_num", "price_num", "area_marla"]
    show_cols = [c for c in display_cols if c in clean_df.columns]
    st.dataframe(clean_df[show_cols].head(10), use_container_width=True)
