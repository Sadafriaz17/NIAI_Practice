import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay,
)

# ---------- Settings (change these) ----------
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 5
EXPERIMENT_NAME = "train_test_demo"
# ---------------------------------------------

mlflow.set_experiment(EXPERIMENT_NAME)

# 1. Load data
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# 2. Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

with mlflow.start_run(run_name="random_forest"):
    # 3. Log parameters
    mlflow.log_params({
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    })

    # 4. Train
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # 5. Evaluate on train and test
    for name, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict(Xs)
        mlflow.log_metric(f"{name}_accuracy", accuracy_score(ys, pred))
        mlflow.log_metric(f"{name}_precision", precision_score(ys, pred))
        mlflow.log_metric(f"{name}_recall", recall_score(ys, pred))
        mlflow.log_metric(f"{name}_f1", f1_score(ys, pred))
        print(f"{name} accuracy: {accuracy_score(ys, pred):.4f}")

    # 6. Log confusion matrix picture
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Test confusion matrix")
    mlflow.log_figure(fig, "test_confusion_matrix.png")
    plt.close(fig)

    # 7. Log the model
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

print("Done. Run: mlflow ui   then open http://127.0.0.1:5000")
