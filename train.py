import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


print("Downloading Bitcoin data...")

data = yf.download("BTC-USD", start="2018-01-01", end="2025-01-01")

data["Daily_Return"] = data["Close"].pct_change()
data["MA_7"] = data["Close"].rolling(window=7).mean()
data["MA_14"] = data["Close"].rolling(window=14).mean()
data["MA_30"] = data["Close"].rolling(window=30).mean()
data["Volatility_7"] = data["Daily_Return"].rolling(window=7).std()

data["Target"] = np.where(
    data["Close"].shift(-1) > data["Close"],
    1,
    0
)

data = data.dropna()

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA_7",
    "MA_14",
    "MA_30",
    "Volatility_7"
]

X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=6
)

print("Training model...")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")
joblib.dump(features, "features.pkl")

metrics = {
    "accuracy": accuracy,
    "f1_score": f1
}

joblib.dump(metrics, "metrics.pkl")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()

y_probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("static/roc_curve.png")
plt.close()

print("\nAll files created successfully!")