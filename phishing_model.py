import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, auc,
                             precision_score, recall_score,
                             f1_score, roc_auc_score)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------
# LOAD DATASETS
# ----------------------------------

print("Loading datasets...")

df1 = pd.read_csv("phishing_url_dataset_unique.csv")
df2 = pd.read_csv("5.urldata.csv")

# ----------------------------------
# FEATURE ENGINEERING
# ----------------------------------

def calculate_entropy(url):
    prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

def extract_features(url):
    return {
        "Have_IP": 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        "Have_At": 1 if "@" in url else 0,
        "URL_Length": len(url),
        "URL_Depth": len([i for i in url.split('/') if i]),
        "Redirection": 1 if '//' in url[8:] else 0,
        "https_Domain": 1 if url.startswith("https") else 0,
        "TinyURL": 1 if any(short in url for short in ["bit.ly", "tinyurl"]) else 0,
        "Prefix_Suffix": 1 if "-" in url else 0,
        "Digit_Count": sum(c.isdigit() for c in url),
        "Letter_Count": sum(c.isalpha() for c in url),
        "Special_Char_Count": len(re.findall(r'[^a-zA-Z0-9]', url)),
        "Entropy": calculate_entropy(url)
    }

print("Extracting features from Dataset 1...")
df1_features = pd.DataFrame([extract_features(str(url)) for url in df1["url"]])
df1_features["Label"] = df1["label"]

print("Preparing Dataset 2...")
df2_selected = df2[[
    "Have_IP", "Have_At", "URL_Length", "URL_Depth",
    "Redirection", "https_Domain", "TinyURL",
    "Prefix/Suffix", "Label"
]].copy()

df2_selected.rename(columns={"Prefix/Suffix": "Prefix_Suffix"}, inplace=True)

for col in ["Digit_Count", "Letter_Count", "Special_Char_Count", "Entropy"]:
    df2_selected[col] = np.nan

# ----------------------------------
# COMBINE DATA
# ----------------------------------

combined_df = pd.concat([df1_features, df2_selected], ignore_index=True)
combined_df.drop_duplicates(inplace=True)
combined_df.fillna(combined_df.mean(), inplace=True)

print("Final dataset shape:", combined_df.shape)
print("Class distribution:\n", combined_df["Label"].value_counts())

# ----------------------------------
# SPLIT DATA
# ----------------------------------

X = combined_df.drop("Label", axis=1)
y = combined_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------
# DEFINE MODELS
# ----------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

# ----------------------------------
# TRAIN & EVALUATE ML MODELS
# ----------------------------------

print("\n========== MACHINE LEARNING MODELS ==========")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "Accuracy": acc,
        "ROC-AUC": auc_score,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    print(classification_report(y_test, y_pred))

# ----------------------------------
# CROSS VALIDATION
# ----------------------------------

print("\n========== 10-FOLD CROSS VALIDATION ==========")

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")

# ----------------------------------
# ROC CURVE COMPARISON
# ----------------------------------

plt.figure(figsize=(8,6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# ----------------------------------
# CONFUSION MATRIX (BEST MODEL - XGBoost)
# ----------------------------------

best_model = models["XGBoost"]
y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.show()

# ----------------------------------
# SHAP EXPLAINABILITY (XGBoost)
# ----------------------------------

print("\nGenerating SHAP Explainability...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# ----------------------------------
# ADVANCED ANN MODEL
# ----------------------------------

print("\n========== DEEP LEARNING MODEL ==========")

ann = Sequential()
ann.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
ann.add(BatchNormalization())
ann.add(Dropout(0.3))

ann.add(Dense(64, activation='relu'))
ann.add(Dropout(0.3))

ann.add(Dense(1, activation='sigmoid'))

ann.compile(loss='binary_crossentropy',
            optimizer=Adam(0.001),
            metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

ann.fit(X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1)

y_pred_ann = (ann.predict(X_test) > 0.5).astype("int32")

ann_accuracy = accuracy_score(y_test, y_pred_ann)
ann_auc = roc_auc_score(y_test, ann.predict(X_test))
ann_precision = precision_score(y_test, y_pred_ann)
ann_recall = recall_score(y_test, y_pred_ann)
ann_f1 = f1_score(y_test, y_pred_ann)

print("\nANN Performance:")
print("Accuracy:", ann_accuracy)
print("ROC-AUC:", ann_auc)
print("Precision:", ann_precision)
print("Recall:", ann_recall)
print("F1 Score:", ann_f1)

# ----------------------------------
# SAVE BEST MODEL
# ----------------------------------

joblib.dump(best_model, "phishing_model.pkl")
print("\nBest model saved as phishing_model.pkl")

# ----------------------------------
# FINAL MODEL COMPARISON TABLE
# ----------------------------------

results_df = pd.DataFrame(results).T
print("\n========== MODEL COMPARISON ==========")
print(results_df)

# ----------------------------------
# TEST CUSTOM URL
# ----------------------------------

print("\n===== TEST CUSTOM URL =====")
test_url = input("Enter URL to test: ")

test_features = extract_features(test_url)
test_df = pd.DataFrame([test_features])

prediction = best_model.predict(test_df)

if prediction[0] == 1:
    print("PHISHING URL DETECTED")
else:
    print("LEGITIMATE URL")
