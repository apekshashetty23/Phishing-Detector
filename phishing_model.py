import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------------
# LOAD DATASETS
# ----------------------------------

print("Loading datasets...")

df1 = pd.read_csv(r"D:\phishing attack detection\phishing_url_dataset_unique.csv")
df2 = pd.read_csv(r"D:\phishing attack detection\5.urldata.csv")

# ----------------------------------
# FEATURE EXTRACTION FOR DATASET 1
# ----------------------------------

def extract_features(url):
    return {
        "Have_IP": 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        "Have_At": 1 if "@" in url else 0,
        "URL_Length": len(url),
        "URL_Depth": url.count('/'),
        "Redirection": 1 if '//' in url[8:] else 0,
        "https_Domain": 1 if "https" in url else 0,
        "TinyURL": 1 if any(short in url for short in ["bit.ly", "tinyurl"]) else 0,
        "Prefix/Suffix": 1 if "-" in url else 0
    }

print("Extracting features from Dataset 1...")

feature_list = []

for url in df1["url"]:
    feature_list.append(extract_features(str(url)))

df1_features = pd.DataFrame(feature_list)
df1_features["Label"] = df1["label"]

# ----------------------------------
# SELECT REQUIRED COLUMNS FROM DATASET 2
# ----------------------------------

print("Preparing Dataset 2...")

df2_selected = df2[[
    "Have_IP", "Have_At", "URL_Length", "URL_Depth",
    "Redirection", "https_Domain", "TinyURL",
    "Prefix/Suffix", "Label"
]]

# ----------------------------------
# COMBINE BOTH DATASETS
# ----------------------------------

print("Combining datasets...")

combined_df = pd.concat([df1_features, df2_selected], ignore_index=True)
combined_df.drop_duplicates(inplace=True)

print("Total combined data:", combined_df.shape)
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
# TRAIN MODEL
# ----------------------------------

print("Training Random Forest model...")

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------
# PREDICT
# ----------------------------------

y_pred = model.predict(X_test)

# ----------------------------------
# EVALUATION
# ----------------------------------

print("\n===== MODEL RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------
# SAVE MODEL (IMPORTANT FOR UI)
# ----------------------------------

joblib.dump(model, "phishing_model.pkl")
print("\nModel saved successfully as phishing_model.pkl")

# ----------------------------------
# FEATURE IMPORTANCE GRAPH
# ----------------------------------

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ----------------------------------
# TEST CUSTOM URL
# ----------------------------------

print("\n===== TEST CUSTOM URL =====")

test_url = input("Enter a URL to test: ")

test_features = extract_features(test_url)
test_df = pd.DataFrame([test_features])

prediction = model.predict(test_df)

if prediction[0] == 1:
    print("⚠️ This URL is PHISHING!")
else:
    print("✅ This URL is LEGITIMATE!")
