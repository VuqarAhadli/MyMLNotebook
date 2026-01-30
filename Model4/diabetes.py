import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("Model4/diabetes.csv")

plt.figure(figsize=(10, 8))
sns.heatmap(
    data.select_dtypes(include=np.number).corr(),
    annot=True
)
plt.title("Correlation Matrix of Diabetes Dataset Features")
plt.show()


fig, axes = plt.subplots(1, 4, figsize=(18, 5))

sns.boxplot(x="Outcome", y="Glucose", data=data, ax=axes[0], color="orange")
axes[0].set_title("Outcome vs. Glucose")

sns.boxplot(x="Outcome", y="BMI", data=data, ax=axes[1], color="magenta")
axes[1].set_title("Outcome vs. BMI")

sns.boxplot(x="Outcome", y="Age", data=data, ax=axes[2], color="green")
axes[2].set_title("Outcome vs. Age")

sns.boxplot(x="Outcome", y="Insulin", data=data, ax=axes[3], color="red")
axes[3].set_title("Outcome vs. Insulin")

plt.tight_layout()
plt.show()


sns.violinplot(x="Outcome", y="Age", data=data, color="blue")
plt.show()


df = pd.read_csv("Model4/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"].copy()

cols_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[cols_missing] = df[cols_missing].replace(0, np.nan)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

num_features = X_train.select_dtypes(include=np.number).columns
cat_features = X_train.select_dtypes(exclude=np.number).columns


num_pipeline = Pipeline(
    [
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler())
    ]
)

cat_pipeline = Pipeline(
    [
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ]
)

transformer = ColumnTransformer(
    [
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ],
    remainder="passthrough"
)

full_pipeline = Pipeline(
    [
        ("transform", transformer),
        (
            "model",
            LogisticRegression(
            solver="saga",
            max_iter=10000,
            l1_ratio=0.5,
            class_weight="balanced",
            verbose=1
            )
        )
    ]
)

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
preci_score = precision_score(y_test, y_pred)
recal_score = recall_score(y_test, y_pred)
f1_scoree = f1_score(y_test, y_pred)

print(
    f"\x1b[0;34maccuracy {acc_score}\n"
    f"\x1b[0;36mprecision score: {preci_score}\n"
    f"\x1b[0;35mrecall score: {recal_score}\n"
    f"\x1b[0;32mf1 score: {f1_scoree}\x1b[0;0m"
)
