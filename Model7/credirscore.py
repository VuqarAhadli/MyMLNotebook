import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("Model7/train.csv", low_memory=False)

drop_cols = ["ID", "Customer_ID", "Name", "SSN"]
df.drop(columns=drop_cols, inplace=True)

def convert_credit_history(age_str):
    if pd.isna(age_str):
        return np.nan
    try:
        parts = age_str.split()
        years = int(parts[0])
        months = int(parts[3])
        return years * 12 + months
    except:
        return np.nan

df["Credit_History_Age"] = df["Credit_History_Age"].apply(convert_credit_history)

num_cols = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance"
]

for col in num_cols:
    df[col] = df[col].astype(str).str.replace('_', '').str.replace('-', '0').str.extract(r'([0-9.]+)')[0]
    df[col] = pd.to_numeric(df[col], errors='coerce')

le = LabelEncoder()
y = le.fit_transform(df['Credit_Score'])
X = df.drop('Credit_Score', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

lgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(n_estimators=200, max_depth=16, random_state=42,verbose=-1))
])
lgb_pipeline.fit(X_train, y_train)
lgb_preds = lgb_pipeline.predict(X_test)
lgb_probs = lgb_pipeline.predict_proba(X_test)

print("LightGBM Accuracy:", accuracy_score(y_test, lgb_preds))
print("LightGBM ROC-AUC:", roc_auc_score(y_test, lgb_probs, multi_class='ovo', average='macro'))

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42))
])
xgb_pipeline.fit(X_train, y_train)
xgb_preds = xgb_pipeline.predict(X_test)
xgb_probs = xgb_pipeline.predict_proba(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("XGBoost ROC-AUC:", roc_auc_score(y_test, xgb_probs, multi_class='ovo', average='macro'))

print("LGB Train Accuracy:", accuracy_score(y_train, lgb_pipeline.predict(X_train)))
print("XGB Train Accuracy:", accuracy_score(y_train, xgb_pipeline.predict(X_train)))
print("LGB Test Accuracy:", accuracy_score(y_test, lgb_pipeline.predict(X_test)))
print("XGB Test Accuracy:", accuracy_score(y_test, xgb_pipeline.predict(X_test)))

print("test split lgb cm\n",confusion_matrix(y_test,lgb_preds))

print("test split xg cm\n",confusion_matrix(y_test,xgb_preds))

test_data = pd.read_csv("Model7/test.csv")

drop_cols = ["ID", "Customer_ID", "Name", "SSN"]
test_data.drop(columns=[c for c in drop_cols if c in test_data.columns], inplace=True)

test_data["Credit_History_Age"] = test_data["Credit_History_Age"].apply(convert_credit_history)

for col in num_cols:
    if col in test_data.columns:
        test_data[col] = test_data[col].astype(str).str.replace('_', '').str.replace('-', '0').str.extract(r'([0-9.]+)')[0]
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

preds_encoded = lgb_pipeline.predict(test_data)
preds_labels = le.inverse_transform(preds_encoded)

print("lgb test.csv",pd.Series(preds_labels).value_counts())

preds_encoded = xgb_pipeline.predict(test_data)
preds_labels = le.inverse_transform(preds_encoded)

print("xgb test.csv",pd.Series(preds_labels).value_counts())