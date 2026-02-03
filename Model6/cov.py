import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    auc
)
df=pd.read_csv('Model6/covtype.csv')
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_lr = LogisticRegression(max_iter=500)

models = {
    "Multinomial Logistic Regression": base_lr,
    "One-vs-One Logistic Regression": OneVsOneClassifier(base_lr),
    "One-vs-Rest Logistic Regression": OneVsRestClassifier(base_lr),
}

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(f"\n{name}")
    print("-"*len(name))
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_test))
    disp.plot(cmap="gnuplot2", values_format="d", ax=ax)
    ax.set_title(name)
    plt.show()
    
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        y_score = pipe.predict_proba(X_test)
        n_classes = y_test_bin.shape[1]

        plt.figure(figsize=(10,8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")
        
        plt.plot([0,1], [0,1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - {name}")
        plt.legend(loc="lower right")
        plt.show()
        
        weighted_auc = roc_auc_score(y_test_bin, y_score, multi_class="ovr", average="weighted")
        print(f"Weighted ROC–AUC: {weighted_auc:.4f}")

for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)
