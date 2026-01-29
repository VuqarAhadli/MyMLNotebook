import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    PolynomialFeatures
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.metrics import mean_squared_error
from scipy.stats import loguniform

df=pd.read_csv('Model3/Student_Performance.csv')

X = df.drop('Performance Index', axis=1)
y = df['Performance Index'].copy()


plt.scatter(
    X['Previous Scores'],
    y,
    color='red',
    marker='*'
)
plt.xlabel('Previous Scores')
plt.ylabel('Performance Index')
plt.title('Previous score and index correlation')
plt.show()



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



num_features = X_train.select_dtypes(include=np.number).columns
cat_features = X_train.select_dtypes(exclude=np.number).columns



num_basic = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

num_poly = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    ))
])



pre_basic = ColumnTransformer([
    ('num', num_basic, num_features),
    ('cat', cat_pipe, cat_features)
])

pre_poly = ColumnTransformer([
    ('num', num_poly, num_features),
    ('cat', cat_pipe, cat_features)
])



pipelines = {
    'Linear': Pipeline([
        ('preprocess', pre_basic),
        ('model', LinearRegression())
    ]),

    'Ridge': Pipeline([
        ('preprocess', pre_basic),
        ('model', Ridge(alpha=3.0))
    ]),

    'Lasso': Pipeline([
        ('preprocess', pre_basic),
        ('model', Lasso(alpha=0.05, max_iter=10000))
    ]),

    'ElasticNet': Pipeline([
        ('preprocess', pre_basic),
        ('model', ElasticNet(alpha=3.0, l1_ratio=0.5))
    ]),

    'Polynomial Linear': Pipeline([
        ('preprocess', pre_poly),
        ('model', LinearRegression())
    ]),

    'Polynomial Ridge': Pipeline([
        ('preprocess', pre_poly),
        ('model', Ridge(alpha=3.0))
    ]),

    'Polynomial Lasso': Pipeline([
        ('preprocess', pre_poly),
        ('model', Lasso(alpha=0.01, max_iter=20000))
    ]),

    'Polynomial ElasticNet': Pipeline([
        ('preprocess', pre_poly),
        ('model', ElasticNet(alpha=1.0, l1_ratio=0.5))
    ])
}


def plot_learning_curve(model, X, y, title):
    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='neg_mean_squared_error',
        shuffle=True,
        random_state=42
    )

    rmse_train = np.sqrt(-train_scores.mean(axis=1))
    rmse_val = np.sqrt(-val_scores.mean(axis=1))

    plt.figure(figsize=(9, 7))
    plt.plot(sizes, rmse_train)
    plt.plot(sizes, rmse_val)
    plt.xlabel('Training size')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.grid()
    plt.show()



for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2_train = pipe.score(X_train, y_train)
    r2_test = pipe.score(X_test, y_test)

    print(
        name,
        'RMSE:', round(rmse, 3),
        'R2 train:', round(r2_train, 3),
        'R2 test:', round(r2_test, 3)
    )

    plot_learning_curve(pipe, X_train, y_train, name)



coef_linear = pipelines['Linear'].named_steps['model'].coef_
coef_ridge = pipelines['Ridge'].named_steps['model'].coef_
coef_lasso = pipelines['Lasso'].named_steps['model'].coef_
coef_elasticnet = pipelines['ElasticNet'].named_steps['model'].coef_

feature_names = pipelines['Linear'] \
    .named_steps['preprocess'] \
    .get_feature_names_out()

coefDataFrame = pd.DataFrame({
    'Feature': feature_names,
    'Linear': coef_linear,
    'Ridge': coef_ridge,
    'Lasso': coef_lasso,
    'ElasticNet': coef_elasticnet
}, index=feature_names)

coefDataFrame



scores = cross_val_score(
    pipelines['Linear'],
    X,
    y,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1
)

rmse_scores = np.sqrt(-scores)
rmse_scores



grid_params = {
    'model__alpha': [0.1, 1, 10]
}

grid = GridSearchCV(
    pipelines['Ridge'],
    param_grid=grid_params,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1
)

grid.fit(X, y)

print(
    grid.best_params_,
    grid.best_score_
)


param_dist = {
    'model__alpha': loguniform(0.001, 1000)
}

random_search = RandomizedSearchCV(
    pipelines['Ridge'],
    param_distributions=param_dist,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_iter=200
)

random_search.fit(X, y)

print(
    random_search.best_params_,
    random_search.best_score_
)
