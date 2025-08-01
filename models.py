'''# models.py

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Dictionary of models and corresponding hyperparameter grids
model_configs = {
    "linear": {
        "model": LinearRegression(),
        "param_grid": None  # No hyperparameters to tune
    },
    "ridge": {
        "model": Ridge(),
        "param_grid": {"alpha": [0.1, 1.0, 10.0]}
    },
    "lasso": {
        "model": Lasso(),
        "param_grid": {"alpha": [0.01, 0.1, 1.0]}
    },
    "elasticnet": {
        "model": ElasticNet(),
        "param_grid": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.5, 0.9]
        }
    },
    "dtree": {
        "model": DecisionTreeRegressor(),
        "param_grid": {"max_depth": [3, 5, 10]}
    },
    "xgb": {
        "model": XGBRegressor(random_state=42, verbosity=0),
        "param_grid": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    }
}
'''


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model_configs = {
    "linear": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('reg', LinearRegression())
        ]),
        "param_grid": None  # No hyperparameters to tune
    },
    "ridge": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('reg', Ridge())
        ]),
        "param_grid": {
            "reg__alpha": [0.1, 1.0, 10.0]
        }
    },
    "lasso": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('reg', Lasso())
        ]),
        "param_grid": {
            "reg__alpha": [0.01, 0.1, 1.0]
        }
    },
    "elasticnet": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('reg', ElasticNet())
        ]),
        "param_grid": {
            "reg__alpha": [0.01, 0.1, 1.0],
            "reg__l1_ratio": [0.1, 0.5, 0.9]
        }
    },
    "dtree": {
        "model": DecisionTreeRegressor(),
        "param_grid": {"max_depth": [3, 5, 10]}
    },
    "xgb": {
        "model": XGBRegressor(random_state=42, verbosity=0),
        "param_grid": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    }
}
