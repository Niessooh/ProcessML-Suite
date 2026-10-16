"""
Plugin Name: Gradient Boosting Regressor with Hyperparameter Grid
Description: Boosting model plugin for comparative analysis in Chapter 4.
"""

from sklearn.ensemble import GradientBoostingRegressor

TARGET_COLUMN = "target"

def build_model():
    """
    ساخت مدل پایه Gradient Boosting.
    """
    return GradientBoostingRegressor(random_state=42)


def get_param_grid():
    """
    فضای جستجوی ابرپارامترها برای الگوریتم تقویت گرادیان.
    """
    return {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0]
    }