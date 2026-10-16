"""
Plugin Name: Random Forest Regressor with Hyperparameter Grid
Description: Standard model plugin for the Industrial Process Prediction & XAI Platform.
Case Study: Lead Oxidation Unit (Apparent Density & Process Parameters Prediction)
"""

from sklearn.ensemble import RandomForestRegressor

TARGET_COLUMN = "target"

def build_model():
    """
    ساخت و بازگرداندن نمونه اولیه (Base Estimator) مدل یادگیری ماشین.
    تنظیم random_state جهت تضمین تکرارپذیری نتایج آزمایش‌ها (Reproducibility) الزامی است.
    """
    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1  # استفاده از پردازش موازی در سطح درخت‌ها
    )
    return model


def get_param_grid():
    """
    تعریف فضای جستجوی ابرپارامترها (Hyperparameter Search Space) جهت استفاده در GridSearchCV.
    
    نکته:
    - در صورتی که بخواهید مدل بدون بهینه‌سازی و با مقادیر پیش‌فرض آموزش ببیند،
      کافی است یک دیکشنری خالی {} برگردانید یا این تابع را کامنت کنید.
    """
    param_grid = {
        "n_estimators": [50, 100, 200],         # تعداد درخت‌های تصمیم
        "max_depth": [None, 5, 10, 15],         # حداکثر عمق درخت برای جلوگیری از بیش‌برازش
        "min_samples_split": [2, 5, 10],        # حداقل نمونه برای شکستن گره
        "min_samples_leaf": [1, 2, 4]           # حداقل نمونه در برگ نهایی (مؤثر در حذف نویز حسگرها)
    }
    return param_grid