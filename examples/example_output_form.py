"""
Plugin Name: Automated Evaluation & SHAP PDF Report Generator
Description: Computes regression metrics and generates diagnostic & XAI plots into a unified PDF.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import shap


def generate_report(model, X_train, X_test, y_train, y_test, output_path):
    """
    محاسبه معیارهای ارزیابی، رسم نمودارهای عملکردی و تحلیل SHAP
    و ذخیره‌سازی یکپارچه تمامی خروجی‌ها در یک فایل PDF.
    """
    # ۱. پیش‌بینی مدل روی داده‌های آموزش و آزمون
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # ۲. محاسبه معیارهای ارزیابی
    metrics = {
        "Metric": ["R2 Score", "RMSE", "MAE"],
        "Train Set": [
            f"{r2_score(y_train, y_train_pred):.4f}",
            f"{root_mean_squared_error(y_train, y_train_pred):.4f}",
            f"{mean_absolute_error(y_train, y_train_pred):.4f}",
        ],
        "Test Set": [
            f"{r2_score(y_test, y_test_pred):.4f}",
            f"{root_mean_squared_error(y_test, y_test_pred):.4f}",
            f"{mean_absolute_error(y_test, y_test_pred):.4f}",
        ],
    }

    # باز کردن فایل PDF برای ذخیره‌سازی صفحات
    with PdfPages(output_path) as pdf:

        # ==========================================
        # صفحه اول: جدول معیارها + نمودار تطابق و باقیمانده
        # ==========================================
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.2], hspace=0.35, wspace=0.25)

        # الف) جدول معیارهای ارزیابی در بالای صفحه
        ax_table = fig.add_subplot(gs[0, :])
        ax_table.axis("off")
        ax_table.set_title(
            f"Model Evaluation Report ({model.__class__.__name__})",
            fontsize=14, fontweight="bold", pad=15
        )
        table_data = list(zip(metrics["Metric"], metrics["Train Set"], metrics["Test Set"]))
        table = ax_table.table(
            cellText=table_data,
            colLabels=["Evaluation Metric", "Train Set", "Test Set"],
            loc="center",
            cellLoc="center",
            colWidths=[0.3, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.6)

        # ب) نمودار مقادیر واقعی در برابر پیش‌بینی‌شده (Parity Plot)
        ax_parity = fig.add_subplot(gs[1, 0])
        ax_parity.scatter(y_test, y_test_pred, alpha=0.7, edgecolors="k", label="Test Data")
        min_val = min(np.min(y_test), np.min(y_test_pred))
        max_val = max(np.max(y_test), np.max(y_test_pred))
        ax_parity.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit (y = x)")
        ax_parity.set_xlabel("Actual Values")
        ax_parity.set_ylabel("Predicted Values")
        ax_parity.set_title("Actual vs. Predicted (Test Set)")
        ax_parity.legend()
        ax_parity.grid(True, linestyle=":", alpha=0.6)

        # ج) نمودار توزیع خطای باقیمانده (Residuals Plot)
        residuals = y_test - y_test_pred
        ax_res = fig.add_subplot(gs[1, 1])
        ax_res.scatter(y_test_pred, residuals, alpha=0.7, color="darkorange", edgecolors="k")
        ax_res.axhline(0, color="r", linestyle="--", lw=2)
        ax_res.set_xlabel("Predicted Values")
        ax_res.set_ylabel("Residuals (Actual - Predicted)")
        ax_res.set_title("Residual Analysis (Test Set)")
        ax_res.grid(True, linestyle=":", alpha=0.6)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ==========================================
        # محاسبه مقادیر SHAP
        # ==========================================
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        except Exception:
            explainer = shap.Explainer(model.predict, X_train)
            shap_values = explainer(X_test).values

        # ==========================================
        # صفحه دوم: نمودار اهمیت سراسری متغیرها (SHAP Bar Plot)
        # ==========================================
        fig_bar = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title("SHAP Global Feature Importance", fontsize=13, fontweight="bold", pad=12)
        pdf.savefig(fig_bar, bbox_inches="tight")
        plt.close(fig_bar)

        # ==========================================
        # صفحه سوم: نمودار توزیع اثرات و جهت اثرگذاری (SHAP Beeswarm Plot)
        # ==========================================
        fig_summary = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("SHAP Feature Impact Summary (Test Set)", fontsize=13, fontweight="bold", pad=12)
        pdf.savefig(fig_summary, bbox_inches="tight")
        plt.close(fig_summary)