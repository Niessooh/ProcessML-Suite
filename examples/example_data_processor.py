import pandas as pd


def process(file_path):
    # ۱. خواندن فایل اکسل و استخراج داده‌های خام
    df_raw = pd.read_excel(file_path)

    times = df_raw.iloc[4, 1:].values
    df_data = df_raw.iloc[5:26, :].copy()

    # ۲. تغییر ساختار داده‌ها (Melt & Pivot) و حذف ستون Time
    cols = ['Feature'] + list(times)
    df_data.columns = cols
    df_data['Feature'] = df_data['Feature'].astype(str).str.strip()

    df_melted = df_data.melt(id_vars=['Feature'], var_name='Time', value_name='Value')
    df_pivot = df_melted.pivot(index='Time', columns='Feature', values='Value').reset_index(drop=True)

    # ۳. تبدیل مقادیر به عددی (مقادیر غیرعددی خودکار حذف و به NaN تبدیل می‌شوند)
    for col in df_pivot.columns:
        df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')

    # ۴. تنظیم متغیر هدف و حذف سطرهای فاقد متغیر هدف (برچسب)
    target_col_raw = 'درصد سرب آزاد'
    if target_col_raw in df_pivot.columns:
        df_pivot.rename(columns={target_col_raw: 'target'}, inplace=True)

    if 'target' in df_pivot.columns:
        df_pivot = df_pivot.dropna(subset=['target']).copy()

    # ۵. انتخاب ویژگی‌ها (Feature Selection) بر اساس لیست تایید شده
    valid_cols = []
    keep_keywords = [
        'وزن سیلو', 'وزن بال میل', 'توان اصلی',
        'دمای1', 'دمای2', 'دمای3', 'حجم هوا',
        'دمای Bag house', 'فشار هوای بالای', 'فشار هوای پایین',
        'فشار مکنده', 'دمای خروجی', 'فرکانس فن',
        'دمای محیط', 'دمای هوای ورودی'
    ]

    for col in df_pivot.columns:
        if col == 'target':
            valid_cols.append(col)
            continue
        if any(keyword in str(col).replace(' ', '') for keyword in [k.replace(' ', '') for k in keep_keywords]):
            valid_cols.append(col)

    df_filtered = df_pivot[valid_cols].copy()

    # خروجی خام بدون دستکاری و بدون جایگذاری مقادیر NaN
    return df_filtered
