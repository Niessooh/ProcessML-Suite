# ML Pipeline Studio

  

A small PySide6 desktop app for a repeatable ML workflow: load raw data,

turn it into a dataset, train a model on it, and run predictions with the

trained model. Everything the app can plug into (data readers, models,

report generators) is a plain `.py` file you drop in through the UI — no

code changes needed to try a new model.

  

## Tabs

  

| Tab | What it does |

|---|---|

| **DATA** | Pick one or more raw files, pick a data processor, and either append the result to an existing dataset or create a new one. |

| **TRAIN MODEL** | Pick a saved dataset, a model, and an output form. Trains the model on an 80/20 split and writes a PDF report. |

| **PREDICT** | Pick a saved dataset (or new raw files + a processor) and a trained model, and writes predictions to a CSV. |

  

Every tab has its own console (the `QPlainTextEdit` at the bottom) that

logs each step — e.g. `Dataset successfully saved in ...` or

`Dataset name should be unique.` — so the user always knows what happened

without needing pop-up dialogs.

  

## Project layout

  

```

appui.py Qt Designer generated UI (do not hand-edit — regenerate

it from appui.ui if you change the design)

main.py All application logic, wired onto appui.Ui_Form

requirements.txt

examples/

example_data_processor.py

example_model.py

example_output_form.py

```

  

On first launch the app creates an `AppData/` folder **next to the

script (or next to the .exe, see below)**:

  

```

AppData/

datasets/ processed data, one .csv per dataset

data_processors/ your process() scripts

models/ your build_model() scripts

trained_models/ .pkl files produced by the Train tab

output_forms/ your generate_report() scripts

outputs/ PDF reports + prediction .xlsx files land here

```

  

This folder is where "existing dataset / existing model / existing

processor / existing output form / existing trained model" dropdowns

read their options from. It's gitignored — don't commit it.

  

## Quick start

  

```bash

python  -m  venv  venv

source  venv/bin/activate  # venv\Scripts\activate on Windows

pip  install  -r  requirements.txt

python  main.py

```

  

## Writing your own plugins

  

The app never imports your scripts at build time — it loads them

dynamically (`importlib`) when you pick them in the UI. Each type has a

one-function contract:

  

### `data_processor.py`

  

```python

import pandas as pd

  

def  process(file_path):

df = pd.read_excel(file_path) # or read_csv, depending on file_path

# ... clean / engineer features ...

# Rename your actual target column to 'target' (Required by the pipeline) 
df = df.rename(columns={'your_actual_target_col': 'target'})

return df

```

  

Called once per file when you select **multiple** files in the DATA or

PREDICT tab's browse dialog; the app loops over your files, calls

`process()` on each, and `pd.concat()`s the results before saving.

  

### `model.py`

  

```python

from sklearn.ensemble import RandomForestRegressor

  

TARGET_COLUMN = "target"  # column name to predict

  

def  build_model():

return RandomForestRegressor(n_estimators=300, random_state=42)



def get_param_grid():
    """
    Defines the hyperparameter search space for use in GridSearchCV.
    
    Note:
    - If you want the model to be trained with default values without hyperparameter tuning,
      simply return an empty dictionary {} or comment out this function.
    """
    param_grid = {
        "n_estimators": [50, 100, 200],         # Number of decision trees
        "max_depth": [None, 5, 10, 15],         # Maximum tree depth to prevent overfitting
    }
    return param_grid
```

  

`build_model()` must return an **unfitted** estimator with `.fit(X, y)`

and `.predict(X)` — any scikit-learn estimator or Pipeline works.

Only one model file can be selected at a time (the model Browse dialog

is single-select).

  

### `output_form.py`

  

```python

def  generate_report(model, X_train, X_test, y_train, y_test, output_path):

# compute metrics (R2, MAE, ...), make plots / SHAP analysis,

# and save everything into output_path (a .pdf path)

...

```

  

Called right after training finishes. `output_path` always ends in

`.pdf` and lives inside `AppData/outputs/`.

  

See the full working versions of all three in `examples/` — the model

one trains a `RandomForestRegressor`, and the output form one writes a

3-page PDF with R2/MAE/RMSE, a predicted-vs-actual scatter plot, and a

SHAP summary plot.

  

## Notes / limitations (kept simple on purpose)

  

- Datasets are stored as plain CSV, so exotic dtypes (categoricals,

datetimes) round-trip as text — encode what you need inside your

data processor or model pipeline.

- Training runs synchronously on the UI thread; for very large datasets

the window will look frozen until it's done. If that becomes a

problem, move `on_train` into a `QThread`.

- Any package your custom scripts import (e.g. `xgboost`) must also be

installed in the environment (or bundled into the `.exe`, see below)

— the app doesn't pip-install things on your behalf.

  



## License

  

MIT License

Copyright (c) 2026 Hossein Fatemi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
