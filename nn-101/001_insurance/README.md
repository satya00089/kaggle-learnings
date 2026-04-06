# Insurance Cross-Sell — Neural Network (Binary Classification)

Overview
--------
Build a neural network to predict whether a customer will respond positively to an
insurance cross-sell offer. This is a binary classification task using tabular
customer data (demographics, vehicle, premiums, previous insurance status).

Dataset
-------
Use the included CSVs in `data/` (e.g., `train.csv`) or the Kaggle dataset:
https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

Typical columns
---------------
- `age`
- `gender` (categorical)
- `income`
- `vehicle_age`
- `previously_insured` (binary)
- `annual_premium`
- `response` (target: yes/no or 1/0)

Project goals
-------------
- Train a Keras/TensorFlow MLP that outputs a probability (sigmoid).
- Use proper preprocessing (encoding + scaling).
- Evaluate with accuracy, precision, recall, F1-score (and ROC AUC).
- Answer experimental questions about feature importance, scaling, and model capacity.

What to practice / checklist
---------------------------
- Label-encode or one-hot encode categorical features.
- Scale numeric features (e.g., `StandardScaler` / `MinMaxScaler`).
- Split data into train / validation / test sets.
- Build an MLP with a sigmoid output and `binary_crossentropy` loss.
- Use callbacks like `EarlyStopping` and `ModelCheckpoint`.
- Report `accuracy`, `precision`, `recall`, `f1-score`, and confusion matrix.

Suggested workflow
------------------
1. Create a reproducible environment: Python 3.8+ and common ML packages.
	```bash
	pip install numpy pandas scikit-learn tensorflow matplotlib seaborn shap
	```
2. Load `data/train.csv` and inspect for missing values and class balance.
3. Preprocess:
	- One-hot encode nominal categories (or use embedding approaches).
	- Label-encode binary flags.
	- Scale numeric features.
4. Split into train / val / test (e.g., 70/15/15).
5. Build a small MLP, compile with `optimizer='adam'`, `loss='binary_crossentropy'`.
6. Train with early stopping and monitor validation loss.
7. Evaluate on the test set (classification report + ROC AUC).
8. Interpret results (permutation importance or SHAP for feature importance).

Quick Keras example
-------------------
```py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)

model = Sequential([
	 Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
	 Dropout(0.3),
	 Dense(32, activation='relu'),
	 Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es])
```

Experiments & questions to answer
---------------------------------
- Which features matter most? (use SHAP or permutation importance)
- Does scaling numeric features improve performance? Compare scaled vs unscaled.
- Is the model overfitting? (compare train/val loss curves)
- What happens if you add more hidden layers or units?

Tips and reproducibility
------------------------
- Set seeds (`numpy`, `random`, `tf`) for reproducible runs.
- If classes are imbalanced, try class weights or resampling.
- Persist your preprocessing pipeline (scaler/encoder) so inference matches training.

Files
-----
- `data/` — dataset CSVs (train/test)
- `README.md` — this file

