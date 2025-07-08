# fp_data_sains.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# --- Tahap 1: Load dan Prapemrosesan Data ---
df = pd.read_csv('obesitas.csv')

# Tambahkan fitur BMI
df['BMI'] = df['Weight'] / np.where(df['Height'] == 0, np.nan, df['Height'] ** 2)

# Cek nilai hilang
missing = df.isna().sum()
print('Missing values per kolom:')
print(missing[missing > 0])

# Korelasi numerik
num_cols = ['Age', 'Height', 'Weight', 'BMI']
plt.figure(figsize=(6,4))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='magma')
plt.title('Korelasi Numerik')
plt.tight_layout()
plt.show()

# Distribusi target
plt.figure(figsize=(6,3))
sns.countplot(x='NObeyesdad', data=df, order=df['NObeyesdad'].value_counts().index)
plt.title('Distribusi Kelas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Tahap 2: Pemodelan ---
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = [col for col in X.columns if col not in cat_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

model = RandomForestClassifier(random_state=42)

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', model)
])

param_dist = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10]
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)
rs.fit(X_train, y_train)

print('Best parameters:', rs.best_params_)

# --- Tahap 3: Evaluasi ---
y_pred = rs.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=rs.classes_)
fig, ax = plt.subplots(figsize=(6,6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rs.classes_).plot(
    ax=ax, xticks_rotation=45, cmap='Blues'
)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Simpan model
joblib.dump(rs.best_estimator_, 'obesity_pipeline.pkl')
print('Model saved as obesity_pipeline.pkl')
