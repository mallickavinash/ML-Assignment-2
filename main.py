import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils import compute_metrics

TARGET_COL = "target"

def get_preprocessor(numerical_features, categorical_features):
    """
    Creates and returns a fresh, un-fitted ColumnTransformer 
    configured for numerical and categorical pipelines.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ]
    )
    return preprocessor
TARGET_COL = "target"
df = pd.read_csv('C:/Users/Avinash/Downloads/bots_vs_users.csv')
df = df.dropna(subset=[TARGET_COL])
X = df.drop(columns=[TARGET_COL])
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL])

# Identify features dynamically
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
}

rows = []
for name, clf in classifiers.items():
    prep = get_preprocessor(num_cols, cat_cols)
    # 1. Package the preprocessor clone and classifier into a single container object
    pipe = Pipeline([
        ("prep", clone(prep)), 
        ("clf", clf)
    ])
    
    # 2. Fit the entire pipeline on the RAW features (X_train)
    # The pipeline internally handles prep.fit_transform() and passes it to clf.fit()
    pipe.fit(X_train, y_train)
    
    # 3. Extract metrics by passing RAW test data (X_test) directly into your metrics function
    m, _ = compute_metrics(pipe, X_test, y_test)
    rows.append({"ML Model Name": name, **m})
    
    # 4. Save the single pipeline object. The preprocessor is baked right inside it!
    joblib.dump(pipe, f"model/{name.replace(' ', '_').lower()}.joblib")
    print(f"trained {name}")

results = pd.DataFrame(rows).round(4)
results.to_csv("model/metrics_table.csv", index=False)
print(results.to_markdown(index=False))  # paste this into README

# Test file for the app: raw features + original (human-readable) target labels
test_df = X_test.copy()
test_df[TARGET_COL] = le.inverse_transform(y_test)
test_df.to_csv("test_data.csv", index=False)

joblib.dump({"features": X.columns.tolist(), "target": TARGET_COL,
             "label_encoder": le}, "model/meta.joblib")
