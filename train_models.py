
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

CSV_PATH = Path("data/raw/loan_approval_dataset.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {CSV_PATH}. "
        "Check the CSV filename and folder."
    )


# --------------------------------------------------
# 2. Load and inspect the dataset
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Remove extra spaces from text values
for column in df.select_dtypes(include=["object"]).columns:
    df[column] = df[column].map(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

if "loan_status" not in df.columns:
    raise ValueError(
        "Could not find 'loan_status'. "
        f"Available columns: {df.columns.tolist()}"
    )

print("\n--- Original dataset ---")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nExact duplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 3. Basic data cleaning
# --------------------------------------------------

# Remove completely identical rows
df = df.drop_duplicates().copy()

# Remove rows whose target is missing
df = df.dropna(subset=["loan_status"]).copy()

# Normalize the target labels
target_text = (
    df["loan_status"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Define our convention:
# Approved = 1
# Rejected = 0
target_mapping = {
    "approved": 1,
    "rejected": 0,
    "1": 1,
    "0": 0,
    "1.0": 1,
    "0.0": 0,
    "yes": 1,
    "no": 0,
    "y": 1,
    "n": 0,
}

converted_target = target_text.map(target_mapping)

if converted_target.isna().any():
    unknown_labels = target_text[
        converted_target.isna()
    ].unique().tolist()

    raise ValueError(
        "Unexpected loan_status values: "
        f"{unknown_labels}. Check the target labels."
    )

df["loan_status"] = converted_target.astype(int)

# Drop identifiers and accidental CSV index columns
id_columns = [
    column
    for column in df.columns
    if column in {"loan_id", "id"}
    or column.startswith("unnamed:")
]

df = df.drop(columns=id_columns, errors="ignore")

print("\n--- Cleaned dataset ---")
print("Shape:", df.shape)

print("\nTarget distribution:")
print(
    df["loan_status"]
    .map({0: "Rejected", 1: "Approved"})
    .value_counts()
)


# --------------------------------------------------
# 4. Separate input features (X) and target (y)
# --------------------------------------------------

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

if X.empty:
    raise ValueError("No input features remain after cleaning.")

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# --------------------------------------------------
# 5. Split the data BEFORE fitting preprocessing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 6. Create a fresh preprocessing pipeline
# --------------------------------------------------

def make_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# --------------------------------------------------
# 7. Define the three classification models
# --------------------------------------------------

model_specs = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=3000),
    ),
    (
        "LDA",
        LinearDiscriminantAnalysis(),
    ),
    (
        "QDA",
        QuadraticDiscriminantAnalysis(reg_param=0.1),
    ),
]


# --------------------------------------------------
# 8. Set up five-fold cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# --------------------------------------------------
# 9. Train and evaluate each model
# --------------------------------------------------

trained_models = {}
cv_rows = []
test_rows = []
test_probabilities = {}

for model_name, classifier in model_specs:
    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", classifier),
        ]
    )

    # Cross-validation uses only the training set
    cv_result = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=cv_scoring,
        n_jobs=-1,
    )

    cv_row = {"Model": model_name}

    for metric in cv_scoring:
        scores = cv_result[f"test_{metric}"]

        cv_row[f"CV_{metric}_mean"] = scores.mean()
        cv_row[f"CV_{metric}_std"] = scores.std()

    cv_rows.append(cv_row)

    # Fit the model on the full training split
    pipeline.fit(X_train, y_train)

    # Predict the held-out test set
    y_pred = pipeline.predict(X_test)

    # Find the probability column belonging to class 1
    classifier_classes = (
        pipeline.named_steps["classifier"].classes_
    )
    positive_index = list(classifier_classes).index(1)

    y_probability = pipeline.predict_proba(X_test)[
        :, positive_index
    ]

    test_probabilities[model_name] = y_probability
    trained_models[model_name] = pipeline

    # Evaluate predictions with approval as the positive class
    test_row = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, pos_label=1, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, y_probability),
    }

    test_rows.append(test_row)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Rejected", "Approved"],
            zero_division=0,
        )
    )

    # Save one confusion matrix per model
    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0, 1],
        display_labels=["Rejected", "Approved"],
        cmap="Blues",
        ax=ax,
    )

    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    fig.savefig(
        OUTPUT_DIR / f"{filename}_confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)


# --------------------------------------------------
# 10. Save and display comparison tables
# --------------------------------------------------

cv_table = pd.DataFrame(cv_rows).sort_values(
    "CV_f1_mean",
    ascending=False,
)

test_table = pd.DataFrame(test_rows).sort_values(
    "F1",
    ascending=False,
)

cv_table.to_csv(
    OUTPUT_DIR / "model_comparison_cv.csv",
    index=False,
)

test_table.to_csv(
    OUTPUT_DIR / "model_comparison_test.csv",
    index=False,
)

print("\n--- Five-fold CV results (training data only) ---")
print(cv_table.round(4).to_string(index=False))

print("\n--- Held-out test results ---")
print(test_table.round(4).to_string(index=False))


# --------------------------------------------------
# 11. Plot ROC curves for all three models
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

for model_name, probabilities in test_probabilities.items():
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=model_name,
        ax=ax,
    )

ax.set_title("ROC Curve Comparison")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "roc_curve_comparison.png",
    dpi=150,
)
plt.close(fig)


# --------------------------------------------------
# 12. Interpret Logistic Regression coefficients
# --------------------------------------------------

lr_pipeline = trained_models["Logistic Regression"]

feature_names = (
    lr_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    lr_pipeline
    .named_steps["classifier"]
    .coef_[0]
)

coefficient_table = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Odds_Ratio": np.exp(
            np.clip(coefficients, -50, 50)
        ),
    }
)

coefficient_table["Absolute_Coefficient"] = (
    coefficient_table["Coefficient"].abs()
)

coefficient_table = coefficient_table.sort_values(
    "Absolute_Coefficient",
    ascending=False,
)

coefficient_table.drop(
    columns=["Absolute_Coefficient"]
).to_csv(
    OUTPUT_DIR / "logistic_regression_coefficients.csv",
    index=False,
)

print("\n--- Largest Logistic Regression coefficients ---")
print(
    coefficient_table.head(10)
    .drop(columns=["Absolute_Coefficient"])
    .round(4)
    .to_string(index=False)
)


# --------------------------------------------------
# 13. Save the chosen model
# --------------------------------------------------

# Choose using cross-validation F1, not test-set F1.
best_model_name = cv_table.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    OUTPUT_DIR / "best_model.joblib",
)

print("\nBest model by mean CV F1:", best_model_name)
print("Saved model to outputs/best_model.joblib")

print("\nAll done. Check the outputs/ folder.")
