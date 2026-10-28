
import joblib
import pandas as pd

# Load the saved pipeline, including its preprocessing steps
model = joblib.load("outputs/best_model.joblib")

# Example application: replace these values with your own
# test record, using the same units as the training dataset.
new_applicant = pd.DataFrame([
    {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 6000000,
        "loan_amount": 15000000,
        "loan_term": 10,
        "cibil_score": 750,
        "residential_assets_value": 5000000,
        "commercial_assets_value": 2000000,
        "luxury_assets_value": 3000000,
        "bank_asset_value": 1000000,
    }
])

# Predict the class
prediction = int(model.predict(new_applicant)[0])

# Obtain the estimated probability of approval
classes = list(model.named_steps["classifier"].classes_)
approved_index = classes.index(1)

approved_probability = model.predict_proba(
    new_applicant
)[0, approved_index]

if prediction == 1:
    print("Predicted class: Approved")
else:
    print("Predicted class: Rejected")

print(
    f"Estimated approval probability: "
    f"{approved_probability:.2%}"
)
