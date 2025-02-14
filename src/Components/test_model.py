import pandas as pd
from src.utils import load_object

# Load model and preprocessor
model_path = "artifacts/model.pkl"
preprocessor_path = "artifacts/preprocessor.pkl"

try:
    model = load_object(model_path)
    preprocessor = load_object(preprocessor_path)
    print("Model and Preprocessor Loaded Successfully!")
except Exception as e:
    print(f"Error loading model/preprocessor: {e}")
    exit()

# Example input (Make sure these are valid values)
sample_data = pd.DataFrame({
    "gender": ["male"],
    "race/ethnicity": ["group B"],
    "parental level of education": ["bachelor's degree"],
    "lunch": ["standard"],
    "test preparation course": ["none"],
    "reading score": [72],
    "writing score": [74]
})

try:
    # Transform the input data
    sample_data_transformed = preprocessor.transform(sample_data)
    print("Transformed Features:", sample_data_transformed[:5])  # Print first 5 rows

    # Predict
    pred = model.predict(sample_data_transformed)
    print("Sample Prediction:", pred)
except Exception as e:
    print(f"Error during prediction: {e}")
