from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd

# Create a sample input
sample_features = pd.DataFrame({
    "gender": ["male"],
    "race/ethnicity": ["group C"],
    "parental level of education": ["bachelor's degree"],
    "lunch": ["standard"],
    "test preparation course": ["none"],
    "reading score": [80],
    "writing score": [75]
})

# Manually call predict()
pipeline = PredictPipeline()
print("🚀 Calling predict() function...")
preds = pipeline.predict(sample_features)
print("🟢 Prediction Output:", preds)
