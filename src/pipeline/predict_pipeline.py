import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading Model and Preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading Model and Preprocessor")

            # Print column names before transformation
            print("Columns before transformation:", features.columns.tolist())

            # Expected column names in the model
            expected_columns = {
                "race or ethnicity": "race/ethnicity",
                "parental_level_of_education": "parental level of education",
                "test_preparation_course": "test preparation course"
            }

            # Rename columns to match the expected format
            features.rename(columns=expected_columns, inplace=True)

            # Print column names after renaming
            print("Columns after renaming:", features.columns.tolist())

            # Check if all expected columns exist
            missing_columns = [col for col in expected_columns.values() if col not in features.columns]
            if missing_columns:
                raise CustomException(f"Missing expected columns: {missing_columns}")

            # Transform input data
            data_scaled = preprocessor.transform(features)

            # Make predictions
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            df = pd.DataFrame(custom_data_input_dict)

            # Print final DataFrame for debugging
            print("\nGenerated Input DataFrame:\n", df)

            return df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Create a test input
    test_data = CustomData(
        gender="male",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        reading_score=72,
        writing_score=74,
    )

    # Convert input to DataFrame
    input_df = test_data.get_data_as_data_frame()

    # Create a prediction pipeline
    pipeline = PredictPipeline()

    # Call predict function
    prediction = pipeline.predict(input_df)

    # Print the final prediction
    print("\n🎯 Prediction Output:", prediction)
