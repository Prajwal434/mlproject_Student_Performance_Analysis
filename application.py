from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    print("Request received")  # Check if the function is running

    if request.method == 'GET':
        print("GET request received")  # Debugging
        return render_template('home.html')

    else:
        print("POST request received")  # Debugging
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            pred_df = data.get_data_as_data_frame()
            print("Data received:\n", pred_df)  # Debugging

            predict_pipeline = PredictPipeline()
            print("Predict pipeline initialized")  # Debugging

            results = predict_pipeline.predict(pred_df)
            print("Prediction result:", results)  # Debugging

            return render_template('home.html', results=results[0])

        except Exception as e:
            print("Error occurred:", str(e))  # Catch and print errors
            return "An error occurred: " + str(e), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0")  # Enable debug mode for better error tracking
