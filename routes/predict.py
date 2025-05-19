from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
import os

# Load model and encoders
model_path = os.path.join("models", "thyroid_model.pkl")
encoder_path = os.path.join("models", "label_encoders.pkl")
features_path = os.path.join("models", "selected_features.pkl")

model = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)
selected_features = joblib.load(features_path)

predict_bp = Blueprint('predict_bp', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.get_json()
        input_features = user_input.get("selectedFeatures", {})

        # Fill missing features with default value ("No" for categorical, 0 for Age if needed)
        input_dict = {}
        for feature in selected_features:
            if feature in input_features:
                input_dict[feature] = input_features[feature]
            else:
                input_dict[feature] = 0 if feature == "Age" else "no"

        input_df = pd.DataFrame([input_dict])
        print("Raw input to model:\n",input_df)
        # Label encode where necessary
        for col in input_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                input_df[col] = le.transform(input_df[col])

        prediction = model.predict(input_df)[0]
        prediction_label = label_encoders["Type"].inverse_transform([prediction])[0]

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)})
