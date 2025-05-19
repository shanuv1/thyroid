from flask import Flask, render_template
import joblib
from routes.predict import predict_bp

app = Flask(__name__)
app.register_blueprint(predict_bp)

# Load selected features for frontend
selected_features = joblib.load("models/selected_features.pkl")

@app.route("/")
def home():
    return render_template("index.html", features=selected_features)

if __name__ == "__main__":
    app.run(debug=True)
