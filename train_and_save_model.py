import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from deap import base, creator, tools, algorithms
import random
import joblib
import os

# --- GLOBALS ---
symptom_pool = []  # Will be updated dynamically
label_encoders = {}

# --- 1. LOAD AND ENCODE DATA ---
def load_data(file_path="Data.xlsx"):
    global label_encoders
    df = pd.read_excel(file_path)
    df.drop("Name", axis=1, inplace=True)

    # Label Encoding
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    return df

# --- 2. SYMPTOM MANAGER ---
def update_symptoms(df, symptoms_to_use=None):
    global symptom_pool
    if symptoms_to_use:
        symptom_pool = [s for s in symptoms_to_use if s in df.columns]
    else:
        symptom_pool = [col for col in df.columns if col != "Type"]
    print("🧠 Using Symptoms:", symptom_pool)

# --- 3. GA FEATURE SELECTION ---
def run_genetic_algorithm(X_train, y_train, X_test, y_test):
    def evaluate(individual):
        indices = [i for i, bit in enumerate(individual) if bit == 1]
        if not indices:
            return 0.0,
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train.iloc[:, indices], y_train)
        preds = model.predict(X_test.iloc[:, indices])
        return accuracy_score(y_test, preds),

    num_features = X_train.shape[1]
    random.seed(42)
    np.random.seed(42)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=30)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=False)

    best_ind = hof[0]
    selected_indices = [i for i, bit in enumerate(best_ind) if bit == 1]
    return selected_indices

# --- 4. FINAL MODEL TRAINING AND SAVE ---
def train_model(data, sprint_iterations=5):
    X = data[symptom_pool]
    y = data["Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for sprint in range(sprint_iterations):
        print(f"\n🚀 Sprint {sprint + 1} - Running GA and Training Model...")
        selected_indices = run_genetic_algorithm(X_train, y_train, X_test, y_test)

        X_train_sel = X_train.iloc[:, selected_indices]
        X_test_sel = X_test.iloc[:, selected_indices]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_sel, y_train)
        preds = model.predict(X_test_sel)

        selected_features = list(X_train.columns[selected_indices])
        print("\n✅ Selected Features by GA:", selected_features)
        print(f"\n✅ Sprint {sprint + 1} - Accuracy:", accuracy_score(y_test, preds))
        print(f"\nClassification Report:\n", classification_report(y_test, preds, target_names=label_encoders["Type"].classes_))

    # Save everything needed for prediction
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/thyroid_model.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(selected_features, "models/selected_features.pkl")
    print("✅ Model and metadata saved!")

# --- MAIN DRIVER ---
if __name__ == "__main__":
    df = load_data("Data.xlsx")

    update_symptoms(df)  # you can hardcode or allow user selection later
    train_model(df, sprint_iterations=5)
