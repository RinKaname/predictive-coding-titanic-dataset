import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Import Standard ML definitions
from titanic_ml_comparison import load_data, create_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Import Alternative architectures
import predictive_coding_experiment as pcn
import forward_forward_experiment as ffa

def load_ground_truth():
    gt = pd.read_csv('ground_truth.csv')
    return gt.sort_values(by='PassengerId')

def get_standard_ml_predictions(test_df):
    X_train, y_train = load_data()
    preprocessor = create_preprocessor()

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    predictions = {}
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        clf.fit(X_train, y_train)
        predictions[name] = clf.predict(test_df)

    return predictions

def main():
    print("Loading Ground Truth...")
    gt_df = load_ground_truth()
    y_true = gt_df['Survived'].values
    test_ids = gt_df['PassengerId'].values

    # We need the raw test df for standard ML models
    test_df_raw = pd.read_csv('titanic_data/test.csv')
    # ensure it's sorted same as ground truth just in case
    test_df_raw = test_df_raw.sort_values(by='PassengerId').reset_index(drop=True)

    results = []

    print("Evaluating Standard ML models...")
    std_ml_preds = get_standard_ml_predictions(test_df_raw)
    for name, preds in std_ml_preds.items():
        acc = accuracy_score(y_true, preds)
        results.append({"Model": name, "Ground Truth Accuracy": acc, "Type": "Standard ML"})
        print(f"  {name}: {acc:.4f}")

    print("Evaluating Predictive Coding Network (PCN Version 3)...")
    # pcn.load_and_preprocess_data handles the specific feature engineering logic for the PCN
    X_train_pcn, y_train_pcn = pcn.load_and_engineer()
    def pcn_load_test():
        df_test = pd.read_csv("titanic_data/test.csv")
        df_test["Embarked"] = df_test["Embarked"].fillna("S")
        df_test["Embarked"] = df_test["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})
        df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())
        df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())
        features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        X = df_test[features].values.astype(float)
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
        return X
    X_test_pcn = pcn_load_test()

    # Train PCN
    model_pcn = pcn.PredictiveCodingNetwork(
        layer_sizes=[X_train_pcn.shape[1], 32, 16, 1],
        lr_weights=0.001,
        lr_activities=0.01,
        lr_precision=0.0001,
        n_inference_steps=30
    )
    model_pcn.fit(X_train_pcn, y_train_pcn, epochs=20, verbose=False)
    pcn_preds = model_pcn.predict(X_test_pcn)

    acc_pcn = accuracy_score(y_true, pcn_preds)
    results.append({"Model": "Predictive Coding Network (V3)", "Ground Truth Accuracy": acc_pcn, "Type": "Alternative"})
    print(f"  Predictive Coding Network (V3): {acc_pcn:.4f}")

    print("Evaluating Forward-Forward Algorithm (FFA Tuned)...")
    X_train_ffa, y_train_ffa, X_test_ffa, _ = ffa.load_and_preprocess_data()

    model_ffa = ffa.ForwardForwardNetwork(
        layer_sizes=[X_train_ffa.shape[1] + 2, 64, 64, 32],
        lr=0.05,
        threshold=5.0,
        epochs_per_layer=300,
        weight_decay=1e-3
    )
    model_ffa.fit(X_train_ffa, y_train_ffa, verbose=False)
    ffa_preds = model_ffa.predict(X_test_ffa)

    acc_ffa = accuracy_score(y_true, ffa_preds)
    results.append({"Model": "Forward-Forward Algorithm (FFA)", "Ground Truth Accuracy": acc_ffa, "Type": "Alternative"})
    print(f"  Forward-Forward Algorithm (FFA): {acc_ffa:.4f}")

    print("\n--- Final Results ---")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Ground Truth Accuracy", ascending=False).reset_index(drop=True)

    print(results_df.to_string())
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved comprehensive results to evaluation_results.csv")

if __name__ == "__main__":
    main()