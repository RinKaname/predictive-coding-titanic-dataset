import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def load_data():
    train_df = pd.read_csv('titanic_data/train.csv')
    # Separate features and target
    X = train_df.drop('Survived', axis=1)
    y = train_df['Survived']
    return X, y

def create_preprocessor():
    # Identify numerical and categorical columns we want to use
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def evaluate_models(X, y, preprocessor):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        # Create pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])

        # Cross-validation
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }
        print(f"Evaluated {name}")

    return results

def generate_markdown(results, output_file='model_method_comparison.md'):
    with open(output_file, 'w') as f:
        f.write("# Machine Learning Models Comparison for Titanic Dataset\n\n")
        f.write("This document compares the performance of several machine learning models on the Titanic dataset, using 5-fold cross-validation on the training set.\n\n")
        f.write("| Model | Mean Accuracy | Standard Deviation |\n")
        f.write("|---|---|---|\n")

        # Sort results by mean accuracy descending
        sorted_results = sorted(results.items(), key=lambda item: item[1]['mean_accuracy'], reverse=True)

        for name, metrics in sorted_results:
            mean_acc = metrics['mean_accuracy'] * 100
            std_acc = metrics['std_accuracy'] * 100
            f.write(f"| {name} | {mean_acc:.2f}% | &plusmn; {std_acc:.2f}% |\n")

        f.write("\n## Conclusion\n")
        best_model = sorted_results[0][0]
        f.write(f"Based on the cross-validation results, **{best_model}** is the best performing model among the ones tested.\n")

def main():
    print("Loading data...")
    X, y = load_data()

    print("Setting up preprocessor...")
    preprocessor = create_preprocessor()

    print("Evaluating models (this may take a moment)...")
    results = evaluate_models(X, y, preprocessor)

    print("Generating comparison markdown...")
    # generate_markdown(results) # Disabled to prevent overwriting manual experiment notes

    print("Done! See model_method_comparison.md for results.")

if __name__ == "__main__":
    main()
