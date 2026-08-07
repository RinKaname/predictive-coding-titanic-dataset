import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Feature Engineering ───────────────────────────────────────────────────────
def load_and_engineer():
    df_train = pd.read_csv('titanic_data/train.csv')

    def engineer_features(df):
        df = df.copy()
        df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True, errors='ignore')

        df['Embarked'].fillna('S', inplace=True)
        df.replace({"Embarked": {"S": 0, "C": 1, "Q": 2}}, inplace=True)
        df.replace({"Sex": {"male": 0, "female": 1}}, inplace=True)

        male_mean   = 30.7266445916114
        female_mean = 27.9157081226057
        df.loc[(df['Sex'] == 0) & (df['Age'].isnull()), 'Age'] = male_mean
        df.loc[(df['Sex'] == 1) & (df['Age'].isnull()), 'Age'] = female_mean
        df['Age'].fillna(df['Age'].median(), inplace=True)

        age        = [0, 5, 15, 25, 30, 35, 45, 50, 200]
        age_label  = ['0-5','5-15','15-25','25-30','30-35','35-40','45-50','>50']
        df['age_group']      = pd.cut(df['Age'], age, labels=age_label)
        df['age_group_code'] = df['age_group'].cat.codes

        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        price       = [0, 10, 30, 35, 80, 1000]
        price_label = ['0-10','10-30','30-35','35-80','>80']
        df['price_group']      = pd.cut(df['Fare'], price, labels=price_label)
        df['price_group_code'] = df['price_group'].cat.codes

        df['SumPeople'] = df['SibSp'].astype(int) + df['Parch'].astype(int) + 1
        df.drop(['age_group', 'price_group'], axis=1, inplace=True, errors='ignore')

        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in ['Survived', 'PassengerId']]
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        return df

    df_train = engineer_features(df_train)
    y_train = df_train['Survived'].values.astype(float)
    X_train = df_train.drop(['Survived', 'PassengerId'], axis=1).values.astype(float)

    return X_train, y_train

# ── Predictive Coding Network ───────────────────────────────────────────────
class PredictiveCodingNetwork:
    def __init__(self, layer_sizes, lr_weights=0.001, lr_activities=0.05, n_inference_steps=30):
        self.sizes   = layer_sizes
        self.lr_w    = lr_weights
        self.lr_a    = lr_activities
        self.n_inf   = n_inference_steps
        self.n_layers = len(layer_sizes)

        self.W = [np.random.randn(layer_sizes[l], layer_sizes[l+1]) * 0.1
                  for l in range(self.n_layers - 1)]
        self.b = [np.zeros(layer_sizes[l])
                  for l in range(self.n_layers - 1)]

    def _f(self, x):   return np.tanh(x)
    def _df(self, x):  return 1.0 - np.tanh(x) ** 2
    def _sig(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _predict_down(self, r_above, l):
        return self._f(self.W[l] @ r_above + self.b[l])

    def _infer(self, x, y=None):
        r = [np.zeros(s) for s in self.sizes]
        r[0] = x.copy()

        for l in range(self.n_layers - 1):
            r[l+1] = self._f(self.W[l].T @ r[l])

        if y is not None:
            r[-1] = np.array([y])

        for _ in range(self.n_inf):
            errors = [r[l] - self._predict_down(r[l+1], l)
                      for l in range(self.n_layers - 1)]

            for l in range(1, self.n_layers - 1):
                grad = (errors[l]
                        - self.W[l-1].T @ (errors[l-1] * self._df(self.W[l-1] @ r[l] + self.b[l-1])))
                r[l] -= self.lr_a * grad

        errors = [r[l] - self._predict_down(r[l+1], l)
                  for l in range(self.n_layers - 1)]
        return r, errors

    def _update(self, r, errors):
        for l in range(self.n_layers - 1):
            self.W[l] -= self.lr_w * np.outer(errors[l], r[l+1])
            self.b[l] -= self.lr_w * errors[l]

    def fit(self, X, y, epochs=100, verbose=True):
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            correct = 0

            loop = tqdm(idx, disable=not verbose, desc=f"Epoch {epoch+1:3d}/{epochs}", leave=False)
            for i in loop:
                r, errors = self._infer(X[i], y[i])
                self._update(r, errors)
                prob = self._sig(self.W[-1].T @ r[-2])
                correct += int((prob[0] > 0.5) == bool(y[i]))

            if verbose:
                loop.set_postfix(Acc=f"{correct/len(X):.4f}")

    def predict(self, X):
        return np.array([
            int(self._sig(self.W[-1].T @ self._infer(x)[0][-2])[0] > 0.5)
            for x in X
        ])

# ── Experiment Loop ────────────────────────────────────────────────────────
def main():
    print("Loading and engineering features...")
    X_train, y_train = load_and_engineer()
    n_feat = X_train.shape[1]

    experiments = [
        {"epochs": 1, "lr_w": 0.005, "lr_a": 0.05, "layers": [n_feat, 32, 16, 1], "name": "Baseline (Notebook Config)"},
        {"epochs": 10, "lr_w": 0.005, "lr_a": 0.05, "layers": [n_feat, 32, 16, 1], "name": "More Epochs (10)"},
        {"epochs": 50, "lr_w": 0.005, "lr_a": 0.05, "layers": [n_feat, 32, 16, 1], "name": "High Epochs (50)"},
        {"epochs": 20, "lr_w": 0.01, "lr_a": 0.1, "layers": [n_feat, 32, 16, 1], "name": "Higher Learning Rates"},
        {"epochs": 20, "lr_w": 0.001, "lr_a": 0.01, "layers": [n_feat, 32, 16, 1], "name": "Lower Learning Rates"},
        {"epochs": 20, "lr_w": 0.005, "lr_a": 0.05, "layers": [n_feat, 64, 32, 1], "name": "Wider Network"},
        {"epochs": 20, "lr_w": 0.005, "lr_a": 0.05, "layers": [n_feat, 16, 8, 1], "name": "Narrower Network"}
    ]

    print("\nStarting Experiments...\n")

    for exp in experiments:
        print(f"--- Running: {exp['name']} ---")
        np.random.seed(42) # Reproducibility

        model = PredictiveCodingNetwork(
            layer_sizes=exp['layers'],
            lr_weights=exp['lr_w'],
            lr_activities=exp['lr_a'],
            n_inference_steps=30
        )

        model.fit(X_train, y_train, epochs=exp['epochs'], verbose=False)
        train_acc = np.mean(model.predict(X_train) == y_train)
        print(f"Final Train Accuracy: {train_acc:.4f}\n")

if __name__ == "__main__":
    main()
