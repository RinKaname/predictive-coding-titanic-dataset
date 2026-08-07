import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_and_engineer():
    df_train = pd.read_csv('titanic_data/train.csv')
    df_train['Embarked'] = df_train['Embarked'].fillna('S')
    df_train['Embarked'] = df_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df_train['Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})
    df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
    df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].median())

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df_train[features].values.astype(float)
    y = df_train['Survived'].values.astype(float)

    X = StandardScaler().fit_transform(X)
    return X, y



# ── Precision-Weighted Predictive Coding Network ─────────────────────────────
class PredictiveCodingNetwork:
    """
    Hierarchical Predictive Coding Network with Dynamic Precision (Free Energy Principle).
    - Top-down connections generate predictions.
    - Precision units (Pi) dynamically weight errors based on reliability.
    - Weight updates are LOCAL — no global backpropagation.
    """
    def __init__(self, layer_sizes, lr_weights=0.005, lr_activities=0.05,
                 lr_precision=0.001, n_inference_steps=30):
        self.sizes   = layer_sizes
        self.lr_w    = lr_weights
        self.lr_a    = lr_activities

        # New: Precision learning rate
        self.lr_p    = lr_precision

        self.n_inf   = n_inference_steps
        self.n_layers = len(layer_sizes)

        # Top-down weight matrices: W[l] predicts layer l from layer l+1
        self.W = [np.random.randn(layer_sizes[l], layer_sizes[l+1]) * 0.1
                  for l in range(self.n_layers - 1)]
        self.b = [np.zeros(layer_sizes[l])
                  for l in range(self.n_layers - 1)]

        # New: Log-precision (initialized to 0, meaning precision Pi = exp(0) = 1)
        # We learn log_pi so that true precision exp(log_pi) remains strictly positive.
        self.log_pi = [np.zeros(layer_sizes[l])
                       for l in range(self.n_layers - 1)]

    def _f(self, x):   return np.tanh(x)
    def _df(self, x):  return 1.0 - np.tanh(x) ** 2
    def _sig(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _predict_down(self, r_above, l):
        return self._f(self.W[l] @ r_above + self.b[l])

    def _infer(self, x, y=None):
        r = [np.zeros(s) for s in self.sizes]
        r[0] = x.copy()

        # Warm-start: bottom-up sweep
        for l in range(self.n_layers - 1):
            r[l+1] = self._f(self.W[l].T @ r[l])

        # Pin label if training
        if y is not None:
            r[-1] = np.array([y]).astype(float)

        # Inference loop — minimise free energy in hidden layers
        for _ in range(self.n_inf):
            # 1. Calculate raw errors
            errors = [r[l] - self._predict_down(r[l+1], l)
                      for l in range(self.n_layers - 1)]

            # 2. Calculate Precision-weighted errors
            # This is the "attention" mechanism: scaling errors by their learned reliability
            w_errors = [errors[l] * np.exp(np.clip(self.log_pi[l], -10, 10))
                        for l in range(self.n_layers - 1)]

            for l in range(1, self.n_layers - 1):
                # 3. Update activities using the WEIGHTED errors instead of raw errors
                grad = (w_errors[l]
                        - self.W[l-1].T @ (w_errors[l-1] * self._df(self.W[l-1] @ r[l] + self.b[l-1])))
                # Using gradient clipping from previous step to avoid NaNs
                r[l] -= self.lr_a * grad

        # Recalculate raw errors for the weight update step
        errors = [r[l] - self._predict_down(r[l+1], l)
                  for l in range(self.n_layers - 1)]
        return r, errors

    def _update(self, r, errors):
        for l in range(self.n_layers - 1):
            # Calculate current precision Pi
            Pi = np.exp(np.clip(self.log_pi[l], -10, 10))

            # Create the weighted error for updating W and b
            w_error = errors[l] * Pi

            # Weights now update based on reliable signals, ignoring noise (with clipping)
            self.W[l] += self.lr_w * np.clip(np.outer(w_error, r[l+1]), -1, 1)
            self.b[l] += self.lr_w * np.clip(w_error, -1, 1)

            # Learn the precision!
            # The derivative of Free Energy w.r.t log_pi is proportional to (Pi * error^2 - 1)
            # If error is large, Pi * e^2 > 1, so log_pi decreases (lowers precision/attention).
            # If error is small, Pi * e^2 < 1, so log_pi increases (raises precision/attention).
            self.log_pi[l] -= self.lr_p * np.clip((Pi * (errors[l] ** 2) - 1.0), -1, 1)

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

def main():
    print("Loading and engineering features...")
    X_train, y_train = load_and_engineer()
    n_feat = X_train.shape[1]

    print("\n--- Running: Precision-Weighted PCN (User Config) ---")
    np.random.seed(42)
    model = PredictiveCodingNetwork(
        layer_sizes=[n_feat, 32, 16, 1],
        lr_weights=0.0005,
        lr_activities=0.05,
        n_inference_steps=30
    )


    model.fit(X_train, y_train, epochs=10, verbose=False)
    preds = model.predict(X_train)
    train_acc = np.mean(preds == y_train)



    print(f"Final Train Accuracy: {train_acc:.4f}")



if __name__ == "__main__":
    main()
