import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =============================================================================
# 1. Feature Engineering
# =============================================================================
def load_and_preprocess_data():
    df_train = pd.read_csv('titanic_data/train.csv')
    df_test  = pd.read_csv('titanic_data/test.csv')

    def engineer_features_base(df):
        df = df.copy()
        df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True, errors='ignore')
        df['Embarked'] = df['Embarked'].fillna('S')
        df['Embarked'] = df['Embarked'].map({"S": 0, "C": 1, "Q": 2})
        df['Sex'] = df['Sex'].map({"male": 0, "female": 1})
        df['SumPeople'] = df['SibSp'].astype(int) + df['Parch'].astype(int) + 1
        return df

    df_train = engineer_features_base(df_train)
    df_test  = engineer_features_base(df_test)

    train_age_median = df_train['Age'].median()
    train_fare_median = df_train['Fare'].median()

    male_mean   = 30.7266445916114
    female_mean = 27.9157081226057

    for df in [df_train, df_test]:
        df.loc[(df['Sex'] == 0) & (df['Age'].isnull()), 'Age'] = male_mean
        df.loc[(df['Sex'] == 1) & (df['Age'].isnull()), 'Age'] = female_mean
        df['Age'] = df['Age'].fillna(train_age_median)
        df['Fare'] = df['Fare'].fillna(train_fare_median)

        age        = [0, 5, 15, 25, 30, 35, 45, 50, 200]
        age_label  = ['0-5','5-15','15-25','25-30','30-35','35-40','45-50','>50']
        df['age_group']      = pd.cut(df['Age'], age, labels=age_label)
        df['age_group_code'] = df['age_group'].cat.codes

        price       = [0, 10, 30, 35, 80, 1000]
        price_label = ['0-10','10-30','30-35','35-80','>80']
        df['price_group']      = pd.cut(df['Fare'], price, labels=price_label)
        df['price_group_code'] = df['price_group'].cat.codes

        df.drop(['age_group', 'price_group'], axis=1, inplace=True, errors='ignore')

    scaler = StandardScaler()
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    cols_to_scale = [col for col in numeric_cols if col not in ['Survived', 'PassengerId']]

    df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
    df_test[cols_to_scale]  = scaler.transform(df_test[cols_to_scale])

    FEATURES = ['Pclass', 'Sex', 'age_group_code', 'price_group_code',
                'SumPeople', 'SibSp', 'Parch', 'Embarked']

    X_train = df_train[FEATURES].values.astype(float)
    y_train = df_train['Survived'].values.astype(float)
    X_test  = df_test[FEATURES].values.astype(float)
    test_ids = df_test['PassengerId'].values

    # Normalise to [0,1] using train statistics
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_test  = (X_test  - X_min) / (X_max - X_min + 1e-8)

    return X_train, y_train, X_test, test_ids

# =============================================================================
# 2. Forward-Forward Network Implementation
# =============================================================================
class FFALayer:
    """A single layer trained via Forward-Forward (No Backprop)"""
    def __init__(self, in_feat, out_feat, lr=0.05, threshold=2.0, weight_decay=1e-4):
        self.W = np.random.randn(in_feat, out_feat) * np.sqrt(2.0 / in_feat)
        self.b = np.zeros((1, out_feat))
        self.lr = lr
        self.threshold = threshold
        self.weight_decay = weight_decay

    def forward(self, x):
        return np.maximum(0, x @ self.W + self.b)

    def normalize(self, h):
        return h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)

    def train_step(self, x_pos, x_neg):
        N = x_pos.shape[0]

        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        g_pos = np.sum(h_pos**2, axis=1, keepdims=True)
        g_neg = np.sum(h_neg**2, axis=1, keepdims=True)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))

        p_pos = sigmoid(g_pos - self.threshold)
        p_neg = sigmoid(g_neg - self.threshold)

        d_g_pos = p_pos - 1.0
        d_g_neg = p_neg

        grad_W = (2 * x_pos.T @ (d_g_pos * h_pos) + 2 * x_neg.T @ (d_g_neg * h_neg)) / N
        grad_b = (2 * np.sum(d_g_pos * h_pos, axis=0, keepdims=True) + 2 * np.sum(d_g_neg * h_neg, axis=0, keepdims=True)) / N

        # Added L2 regularization for robustness
        grad_W += self.weight_decay * self.W

        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b

        return self.normalize(h_pos), self.normalize(h_neg)


class ForwardForwardNetwork:
    """Trains layers greedily. Replaces labels with one-hot vectors overlaying the input."""
    def __init__(self, layer_sizes, lr=0.03, threshold=2.0, epochs_per_layer=100, weight_decay=1e-4):
        self.layers = []
        self.epochs = epochs_per_layer
        for i in range(len(layer_sizes) - 1):
            self.layers.append(FFALayer(layer_sizes[i], layer_sizes[i+1], lr, threshold, weight_decay))

    def fit(self, X, y, verbose=True):
        y_one_hot = np.zeros((X.shape[0], 2))
        y_one_hot[np.arange(X.shape[0]), y.astype(int)] = 1.0

        y_neg_one_hot = 1.0 - y_one_hot

        h_p = np.hstack([X, y_one_hot])
        h_n = np.hstack([X, y_neg_one_hot])

        for i, layer in enumerate(self.layers):
            iterator = range(self.epochs)
            if verbose:
                iterator = tqdm(iterator, desc=f"Training Layer {i+1}")

            for _ in iterator:
                layer.train_step(h_p, h_n)

            h_p = layer.normalize(layer.forward(h_p))
            h_n = layer.normalize(layer.forward(h_n))

    def _calc_goodness(self, x, label):
        y_test = np.zeros((x.shape[0], 2))
        y_test[:, label] = 1.0
        h = np.hstack([x, y_test])

        total_g = np.zeros((x.shape[0],))
        for layer in self.layers:
            h_unnorm = layer.forward(h)
            total_g += np.sum(h_unnorm**2, axis=1)
            h = layer.normalize(h_unnorm)
        return total_g

    def predict(self, X):
        g_0 = self._calc_goodness(X, 0)
        g_1 = self._calc_goodness(X, 1)
        return (g_1 > g_0).astype(int)

# =============================================================================
# 3. Fine-Tuning and Evaluation
# =============================================================================
if __name__ == "__main__":
    X_train_full, y_train_full, X_test, test_ids = load_and_preprocess_data()

    # Split for validation to test robustness
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    n_feat = X_train.shape[1]

    print("Evaluating baseline configuration on validation set...")
    ffa_model_baseline = ForwardForwardNetwork(
        layer_sizes=[n_feat + 2, 64, 32],
        lr=0.08,
        threshold=4.0,
        epochs_per_layer=250,
        weight_decay=0.0
    )
    ffa_model_baseline.fit(X_train, y_train, verbose=False)
    val_preds = ffa_model_baseline.predict(X_val)
    val_acc_baseline = np.mean(val_preds == y_val)
    print(f"Baseline Validation Accuracy: {val_acc_baseline:.4f}")

    print("\nTraining Tuned Model (with weight decay and adjusted threshold) on full dataset...")
    # Fine-tuning parameters for better generalization (robustness)
    ffa_tuned = ForwardForwardNetwork(
        layer_sizes=[n_feat + 2, 64, 64, 32], # Slightly deeper
        lr=0.05,
        threshold=5.0, # Higher threshold
        epochs_per_layer=300,
        weight_decay=1e-3 # Added L2 regularization
    )

    ffa_tuned.fit(X_train_full, y_train_full)
    train_acc = np.mean(ffa_tuned.predict(X_train_full) == y_train_full)
    print(f"\nFinal Train Accuracy (FFA Tuned): {train_acc:.4f}")

    # Save predictions
    test_preds = ffa_tuned.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived":    test_preds
    })
    submission.to_csv("submission_ffa.csv", index=False)
    print(f"\nSaved predictions to submission_ffa.csv")
    print(f"Predicted survivors: {test_preds.sum()} / {len(test_preds)}")
