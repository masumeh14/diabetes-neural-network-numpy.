"""Pima Indians diabetes prediction.

Trains a small Keras network (8 -> 8 -> 1, sigmoid), re-implements its forward
pass from scratch in NumPy (loop and vectorized versions), and studies how the
decision threshold trades precision against recall.

Run:  python diabetes_nn.py
"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

SEED = 42
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
# In this dataset a 0 in these columns means "not measured", not a real value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, header=None, names=COLUMNS)


def explore(df: pd.DataFrame) -> None:
    print(df.shape)
    print(df.head())
    print(df["Outcome"].value_counts())
    print(df.describe().T)
    print("zeros per column:\n", (df[ZERO_AS_MISSING] == 0).sum())


def preprocess(df: pd.DataFrame):
    """Clean, split, impute and normalize.

    Medians, means and standard deviations are computed on the training set
    only, so no information from the test set leaks into training.
    """
    df = df.copy()
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    print("missing values per column:\n", df.isna().sum())

    X = df.drop(columns="Outcome")
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(X_train.shape, X_test.shape)
    print("positive rate (train/test):", y_train.mean(), y_test.mean())

    medians = X_train.median()
    X_train = X_train.fillna(medians).to_numpy(dtype=float)
    X_test = X_test.fillna(medians).to_numpy(dtype=float)

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    Xn_train = (X_train - mu) / sigma
    Xn_test = (X_test - mu) / sigma
    print("train std after scaling:", Xn_train.std(axis=0).round(2))

    return Xn_train, Xn_test, y_train.to_numpy(), y_test.to_numpy()


# --------------------------------------------------------------------------
# Keras model
# --------------------------------------------------------------------------
def build_and_train(Xn_train, y_train):
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(8, activation="sigmoid", name="layer1"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="layer2"),
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )
    history = model.fit(
        Xn_train, y_train,
        epochs=100, batch_size=32, validation_split=0.2, verbose=0,
    )
    return model, history


# --------------------------------------------------------------------------
# Forward pass from scratch (NumPy)
# --------------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def dense_loop(a_in, W, b):
    """One dense layer, one neuron at a time."""
    a_out = np.zeros(W.shape[1])
    for j in range(W.shape[1]):
        a_out[j] = sigmoid(np.dot(W[:, j], a_in) + b[j])
    return a_out


def predict_loop(X, W1, b1, W2, b2):
    """Forward pass, one sample at a time."""
    p = np.zeros((X.shape[0], 1))
    for i, x in enumerate(X):
        a1 = dense_loop(x, W1, b1)
        p[i, 0] = dense_loop(a1, W2, b2)[0]
    return p


def dense_vec(A_in, W, b):
    return sigmoid(A_in @ W + b)


def predict_vec(X, W1, b1, W2, b2):
    """Forward pass for all samples at once."""
    return dense_vec(dense_vec(X, W1, b1), W2, b2)


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def safe_div(num, den):
    return num / den if den else float("nan")


def confusion_counts(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int).ravel()
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def threshold_metrics(y_true, probs, threshold):
    tp, tn, fp, fn = confusion_counts(y_true, probs, threshold)
    return {
        "acc": (tp + tn) / len(y_true),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "fn": fn,
        "fp": fp,
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_results(history, y_test, probs, path="results.png", dpi=300):
    ths = np.linspace(0.05, 0.95, 91)
    sweep = [threshold_metrics(y_test, probs, t) for t in ths]
    prec = [m["precision"] for m in sweep]
    rec = [m["recall"] for m in sweep]

    tp, tn, fp, fn = confusion_counts(y_test, probs, 0.5)
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="validation")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Training curve")
    ax[0].legend()

    ax[1].plot(ths, prec, label="precision")
    ax[1].plot(ths, rec, label="recall")
    ax[1].axvline(0.5, color="gray", ls="--", label="0.5")
    ax[1].axvline(0.3, color="green", ls=":", label="0.3")
    ax[1].set(xlabel="threshold", title="Precision / Recall vs threshold")
    ax[1].legend()

    ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[2].text(
                j, i, cm[i, j], ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax[2].set_xticks([0, 1], ["pred healthy", "pred diabetic"])
    ax[2].set_yticks([0, 1], ["true healthy", "true diabetic"])
    ax[2].set_title("Confusion matrix (threshold 0.5)")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)

    # Also save each panel as its own image (handy for posts and slides).
    names = ["training_curve", "precision_recall", "confusion_matrix"]
    renderer = fig.canvas.get_renderer()
    for axis, name in zip(ax, names):
        box = axis.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"{name}.png", dpi=dpi, bbox_inches=box.expanded(1.05, 1.08))

    plt.show()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    df = load_data()
    explore(df)
    Xn_train, Xn_test, y_train, y_test = preprocess(df)

    # 1) Keras model
    model, history = build_and_train(Xn_train, y_train)
    _, test_acc = model.evaluate(Xn_test, y_test, verbose=0)
    print(f"test accuracy = {test_acc:.3f}")
    print(f"baseline (always 'healthy') = {1 - y_test.mean():.3f}")

    W1, b1 = model.layers[0].get_weights()
    W2, b2 = model.layers[1].get_weights()
    print(W1.shape, b1.shape, W2.shape, b2.shape)

    # 2) NumPy forward pass must match Keras
    p_keras = model.predict(Xn_test, verbose=0)
    p_loop, t_loop = timed(predict_loop, Xn_test, W1, b1, W2, b2)
    p_vec, t_vec = timed(predict_vec, Xn_test, W1, b1, W2, b2)

    print("max difference vs Keras:", np.abs(p_loop - p_keras).max())
    print("allclose (loop vs Keras):", np.allclose(p_loop, p_keras, atol=1e-5))
    print("same result (loop vs vectorized):", np.allclose(p_loop, p_vec))
    print(f"loop time: {t_loop:.4f}s   vectorized time: {t_vec:.4f}s")

    X_big = np.tile(Xn_test, (100, 1))
    _, t_loop_big = timed(predict_loop, X_big, W1, b1, W2, b2)
    _, t_vec_big = timed(predict_vec, X_big, W1, b1, W2, b2)
    print(f"big input -> loop: {t_loop_big:.3f}s  vec: {t_vec_big:.3f}s  "
          f"ratio: {t_loop_big / t_vec_big:.1f}x")

    # 3) Metrics at the default threshold
    tp, tn, fp, fn = confusion_counts(y_test, p_vec, 0.5)
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    m = threshold_metrics(y_test, p_vec, 0.5)
    print(f"accuracy : {m['acc']:.3f}")
    print(f"precision: {m['precision']:.3f}")
    print(f"recall   : {m['recall']:.3f}")

    # 4) Threshold trade-off
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6]:
        m = threshold_metrics(y_test, p_vec, thr)
        print(f"thr={thr}  acc={m['acc']:.3f}  precision={m['precision']:.3f}  "
              f"recall={m['recall']:.3f}  FN={m['fn']}  FP={m['fp']}")

    plot_results(history, y_test, p_vec)


if __name__ == "__main__":
    main()