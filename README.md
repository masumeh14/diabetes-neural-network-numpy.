# Diabetes Prediction with a 2-Layer Neural Network (NumPy Forward Propagation)

A small neural network for predicting diabetes on the Pima Indians Diabetes dataset. The model is trained with Keras, and then **forward propagation is re-implemented from scratch in NumPy** (using the weights learned by Keras) to show what a dense layer actually computes. The NumPy output is verified against `model.predict`.

## Results

Test set: 154 samples (100 healthy, 54 diabetic), stratified 80/20 split, `random_state=42`.

| Model | Accuracy |
|---|---|
| Baseline (always predict "healthy") | 0.649 |
| Neural network (threshold 0.5) | 0.740 |

Confusion matrix at threshold 0.5:

|  | Predicted healthy | Predicted diabetic |
|---|---|---|
| **Actually healthy** | TN = 81 | FP = 19 |
| **Actually diabetic** | FN = 21 | TP = 33 |

Precision = 0.635, recall = 0.611.

### Effect of the decision threshold

| Threshold | Accuracy | Precision | Recall | FN | FP |
|---|---|---|---|---|---|
| 0.2 | 0.701 | 0.547 | 0.870 | 7 | 39 |
| 0.3 | 0.747 | 0.606 | 0.796 | 11 | 28 |
| 0.4 | 0.740 | 0.609 | 0.722 | 15 | 25 |
| 0.5 | 0.740 | 0.635 | 0.611 | 21 | 19 |
| 0.6 | 0.747 | 0.692 | 0.500 | 27 | 12 |

Accuracy stays almost constant between 0.3 and 0.6, while the number of missed diabetic patients (FN) changes from 27 to 11. This is why accuracy alone is not enough for a medical screening task. Lowering the threshold trades extra false alarms for higher recall.

## Method

1. **Data loading:** 768 samples, 8 features, binary label `Outcome`.
2. **Cleaning:** zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are physiologically impossible and are treated as missing values (`NaN`). Zeros in `Pregnancies` are valid and kept.
3. **Split:** 80/20 train/test, stratified by label.
4. **Imputation:** missing values are filled with the **median of the training set** (also used for the test set).
5. **Normalization:** z-score using the mean and standard deviation of the **training set only**, to avoid data leakage.
6. **Model (Keras):** `Dense(8, sigmoid)` then `Dense(1, sigmoid)`, binary cross-entropy, Adam (lr = 0.01), 100 epochs, batch size 32, 20% validation split.
7. **NumPy re-implementation:** `my_dense`, `my_sequential`, `my_predict` using the trained weights.
8. **Verification:** maximum absolute difference to Keras is about 1.2e-7, `np.allclose` is `True`, and the accuracy is identical (0.7403).
9. **Vectorized version:** `sigmoid(A_in @ W + b)` replaces the per-neuron loop and gives the same output. It was about 13x faster on the test set in my environment.

## Core idea

Each neuron is one **column** of `W`:

```python
def my_dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out
```

Shapes: `x (8,) -> W1 (8, 8) -> a1 (8,) -> W2 (8, 1) -> a2 (1,)`.

## Limitations

- The test set is small (54 positive cases), so one sample changes recall by about 1.9 percentage points. Small differences between thresholds are not reliable.
- The decision threshold was explored on the test set. For a rigorous evaluation, choose the threshold on a validation set (or with cross-validation) and evaluate on the test set once.
- Median imputation is simple. `Insulin` and `SkinThickness` have many missing values (about 49% and 30%), so more careful imputation could help.
- No hyperparameter tuning was done.

## How to run

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
jupyter notebook diabetes_numpy_nn.ipynb
```

Dataset: Pima Indians Diabetes (UCI / Kaggle), loaded from
`https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`

## Possible next steps

- ReLU in the hidden layer and more neurons, compared with cross-validation
- Threshold selection on a validation set
- ROC and precision-recall curves
