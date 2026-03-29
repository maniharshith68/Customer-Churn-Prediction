import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# cost function
def costFunction(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    error = (y * np.log(h) + (1 - y) * np.log(1 - h))
    cost = -1 / m * np.sum(error)
    grad = 1 / m * X.T.dot(h - y)
    return cost, grad

# gradient descent
def gradientDescent(X, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        cost, grad = costFunction(X, y, theta)
        theta -= alpha * grad
        cost_history[i] = cost
    return theta, cost_history

# predict
def predict(X, theta):
    return sigmoid(X.dot(theta))

# accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('../data/processed_customer_churn.csv')

# split features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].values  # convert to numpy

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# ADD INTERCEPT
# =========================
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# =========================
# TRAIN MODEL
# =========================
theta = np.zeros(X_train.shape[1], dtype=float)

theta, cost_history = gradientDescent(X_train, y_train, theta,alpha=0.01,iterations=1000)

# =========================
# PLOT COST
# =========================
plt.plot(range(1000), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History')
plt.show()

# =========================
# TEST PREDICTIONS
# =========================
y_pred = predict(X_test, theta)
y_pred = np.array([1 if i > 0.5 else 0 for i in y_pred])

print("Accuracy:", accuracy(y_test, y_pred) * 100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# ROC CURVE
# =========================
# probabilities instead of 0/1
y_prob = predict(X_test, theta)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("AUC Score:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# =========================
# K-FOLD CROSS-VALIDATION
# =========================
def cross_validate(X, y, k=5, alpha=0.01, iterations=1000):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    auc_scores = []

    for train_index, test_index in kf.split(X):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

        # scale inside each fold
        scaler = StandardScaler()
        X_train_cv = scaler.fit_transform(X_train_cv)
        X_test_cv = scaler.transform(X_test_cv)

        # add intercept
        X_train_cv = np.c_[np.ones((X_train_cv.shape[0], 1)), X_train_cv]
        X_test_cv = np.c_[np.ones((X_test_cv.shape[0], 1)), X_test_cv]

        # train
        theta = np.zeros(X_train_cv.shape[1])
        theta, _ = gradientDescent(X_train_cv, y_train_cv, theta, alpha, iterations)

        # predict
        y_prob_cv = predict(X_test_cv, theta)
        y_pred_cv = np.array([1 if i > 0.5 else 0 for i in y_prob_cv])

        # accuracy
        acc = np.mean(y_test_cv == y_pred_cv)
        accuracies.append(acc)

        # auc
        fpr, tpr, _ = roc_curve(y_test_cv, y_prob_cv)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)

    print("Cross-validation Accuracy:", np.mean(accuracies))
    print("Cross-validation AUC:", np.mean(auc_scores))


# =========================
# Calling cross-validation with original (unscaled, no intercept) X and y
# =========================

# use original (unscaled, no intercept) X
X_np = X.values
y_np = y
cross_validate(X_np, y_np)

# =========================
# COLLECT METRICS
# =========================

# Accuracy on test set
test_accuracy = accuracy(y_test, y_pred)

# AUC on test set
test_auc = roc_auc

# Cross-validation metrics (recompute for storage, or modify cross_validate to return values)
def cross_validate_metrics(X, y, k=5, alpha=0.01, iterations=1000):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    auc_scores = []

    for train_index, test_index in kf.split(X):
        X_train_cv, X_test_cv = X.values[train_index], X.values[test_index]
        y_train_cv, y_test_cv = y.values[train_index], y.values[test_index]

        scaler_cv = StandardScaler()
        X_train_cv = scaler_cv.fit_transform(X_train_cv)
        X_test_cv = scaler_cv.transform(X_test_cv)

        X_train_cv = np.c_[np.ones((X_train_cv.shape[0], 1)), X_train_cv]
        X_test_cv = np.c_[np.ones((X_test_cv.shape[0], 1)), X_test_cv]

        theta_cv = np.zeros(X_train_cv.shape[1])
        theta_cv, _ = gradientDescent(X_train_cv, y_train_cv, theta_cv, alpha, iterations)

        y_prob_cv = predict(X_test_cv, theta_cv)
        y_pred_cv = np.array([1 if i > 0.5 else 0 for i in y_prob_cv])

        accuracies.append(np.mean(y_test_cv == y_pred_cv))
        fpr_cv, tpr_cv, _ = roc_curve(y_test_cv, y_prob_cv)
        auc_scores.append(auc(fpr_cv, tpr_cv))

    return np.mean(accuracies), np.mean(auc_scores)

cv_accuracy, cv_auc = cross_validate_metrics(X, pd.Series(y))

# =========================
# SAVE METRICS as pickle
# =========================
metrics = {
    "accuracy": test_accuracy,
    "auc": test_auc,
    "cv_accuracy": cv_accuracy,
    "cv_auc": cv_auc,
    "fpr": fpr,
    "tpr": tpr
}

with open("../models/churn_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("Metrics saved successfully!")

# =========================
# NEW DATA PREDICTION
# =========================
df2 = pd.read_csv('../data/new_customers_1.csv')

# same preprocessing
df2.drop(['Names', 'Location', 'Company', 'Onboard_date'], axis=1, inplace=True)

# scale + intercept
X_new = scaler.transform(df2)
X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]

# predict
y_pred_new = predict(X_new, theta)
y_pred_new = np.array([1 if i > 0.5 else 0 for i in y_pred_new])

# attach predictions
df2['Churn'] = y_pred_new

print(df2.head())


# =========================
# SAVE MODEL as pickle file
# =========================

model_artifacts = {
    "theta": theta,
    "scaler": scaler,   # make sure you renamed it earlier
    "feature_names": X.columns.tolist(),
    "X_train_sample": X.sample(100, random_state=42) # sample 100 rows for SHAP background
}

with open("../models/churn_model.pkl", "wb") as f:
    pickle.dump(model_artifacts, f)

print("Model saved successfully!")
