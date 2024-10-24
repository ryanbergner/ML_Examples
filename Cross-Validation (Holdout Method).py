# Import all necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Iris dataset and prepare the data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Initial data splitting (60% training, 40% test)
X_train_initial, X_test, y_train_initial, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Further split the test set to create an evaluation and a final holdout set (50% from the test set each)
X_eval, X_holdout, y_eval, y_holdout = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_initial_scaled = scaler.fit_transform(X_train_initial)
X_eval_scaled = scaler.transform(X_eval)
X_holdout_scaled = scaler.transform(X_holdout)

# Evaluate various C values on the evaluation set (no need of different models here as it's just a loop)
best_c_value = None
best_c_accuracy = 0

c_values = [0.1, 1, 10, 100, 1000]
for c in c_values:
    model_eval = SVC(C=c)
    model_eval.fit(X_train_initial_scaled, y_train_initial)
    y_pred_eval = model_eval.predict(X_eval_scaled)
    accuracy = accuracy_score(y_eval, y_pred_eval)
    print(f"Accuracy for C={c}: {accuracy}")

    # Select the best performing C value
    if accuracy > best_c_accuracy:
        best_c_value = c
        best_c_accuracy = accuracy

print(f"Best performing C value: {best_c_value} with accuracy {best_c_accuracy}")

# Combine the initial training data and the evaluation set for the final model training
X_train_full = np.vstack((X_train_initial_scaled, X_eval_scaled))
y_train_full = np.hstack((y_train_initial, y_eval))

# Train the final model with the best C value on the combined training data
final_model_svc = SVC(C=best_c_value)
final_model_svc.fit(X_train_full, y_train_full)

# Perform the final evaluation on the holdout set
y_pred_holdout_final = final_model_svc.predict(X_holdout_scaled)
final_accuracy = accuracy_score(y_holdout, y_pred_holdout_final)
print(f"Final evaluation on holdout data accuracy with C={best_c_value}: {final_accuracy}")
```

This code block does the following:

1. Imports the necessary libraries and loads the dataset.
2. Prepares the data by splitting it into initial training, evaluation, and holdout sets.
3. Scales the feature data using `StandardScaler`.
4. Loops through different C values to find the best one based on performance on the evaluation set.
5. Combines the initial training set and the evaluation set to create a full training set (`X_train_full`, `y_train_full`).
6. Trains a new model, `final_model_svc`, using the full training data with the best C value.
7. Makes predictions on the holdout set and calculates the final accuracy, which is expected to be a realistic estimate of the model's generalization performance on new data.