
### Section 0.1: Importing Initial Libraries

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
```

### Section 0.2: Load the breast_cancer Dataset

```python
# Load the dataset
data = load_breast_cancer()
```

### Section 2: Data Understanding and Data Preparation

This section is left for you to fill in based on prior examples and explorations.

```python
# Assuming you have done the EDA and data preparation already
```

### Section 3.1: Imports

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
```

### Section 3.2: Define Features and Target Variables

```python
X = data.data
y = data.target
print(y[-5:])
```

### Section 3.3: Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
```

### Section 3.4: Fit/Train Scaler

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### Section 3.5: Scale Test Data

```python
X_test_scaled = scaler.transform(X_test)
```

### Section 3.6: Create Model

```python
model_1 = SVC(C=0.1, kernel='rbf', gamma=1, class_weight="balanced")
```

### Section 3.7: Evaluate model_1

```python
cv_scores = cross_val_score(model_1, X_train_scaled, y_train, cv=10)
print(cv_scores)
```

```python
print(f"Mean: {cv_scores.mean():.2f}, Std: {cv_scores.std():.2f}")
```

### Section 3.7: Evaluate model_1 on Test Data

```python
model_1.fit(X_train_scaled, y_train)
test_accuracy = model_1.score(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
```

### Section 4.1: Define param_grid

```python
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf', 'linear'],
    'class_weight': [None, "balanced"]
}
```

### Section 4.2: Create Grid Search Model

```python
grid_model = SVC()
```

### Section 4.3: Create GridSearchCV Object

```python
grid_search = GridSearchCV(grid_model, param_grid, cv=10)
```

### Section 4.4: Fit grid_search

```python
grid_search.fit(X_train_scaled, y_train)
cv_results_df = pd.DataFrame(grid_search.cv_results_)
print(cv_results_df.columns)
```

```python
top_5_ranked_models = cv_results_df.sort_values(by="rank_test_score").head(5)
print(top_5_ranked_models)
```

### Section 4.5: Best Estimator

```python
best_estimator = grid_search.best_estimator_
```

### Section 4.6: Show Best Parameters

```python
print(grid_search.best_params_)
```

### Section 4.7: Evaluate Best Estimator

```python
best_score = best_estimator.score(X_test_scaled, y_test)
print(f"Best test score: {best_score:.2f}")