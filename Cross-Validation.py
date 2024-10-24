import numpy as np                     # For numerical computations
import pandas as pd                    # For handling dataframes
from sklearn import datasets           # To load datasets
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt        # For plotting and visualization
import seaborn as sns                 # For enhanced data visualization
%matplotlib inline

# Load the iris Dataset
iris = datasets.load_iris()  # Load the dataset

# Converting to DataFrame for easier manipulation and understanding
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                       columns= iris['feature_names'] + ['target'])

# Each iris sample is labeled with a target: 0 for 'setosa', 1 for 'versicolor', and 2 for 'virginica'
# Let's map these integers back to the actual species names for better understanding
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Let's check the first 5 entries of the dataset
iris_df.head()

# Iris dataset has already been loaded with the variable name 'iris'
# Now, let's define the feature and target data

# Define feature data (predictor variables)
X = iris.data
feature_names = iris.feature_names

# Define target data (response variable)
y = iris.target
target_names = iris.target_names

# Convert feature data into a DataFrame for easier manipulation
df_features = pd.DataFrame(X, columns=feature_names)

# Convert target data into a DataFrame
# We'll create two columns: one for numerical target, and another for the corresponding species name
df_target = pd.DataFrame(y, columns=['target'])
df_target['species'] = df_target['target'].map({i:name for i, name in enumerate(target_names)})

# Combine features and target into a single DataFrame for a holistic view
iris_df = pd.concat([df_features, df_target], axis=1)

# Display the first few entries of the combined DataFrame
iris_df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% testing

# Step 4: Fit/Train Scaler on Training X Data (normalize feature data before training models)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Compute the mean and std to be used for later scaling

# Step 5: Scale X Test Data 
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Create Model
model = RandomForestClassifier(n_estimators=100, random_state=42) # You can also try other models

# Step 7: Fit/Train Model on X Train Data
model.fit(X_train_scaled, y_train)

# Step 8: Evaluate Model on X Test Data
from sklearn.metrics import accuracy_score, classification_report

# We make predictions with the scaled data
y_pred = model.predict(X_test_scaled)

# Now, compare these predictions to the actual labels (y_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)

print("Accuracy: ", accuracy)
print("Classification Report:")
print(report)

# Step 9: Adjust Parameters as Necessary and repeat steps 5 and 6
# If you're not satisfied with the model performance, you can try tweaking some hyperparameters of your model,
# and then refit and evaluate it again using the above code (steps 6-8).
# For RandomForestClassifier, you can modify parameters such as: n_estimators, max_depth, min_samples_split, etc.

# For example, let's try to adjust the number of trees in the forest (n_estimators) and max_depth:

from sklearn.svm import SVC  # Example using Support Vector Classifier

# Suppose we want to use SVM with a specific C value
model = SVC(C=1.0)  # The default C value is 1.0
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (with SVC and C=1.0): ", accuracy)
```

In the context of reporting results, you would report both the model's accuracy and the hyperparameters used, including the `C` value if applicable.
Wait, we  were supposed to use a support vector machine model instead - can you show me how to do that?
Of course! If you're supposed to use a Support Vector Machine (SVM) model, you'll use sklearn's `SVC` (Support Vector Classifier) instead of `RandomForestClassifier`. Let's adjust the previous code to use an SVM model, including how to handle the hyperparameter `C`.

```python

# Step 1: Clean and adjust data as necessary for X and y
# No additional cleaning needed for the Iris dataset

# Step 2: Import Libraries (if not already imported at the beginning)

# Step 3: Split Data in Train/Test for both X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Fit/Train Scaler on Training X Data
scaler = StandardScaler()
scaler.fit(X_train)  # Only fit on the training data

# Step 5: Scale X Test Data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Create an SVM Model with a specific C value
# The C value is the inverse of regularization strength; a smaller value means more regularization
model2 = SVC(C=1.0, kernel='linear', random_state=42)  # You can start with default C=1.0 and linear kernel

# Step 7: Fit/Train Model on X Train Data
model2.fit(X_train_scaled, y_train)

# Step 8: Evaluate Model on X Test Data
y_pred = model2.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Accuracy: ", accuracy)
print("Classification Report:")
print(report)

# Step 9: Adjust Parameters as Necessary and repeat steps 5 and 6
# If the performance is not satisfactory, you can try adjusting the C value or changing the kernel
# Try again with a different C value
model_with_different_c = SVC(C=0.5, kernel='linear', random_state=42)
model_with_different_c.fit(X_train_scaled, y_train)
y_pred_different_c = model_with_different_c.predict(X_test_scaled)
new_accuracy = accuracy_score(y_test, y_pred_different_c)
print("New Accuracy (with adjusted C=0.5): ", new_accuracy)
```

You start with a default `C` value of 1.0 and a `linear` kernel, which are common starting points. After evaluating the model, if you want to optimize the `C` value or try a different kernel (`'rbf'`, `'poly'`, `'sigmoid'`, etc.), you might use grid search or a validation set to find the best `C`.

In the case you switch to a non-linear kernel like `'rbf'` (Radial Basis Function), the model might capture more complex patterns in the data, which can work very well for datasets that are not linearly separable. The choice really depends on the specific characteristics of your dataset.
Okay, Awesome. Continuing with this problem: Holdout Testing and Cross Validation
We left off the prior video with an example of trying to mannually adjust the C parameter until we were satisfied with the accuracy reported
However, training a model this way can lead to what is known as model "leakage" which means our model evaluation metrics no longer are reporting on generalization performance
What we want to be able to do, is confidently give evaluation metrics that will generalize to future data
In the next few examples we will show a procedure to create another split on the data to create what is known as a holdout dataset. This data will never be seen by the model in training and therefore will serve as a better evaluation of our model performance.

keyboard_arrow_down
Holdout Testing
Because adjusting models and evaluating performance should never be done on the same testing and training data, let's create a new split of the data that the model will never see in training.
For sake of clarity and a reminder of what we are working wiht, we are going to redefine our data for this process


Here, we will be using train_test_split again but only on the the test data defined in the prior cell. This will create our final holdout test set for model evaluation.
We will use the _eval data to test parameters and make other model adjustments until satisfied
Then we will use the _holdout data to perform our final model evaluations which should be more generalizable
Here, we will set test_size = .5 which will take 50% of the test data we defined in the previous cell.
So we will get 60 * .5 = 30 rows for our holdout data


Note: We will skip the scaling in this example but you could go back and try it later to see if it makes a difference
In this example we go back through building the model and testing some hyperparameters
What is different here is that now once we are satisfied with our parameters, we will do a final evaluation on the holdout data.
We will test the following values for C:
.1, 1, 10, 100, 1000
We will run these all in the same cell and the jot down the notes in the markdown cell below this code cell
[ ]
0s


keyboard_arrow_down
Evaluation Summary
Here was the accuracy of our model based on the various values of C:
C value	Accuracy
.1	90%
1	93%
10	87%
100	93%
1000	93%
Even though we had some ties for "best" parameter, let's use C=1.
Note: Once we pick the parameters based on the _eval data, we cannot go back and adjust them after seeing results on the holdout set.
If you do this, you will be creating the same problem we had prior to using the holdout data and creating model/data leakage
Final evaluation on holdout data.
[ ]
0s

Should we trust these numbers?
Wow, we got 100% accuracy on the holdout testing data with a C=1. However, anytime a model produces 100% accuracy we should be very suspicious. Here's some reasons why we might want to pump the breaks on this evaluation:
We are dealing with a small dataset. The iris dataset only has 150 total records
When we split this data up twice it left us with some really small training, evaluation, and holdout datasets
It is likely that our randomly selected holdout data just happened to be data our model could predict perfectly
Because we have this potential issue that the one, randomly selected, holdout dataset could lead to misleading assumptions of our model's accuracy, Cross Validation is often used to calculate many different training and testing/holdout datasets.
We will go over Cross Validation in the next section.
To implement a holdout testing approach followed by cross-validation to ensure we have a reliable evaluation of model generalization, you can follow the code below. Let's start with defining the holdout subset:

```
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# First, create the training and test sets as usual
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% training and 40% testing

# Now, create a holdout set based on the test data
X_eval, X_holdout, y_eval, y_holdout = train_test_split(X_test, y_test, test_size=0.5, random_state=42)  # Half of the 40% test data

# Back to Steps 5-8, we are going to scale the data but here’s the evaluation section
c_values = [0.1, 1, 10, 100, 1000]
best_c_value = None
best_c_accuracy = 0

# Scaling training and evaluation sets (skipping holdout for now)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

# Fit the model and evaluate different C values
for c in c_values:
    model = SVC(C=c)
    model.fit(X_train_scaled, y_train)
    y_pred_eval = model.predict(X_eval_scaled)
    accuracy = accuracy_score(y_eval, y_pred_eval)
    print(f"Accuracy for C={c}: {accuracy}")

    # Select the best performing C value
    if accuracy > best_c_accuracy:
        best_c_value = c
        best_c_accuracy = accuracy

print(f"Best performing C value: {best_c_value} with accuracy {best_c_accuracy}")

# Final evaluation on holdout data (first scale it)
X_holdout_scaled = scaler.transform(X_holdout)
final_model = SVC(C=best_c_value)
final_model.fit(X_train_scaled, y_train)
y_pred_holdout = final_model.predict(X_holdout_scaled)
holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
print(f"Final evaluation on holdout data accuracy with C={best_c_value}: {holdout_accuracy}")
```

You must resist the temptation to keep tweaking your model after this point—it's important to avoid 'leakage' from the holdout set into your model training process.

Cross-validation is a method to evaluate the performance of your model by creating several different partitions of the data into training and validation sets and averaging the performance across these. In the next step, we will implement cross-validation.

```python
from sklearn.model_selection import cross_val_score

# Implementing cross-validation (let's use 5 folds here)
cv_scores = cross_val_score(SVC(C=best_c_value), X_train_scaled, y_train, cv=5)

# The scores for each fold
print("Cross-validation accuracy scores:", cv_scores)

# The mean cross-validation accuracy
print("Mean cross-validation accuracy:", cv_scores.mean())

# This gives us a better estimate of how our model will perform on unseen data.
```

We used the best `C` value as chosen from the previous step for cross-validation. It averages the model's performance over multiple subsets rather than relying on a single holdout test, making our estimate of the model's generalization more reliable.

Note that in a real-world scenario, you would usually use cross-validation to choose the `C` parameter rather than relying on the holdout set alone. The holdout approach adds an additional layer of validation and should be used judiciously, especially for small datasets where the partitioning of data can lead to high variance in the estimations of model performance.
Ok, now let's try one more value for C

```python
# Define the new C value to test
new_c_value = 0.5

# Fit the model on the training data using the new C value
model_with_new_c = SVC(C=new_c_value)
model_with_new_c.fit(X_train_scaled, y_train)

# Evaluate on the evaluation set
y_pred_eval_with_new_c = model_with_new_c.predict(X_eval_scaled)
accuracy_with_new_c = accuracy_score(y_eval, y_pred_eval_with_new_c)
print(f"Accuracy for C={new_c_value} on evaluation data: {accuracy_with_new_c}")

# Compare with the previous best C value
if accuracy_with_new_c > best_c_accuracy:
    print(f"New C value {new_c_value} is better than the previous best C value {best_c_value}.")
    best_c_value = new_c_value
    best_c_accuracy = accuracy_with_new_c
    # Final evaluation should be done only with the best C value found
    final_model_with_new_c = SVC(C=best_c_value)
    final_model_with_new_c.fit(X_train_scaled, y_train)
    y_pred_holdout_with_new_c = final_model_with_new_c.predict(X_holdout_scaled)
    holdout_accuracy_with_new_c = accuracy_score(y_holdout, y_pred_holdout_with_new_c)
    print(f"Final evaluation on holdout data accuracy with new C={best_c_value}: {holdout_accuracy_with_new_c}")
else:
    print(f"New C value {new_c_value} does not improve the best accuracy so far {best_c_accuracy} with C value {best_c_value}. Using previous best C value for final evaluation.")
    # Final model evaluation on holdout data with the best C value found previously
    final_model = SVC(C=best_c_value)
    final_model.fit(X_train_scaled, y_train)
    y_pred_holdout = final_model.predict(X_holdout_scaled)
    holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
    print(f"Final evaluation on holdout data accuracy with C={best_c_value}: {holdout_accuracy}")
```
In this code, we have reevaluated our model with the new `C` value and compared its performance to the previously identified best `C` value. If the new `C` value performs better, then we've updated our model with it and performed the final holdout evaluation. Otherwise, we've stuck with the original best `C` value.

An important note here is that model tuning should be done based on validation results, and the holdout set should only be used for the final assessment to get a reliable estimate of the generalization performance.
Finally, since C=1 is still producting the best results, we evaluate on the test data.
Now, we need to fit our model prior to the final evaluation
That's correct. If you have determined that C=1 is indeed the best choice based on your previous assessments with the evaluation set, it's time to conduct the final evaluation on the holdout test data.

To do this, you'll need to fit the Support Vector Machine model one last time using all the available training data (which includes the original training and evaluation sets) with C=1, and then make predictions on the holdout set. Here is how you can do it:
```# Best C value determined previously
best_c_value = 1

# Combine original training data and the evaluation set for the final model training
X_train_eval = np.vstack((X_train_scaled, X_eval_scaled))
y_train_eval = np.hstack((y_train, y_eval))

# Fit the final model with the combined training data
final_model = SVC(C=best_c_value)
final_model.fit(X_train_eval, y_train_eval)

# Perform the final evaluation using the holdout set
y_pred_holdout_final = final_model.predict(X_holdout_scaled)
final_accuracy = accuracy_score(y_holdout, y_pred_holdout_final)

print(f"Final evaluation on holdout data accuracy with C={best_c_value}: {final_accuracy}")

# Now you can report this `final_accuracy` as the performance metric of your model.
```

In this code:
- We stack the training and evaluation datasets using `vstack` for features and `hstack` for labels to train the model on the full set of data that was not part of the holdout set.
- We train the final model on this combined set using the chosen C value.
- We make predictions on the holdout set to get our final assessment of the model's generalization performance.

This final accuracy gives us a realistic estimate of how well the model will perform on entirely new, unseen data. Remember, this is the final test—you should not make any further adjustments to the model after this point to avoid data leakage.

Ensure you're using the proper scaling applied to the full training data and using the same scaler to transform the holdout data before making predictions.
