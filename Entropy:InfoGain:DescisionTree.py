import random
import numpy as np
import pandas as pd
random.seed(50)
liability_data = ["Normal", "High"]

random_data = []

for i in range(100):
  liability = random.choice(liability_data)

  if liability == "Normal":
    credit_rating = random.randint(600, 850)
  else:
    credit_rating = random.randint(300,700)

  random_data.append((credit_rating, liability))


prob1_df = pd.DataFrame(random_data, columns = ['Credit Rating', 'Liability'])
prob1_df.head()

# Calculate Entropy Components
prob_dist = random.random() # random probability
H = - np.sum(prob_dist * np.log2(prob_dist))
print("The function gives: " + str(H))

### Calculate parent Entropy
prob_dist_compliment = 1 - prob_dist
Hcomp = - np.sum(prob_dist_compliment * np.log2(prob_dist_compliment))
print(H, Hcomp)



 """ Calculate the entropy of an RV (data column). """
def entropy(data_column):

    probabilities = data_column.value_counts(normalize = True)

    return -np.sum(probabilities * np.log2(probabilities))
    

'''Info Gain Function'''


def info_gain(df,info_col,target_col,threshold):
  data_above_thresh = df[df[info_col] <= threshold]
  data_below_thresh = df[df[info_col] > threshold]

  H = entropy(df[target_col])

  entropy_above = entropy(data_above_thresh[target_col])
  entropy_below = entropy(data_below_thresh[target_col])

  values_above = data_above_thresh.shape[0]
  values_below = data_below_thresh.shape[0]

  values_total = float(df.shape[0])

  weighted_entropy = ((values_above / values_total) * entropy_above) + ((values_below / values_total) * entropy_below)

  return round(H - weighted_entropy, 4)
    
    
#Find parent entropy for species
parent_entropy = round(entropy(pen_df['Class']), 4)
parent_entropy

# Find Best Threshold

def best_threshold(df, info_col, target_col, criteria = info_gain):
  ig_max = 0
  threshhold_max = 0

  for thresh in df[info_col]:
    ig = criteria(df,info_col,target_col,thresh)

    if ig > ig_max:
      ig_max = ig
      threshhold_max =  thresh
      max_column = info_col

  return info_col, ig_max, threshhold_max



#So
best_threshold(pen_df, 'Amount', 'Class', criteria = info_gain)


#Best Split
def best_split(df, info_columns, target_column, criteria=info_gain):
    max_info_gain = -np.inf  # Start with the lowest possible value for information gain
    best_feature = None
    best_threshold = None

    # Iterate over each feature column to find the best threshold for information gain
    for column in info_columns:
        # Initialize variables for tracking the best threshold and its info gain for the current column
        current_best_threshold = None
        current_max_info_gain = -np.inf

        # Get unique values of the column and sort them
        unique_values = sorted(df[column].unique())

        # Iterate over each unique value to find the best threshold
        for i in range(len(unique_values) - 1):
            # Compute the threshold as the midpoint between consecutive unique values
            threshold = (unique_values[i] + unique_values[i + 1]) / 2

            # Calculate information gain for the current threshold
            ig = criteria(df, column, target_column, threshold)

            # Update the best threshold and info gain for this column if the current ig is better
            if ig > current_max_info_gain:
                current_max_info_gain = ig
                current_best_threshold = threshold

        # Update the overall best feature, threshold, and info gain if this column's best is better
        if current_max_info_gain > max_info_gain:
            max_info_gain = current_max_info_gain
            best_feature = column
            best_threshold = current_best_threshold

    return best_feature, max_info_gain, best_threshold
    
#So
best_split(pen_df, ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm','body_mass_g'], 'species', criteria = info_gain)



### Descision Tree

from sklearn.tree import DecisionTreeClassifier, plot_tree

X = pen_df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = pen_df['species']

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# Print feature importances
print(clf.feature_importances_)

# Df of feature importances
importances_df = pd.DataFrame({

    'Feature': X.columns,
    'Importance': clf.feature_importances_ }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print(importances_df)

# Plot 
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))  # Adjust the figure size as needed
plot_tree(clf, filled=True, feature_names=X.columns, class_names=np.unique(y).astype(str), rounded=True, proportion=False, precision=2)
plt.show()

