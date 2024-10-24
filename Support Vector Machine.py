import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


filepath = '/content/drive/MyDrive/Advanced Business Analytics/data/wine_fraud.csv'
wine = pd.read_csv(filepath)
wine.shape
wine.head()


# Check Target Distribution & Bar Graph
round(wine['quality'].value_counts()/len(wine['quality']),3)
plt.figure(figsize=(6, 4))
sns.countplot(x = 'quality', data = wine)
plt.title('Distribution of Quality')
plt.show()


# Does there appear to be a difference in quaity From Red & White wine?
round(pd.crosstab(wine['quality'], wine['type'], normalize = True, margins = True), 2)
plt.figure(figsize=(6, 4))
sns.countplot(x = 'quality', hue = 'type', data = wine)
plt.title('Distribution of Quality by Type')
plt.show()


# Create Heatmap of correlation Matrix
# Calculate the correlation matrix
corr = wine.corr()
# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))  # Adjust the figure size as necessary
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
plt.show()


# Prepare Target Data And One Hot Encode Type
wine['Fraud'] = wine['quality'].map({'Legit': 0 , 'Fraud': 1})
wine.head()
# Create dummy variables for the 'type' column
type_dummies = pd.get_dummies(wine['type'], prefix='type')
# Concatenate the dummy variables with the original DataFrame
wine = pd.concat([wine, type_dummies], axis=1)
wine.head()
#drop the old
wine.drop(['quality', 'type'], axis=1, inplace=True)


# What Variables have the most Correlation?
# Calculate the correlation matrix
corr_matrix = wine.corr()
# Get the 'Fraud' column, sort in descending order, and round to two decimals
feature_correlation_with_fraud = corr_matrix['Fraud'].sort_values(ascending=False).round(2)
# Drop the 'Fraud' self-correlation
feature_correlation_with_fraud = feature_correlation_with_fraud.drop(labels='Fraud')
# Display the correlations
feature_correlation_with_fraud


#Plot Correlations
sns.barplot(data = feature_correlation_with_fraud)
plt.title('Featur Corr W Fraud')
plt.xticks()
plt.ylabel('Characteristic')
plt.show()


#Scatterplot W 2 Highest Features
feature1 = 'volatile acidity'
feature2 = 'free sulfur dioxide'
# Now create the scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=wine[feature1], y=wine[feature2], hue=wine['Fraud'])
plt.title('Scatterplot of Features with Two Highest Correlations')
plt.show()



# Now Fit/Run SVM

X = wine.drop('Fraud', axis=1)
y = wine['Fraud']
X.head()

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

svm_model = SVC(kernel='linear', C=1, class_weight='balanced')
svm_model.fit(scaled_X, y)




# Check # of Support Vectors
# Create a DataFrame to store the support vectors and their target values
support_vectors = svm_model.support_vectors_
support_vector_indices = svm_model.support_
support_vector_targets = y.iloc[support_vector_indices]
df_support_vectors = pd.DataFrame(support_vectors, columns=X.columns)
df_support_vectors['Target'] = support_vector_targets.values
df_support_vectors.head()
num_support_vectors = df_support_vectors.shape[0]
num_support_vectors


# Create a DataFrame to store the coefficients of the SVM model
coefficients = svm_model.coef_.flatten()
df_coefficients = pd.DataFrame(coefficients, index=X.columns, columns=['Coefficient']).round(2)
df_coefficients


# 2 Most Important Features
sorted_coefficients = df_coefficients.abs().sort_values('Coefficient', ascending=False)
top_two_features = sorted_coefficients.head(2)
top_two_features

