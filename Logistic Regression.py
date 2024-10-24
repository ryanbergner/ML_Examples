from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

# Read in Data Then

myfilepath = "/content/drive/MyDrive/Advanced Business Analytics/data/heart.csv"
heart = pd.read_csv(myfilepath)
heart.shape
heart.head()


# Drop rows w nulls 
heart.dropna(inplace = True)
heart1 = heart.isna().any()
heart1

heart.shape


# Describe
round(heart.describe(),2)


# Target Frequency
heart["target"].value_counts()


#What features have the strongest correlation?
round(heart.corr()['target'].sort_values(), 2)



#Model

X = heart[['age','sex','cp','trestbps','chol','fbs',
        'restecg','thalach','exang','oldpeak','slope','ca','thal']]

y = heart['target']

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

log_model = LogisticRegression(max_iter=200)  # Increasing max_iter for convergence

# Fit the model with the training data
log_model.fit(scaled_X, y)





#Get df of feature name, Coefficient, and Odds Ratio
model_coef = log_model.coef_[0]
odds_ratios = np.exp(model_coef)
feature_info = list(zip(X.columns, model_coef, odds_ratios))


# Step 4: Create a DataFrame
importance_df = pd.DataFrame(feature_info, columns=['Feature Name', 'Coefficient', 'Odds Ratio'])
importance_df['Coefficient'] = round(importance_df['Coefficient'], 2)
importance_df['Odds Ratio'] = round(importance_df['Odds Ratio'], 2)
importance_df.sort_values(by='Odds Ratio', ascending=False, inplace=True)
importance_df.reset_index(drop=True, inplace=True)
importance_df


#So to calculaye the odds ratio for females:
# male is 1, female = 0
females_with_event = heart[(heart['sex'] == 0) & (heart['target'] == 1)].shape[0]
females_without_event = heart[(heart['sex'] == 0) & (heart['target'] == 0)].shape[0]
odds_females = females_with_event / females_without_event

males_with_event = heart[(heart['sex'] == 1) & (heart['target'] == 1)].shape[0]
males_without_event = heart[(heart['sex'] == 1) & (heart['target'] == 0)].shape[0]
odds_males = males_with_event / males_without_event

odds_ratio_females = odds_females / odds_males
round(odds_ratio_females, 2)