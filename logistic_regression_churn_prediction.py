import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'


#%% data preparation

df = pd.read_csv('data-week-3.csv')
print(df.head())

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

tc = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df.churn = (df.churn == 'yes').astype(int)

#%% validation framework

from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)



df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

#%% EDA

df_full_train = df_full_train.reset_index(drop=True)
print(df_full_train.isnull().sum())
print(df_full_train.churn.mean())

numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

print(df_full_train[categorical].nunique())

#%% Risk ratio

print(df_full_train.head())
churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
print(churn_female)
churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
print(churn_male)
global_churn = df_full_train.churn.mean()
print(global_churn)
churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
print(churn_partner)
churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
print(churn_no_partner)

for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)
    print()
    print()

#%% Mutual information

from sklearn.metrics import mutual_info_score

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)
mi = df_full_train[categorical].apply(mutual_info_churn_score)
print(mi.sort_values(ascending=False))

#%% numerical variables correlation

print(df_full_train[numerical].corrwith(df_full_train.churn).abs())
print(df_full_train[df_full_train.tenure <= 2].churn.mean())
print(df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean())
print(df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges <= 50)].churn.mean())

#%% One-hot encoding

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


#%% Logistic regression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def linear_regression(xi):
    result = w0    
    for j in range(len(w)):
        result = result + xi[j] * w[j]        
    return result
  
def logistic_regression(xi):
    score = w0    
    for j in range(len(w)):
        score = score + xi[j] * w[j]        
    result = sigmoid(score)
    return result

#%% Training logistic regression

from sklearn.linear_model import LogisticRegression


model = LogisticRegression(solver='lbfgs')
# solver='lbfgs' is the default solver in newer version of sklearn
# for older versions, you need to specify it explicitly
print(model.fit(X_train, y_train))

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
print((y_val == churn_decision).mean())


df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
df_pred['correct'] = df_pred.prediction == df_pred.actual

#%% Using the model 

dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
y_full_train = df_full_train.churn.values
model = LogisticRegression(solver='lbfgs')

dicts_test = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(dicts_test)
y_pred = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred >= 0.5)
print((churn_decision == y_test).mean())

customer = dicts_test[-1]
X_small = dv.transform([customer])
print(model.predict_proba(X_small)[0, 1])
