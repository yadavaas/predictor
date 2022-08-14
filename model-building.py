import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle

import warnings
warnings.filterwarnings('ignore')

# loading the dataset
data = pd.read_csv('updated_data.csv')
df = data[['AvailableBankcardCredit', 'BankcardUtilization', 'BorrowerAPR',
          'BorrowerRate', 'DebtToIncomeRatio', 'DelinquenciesLast7Years',
          'EmploymentStatus', 'EmploymentStatusDuration',
          'EstimatedEffectiveYield', 'EstimatedLoss', 'EstimatedReturn',
          'IncomeRange', 'Investors', 'LenderYield', 'LoanOriginalAmount',
          'LoanOriginationQuarter', 'LoanStatus', 'Occupation',
          'OpenRevolvingMonthlyPayment', 'ProsperRating (numeric)',
          'ProsperScore', 'RevolvingCreditBalance', 'StatedMonthlyIncome', 'Term',
          'TotalCreditLinespast7years', 'TotalTrades']]

# Lebel Encoding the variables
LE = LabelEncoder()
df['IncomeRange'] = LE.fit_transform(df['IncomeRange'])

# one hot encoding
# Listing the columns with object datatype
col = df.dtypes[df.dtypes == 'object'].index

df_num = pd.get_dummies(data=df, columns=col, drop_first=True)
df = df_num

# Dependent variable
y = df['LoanStatus']
# Independent variable
X = df.drop(['LoanStatus'], axis=1)

X_scaled = preprocessing.StandardScaler().fit(X).transform(X)
# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

LR = LogisticRegression(solver='liblinear')

LR = LR.fit(X_pca, y)

pickle.dump(LR, open('LR_pickle.pkl', 'wb'))
