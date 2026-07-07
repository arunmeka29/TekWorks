# =====================================================
# TRAIN AND SAVE LOGISTIC REGRESSION MODEL
# =====================================================

import pandas as pd
import pickle

# Required: pip install scikit-learn
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.preprocessing import LabelEncoder  # type: ignore[import-not-found]


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# =====================================================
# DATA PREPROCESSING
# =====================================================

df['TotalCharges'] = pd.to_numeric(
    
    df['TotalCharges'],
    
    errors='coerce'
)

df.dropna(inplace=True)


# =====================================================
# FEATURES
# =====================================================

X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]


# =====================================================
# TARGET COLUMN
# =====================================================

le = LabelEncoder()

y = le.fit_transform(df['Churn'])


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    
    X,
    y,
    
    test_size=0.2,
    
    random_state=42
)


# =====================================================
# TRAIN MODEL
# =====================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# =====================================================
# SAVE MODEL
# =====================================================

pickle.dump(
    
    model,
    
    open('churn_model.pkl', 'wb')
)

print("Model saved successfully!")