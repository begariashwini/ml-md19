#preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
def create_pipeline():
#define which feature are numeric vs categorical
numeric_feature = ['age','income']
categorical_feature = ['city','occupation']
#preprocessing for numerical data(impute missing,then scale)
numeric_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
#preprocessing for categorical data(impute missing,then one-hot-encode)
categorical_transformer=Pipeline(steps=[('impute',SimpleImputer(strategy='constant',fill_value='missing')),
('onehot',OneHotEncoder(handle_unknown='igmore'))
])
#combine into a single ColumnTransformer
preprocessor=ColumnTransformer(transformer=[('num',numeric_transformer,numeric_feature),('cat',
categorical_transformer,
ctaegorical_feature)
])
return preprocessor
if_name_=="_main_":
print("Preprocessing pipeline template ready".)
