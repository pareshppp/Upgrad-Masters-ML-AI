import pandas as pd
import numpy as np
from sklearn import preprocessing
from constants import *


# apply same pre-processing and feature engineering techniques as applied during the training process
def encode_features(df, features):
    '''
    Method for one-hot encoding all selected categorical fields
    '''
    # Implement these steps to prevent dimension mismatch during inference
    encoded_df = pd.DataFrame(columns= ONE_HOT_ENCODED_FEATURES) # from constants.py
    placeholder_df = pd.DataFrame()
    
    # One-Hot Encoding using get_dummies for the specified categorical features
    for f in features:
        if(f in df.columns):
            encoded = pd.get_dummies(df[f])
            encoded = encoded.add_prefix(f + '_')
            placeholder_df = pd.concat([placeholder_df, encoded], axis=1)
        else:
            print('Feature not found')
            return df
    
    # Implement these steps to prevent dimension mismatch during inference
    for feature in encoded_df.columns:
        if feature in df.columns:
            encoded_df[feature] = df[feature]
        if feature in placeholder_df.columns:
            encoded_df[feature] = placeholder_df[feature]
    # fill all null values
    encoded_df.fillna(0, inplace=True)
    
    return encoded_df

def normalize_data(df):
    val = df.values 
    min_max_normalizer = preprocessing.MinMaxScaler()
    norm_val = min_max_normalizer.fit_transform(val)
    df2 = pd.DataFrame(norm_val)
    
    return df2

def apply_pre_processing(data):
    features_to_encode = FEATURES_TO_ENCODE # from constants.py
    encoded = encode_features(data, features_to_encode)
    processed_data = normalize_data(encoded)
    return processed_data