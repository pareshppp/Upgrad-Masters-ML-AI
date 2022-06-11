MODEL_NAME = 'aditya_model1_adaboost.joblib'
ORIGINAL_FEATURES = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholestoral',
       'fasting_blood_sugar', 'restecg', 'max_hr', 'exang', 'oldpeak', 'slope',
       'num_major_vessels', 'thal']
FEATURES_TO_ENCODE = ['thal', 'slope', 'chest_pain_type', 'restecg']
ONE_HOT_ENCODED_FEATURES = ['age', 'sex', 'resting_bp', 'cholestoral', 'fasting_blood_sugar',
       'max_hr', 'exang', 'oldpeak', 'num_major_vessels', 'thal_0', 'thal_1',
       'thal_2', 'thal_3', 'slope_0', 'slope_1', 'slope_2',
       'chest_pain_type_0', 'chest_pain_type_1', 'chest_pain_type_2',
       'chest_pain_type_3', 'restecg_0', 'restecg_1', 'restecg_2']