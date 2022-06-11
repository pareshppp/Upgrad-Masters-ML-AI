import joblib
from fastapi import FastAPI
from constants import *
from utils import *
from data_model import *


app = FastAPI()
model = joblib.load(MODEL_NAME) # Load from constants

def apply_model(model, inference_data):
	# Prepare prediction dataframe
	inf_df = pd.DataFrame([[inference_data.age, 
						   inference_data.sex, 
						   inference_data.chest_pain_type, 
						   inference_data.resting_bp,
						   inference_data.cholestoral,
						   inference_data.fasting_blood_sugar,
						   inference_data.restecg,
						   inference_data.max_hr,
						   inference_data.exang,
						   inference_data.oldpeak,
						   inference_data.slope,
						   inference_data.num_major_vessels,
						   inference_data.thal]], columns = ORIGINAL_FEATURES)
	inf_df[ORIGINAL_FEATURES[0]] = inference_data.age
	inf_df[ORIGINAL_FEATURES[1]] = inference_data.sex
	inf_df[ORIGINAL_FEATURES[2]] = inference_data.chest_pain_type
	inf_df[ORIGINAL_FEATURES[3]] = inference_data.resting_bp
	inf_df[ORIGINAL_FEATURES[4]] = inference_data.cholestoral
	inf_df[ORIGINAL_FEATURES[5]] = inference_data.fasting_blood_sugar
	inf_df[ORIGINAL_FEATURES[6]] = inference_data.restecg
	inf_df[ORIGINAL_FEATURES[7]] = inference_data.max_hr
	inf_df[ORIGINAL_FEATURES[8]] = inference_data.exang
	inf_df[ORIGINAL_FEATURES[9]] = inference_data.oldpeak
	inf_df[ORIGINAL_FEATURES[10]] = inference_data.slope
	inf_df[ORIGINAL_FEATURES[11]] = inference_data.num_major_vessels
	inf_df[ORIGINAL_FEATURES[12]] = inference_data.thal
	
	processed_inference_data = apply_pre_processing(inf_df)
	pred = model.predict(processed_inference_data)[0]
	
	if pred == 1:
		pred_value = True
		pred_class = "heart disease"
	else:
		pred_value = False
		pred_class = "No heart disease"

	return pred_value, pred_class

@app.get('/')
def get_root():

	return {'message': 'Welcome to the Heart Disease Detection API'}
	
@app.post("/predict", response_model=OutputDataModel)
async def post_predictions(inference_data: InputDataModel):

    pred_value, pred_class = apply_model(model, inference_data)

    response = {
        "predicted_value": pred_value,
        "predicted_class": pred_class
    }
    return response