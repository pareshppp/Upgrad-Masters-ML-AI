from pydantic import BaseModel
from typing import Optional

class InputDataModel(BaseModel):
	age: float
	sex: int
	chest_pain_type: int
	resting_bp: float
	cholestoral: float
	fasting_blood_sugar: float
	restecg: int
	max_hr: float
	exang: int
	oldpeak: float
	slope: int
	num_major_vessels: int
	thal: int
	
class OutputDataModel(BaseModel):
    predicted_value: bool
    predicted_class: str
