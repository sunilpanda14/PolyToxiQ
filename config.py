import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
# Updated to point to Tox_monomer_LOC_FP.csv
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "Tox_monomer_LOC_FP.csv")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
AUTOGLUON_MODEL_PATH = os.path.join(MODEL_DIR, "ag-20240802_121037")

# Fallback paths (for local development)
# Ensure this absolute path is correct for your system
LOCAL_TRAIN_DATA_PATH = "/home/sunil/am2/Toxpred/Poly_Toxin_AM_Final/PolyToxiQ/data/Tox_monomer_LOC_FP.csv"
LOCAL_AUTOGLUON_MODEL_PATH = "/home/sunil/am2/Toxpred/Poly_Toxin_AM_Final/AutogluonModels/ag-20240802_121037"

def get_data_path():
    """Returns the appropriate data path based on availability"""
    if os.path.exists(TRAIN_DATA_PATH):
        return TRAIN_DATA_PATH
    return LOCAL_TRAIN_DATA_PATH

def get_model_path():
    """Returns the appropriate model path based on availability"""
    if os.path.exists(AUTOGLUON_MODEL_PATH):
        return AUTOGLUON_MODEL_PATH
    return LOCAL_AUTOGLUON_MODEL_PATH