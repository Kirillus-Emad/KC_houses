DATA_PATH=r'../data/KC_Data.csv'
DF_TRAIN_ORG=r'../data/df_train_org.csv'
DF_TEST_ORG=r'../data/df_test_org.csv'

# Power Transformer model
DF_TRAIN_TRANS=r'../data/df_train_trans.csv'
DF_TEST_TRANS=r'../data/df_test_trans.csv'
TRANS_MODEL=r'Transformer/yeo_johnson_model.pkl'

# Standard Scaling

DF_TRAIN_TRANS_SCALE = r"../data/df_train_trans_scale.csv"
DF_TEST_TRANS_SCALE = r"../data/df_test_trans_scale.csv"
DF_TRAIN_SCALE = r"../data/df_train_scale.csv"
DF_TEST_SCALE = r"../data/df_test_scale.csv"

SCALER_MODEL_TRANS = r"scaling_data/standard_scaler_trans.pkl"
SCALER_MODEL = r"scaling_data/standard_scaler.pkl"

DF_TRAIN_ENCODED = r"../data/df_train_encoded.csv"
DF_TEST_ENCODED = r"../data/df_test_encoded.csv"

ENCODER_MODEL = r"OneHot_encoding/onehot_encoder.pkl"

# ML models
TARGET_COLUMN = "price"
RESULTS_DIR = r"results"

# ANN scaling

DF_TRAIN_ANN_SCALE=r"../data/df_train_ANN_scale.csv"
DF_TEST_ANN_SCALE=r"../data/df_test_ANN_scale.csv"
ANN_SCALER_MODEL= r"scaling_data/minmax_scaler.pkl"