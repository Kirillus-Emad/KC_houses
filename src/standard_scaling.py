import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from config import (
    DF_TRAIN_TRANS_SCALE,
    DF_TEST_TRANS_SCALE,
    DF_TRAIN_SCALE,
    DF_TEST_SCALE,
    DF_TRAIN_TRANS,
    DF_TEST_TRANS,
    SCALER_MODEL,
    SCALER_MODEL_TRANS,
    DF_TRAIN_ENCODED,
    DF_TEST_ENCODED
)


def apply_standard_scaler(
    train_path=DF_TRAIN_TRANS,
    test_path=DF_TEST_TRANS,
    columns=None,
    save_train_path=DF_TRAIN_TRANS_SCALE,
    save_test_path=DF_TEST_TRANS_SCALE,
    save_model_path=SCALER_MODEL_TRANS
):
    """
    Fit StandardScaler on training data and transform both
    training and test datasets.

    Parameters
    ----------
    train_path : str
        Path to transformed training CSV.

    test_path : str
        Path to transformed test CSV.

    columns : list
        Numeric columns to scale.

    save_train_path : str
        Output path for scaled training data.

    save_test_path : str
        Output path for scaled test data.

    save_model_path : str
        Output path for scaler (.pkl).

    Returns
    -------
    train_df
    test_df
    scaler
    """

    if columns is None:
        raise ValueError("Please provide columns to scale.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    scaler = StandardScaler()

    train_df[columns] = scaler.fit_transform(train_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])

    train_df.to_csv(save_train_path, index=False)
    test_df.to_csv(save_test_path, index=False)

    joblib.dump(scaler, save_model_path)

    return train_df, test_df, scaler


if __name__ == "__main__":

    scale_columns = ['sqft_living',	'sqft_living15', 
                     'sqft_liv_per_bedroom', 'bathrooms', 'grade']

    apply_standard_scaler(train_path=DF_TRAIN_ENCODED,test_path=DF_TEST_ENCODED,columns=scale_columns,
                          save_train_path=DF_TRAIN_SCALE,save_test_path=DF_TEST_SCALE,save_model_path=SCALER_MODEL)
    
    apply_standard_scaler(train_path=DF_TRAIN_TRANS,test_path=DF_TEST_TRANS,columns=scale_columns,
                        save_train_path=DF_TRAIN_TRANS_SCALE,save_test_path=DF_TEST_TRANS_SCALE,save_model_path=SCALER_MODEL_TRANS)