import pandas as pd
import joblib

from sklearn.preprocessing import PowerTransformer
from config import DF_TRAIN_ENCODED,DF_TEST_ENCODED, DF_TEST_TRANS, DF_TRAIN_TRANS, TRANS_MODEL

def apply_power_transformer(
    train_df,
    test_df,
    columns,
    save_train_path=DF_TRAIN_TRANS,
    save_test_path=DF_TEST_TRANS,
    save_model_path=TRANS_MODEL,
    method="yeo-johnson"
):
    """
    Fit PowerTransformer on training data and transform both train and test.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    columns : list
        Columns to transform.
    save_train_path : str, optional
    save_test_path : str, optional
    save_model_path : str, optional
    method : str
        'yeo-johnson' or 'box-cox'

    Returns
    -------
    train_df
    test_df
    transformer
    """

    transformer = PowerTransformer(method=method,standardize=False)

    train_df[columns] = transformer.fit_transform(train_df[columns])
    test_df[columns] = transformer.transform(test_df[columns])

    if save_train_path:
        train_df.to_csv(save_train_path, index=False)

    if save_test_path:
        test_df.to_csv(save_test_path, index=False)

    if save_model_path:
        joblib.dump(transformer, save_model_path)

    return train_df, test_df, transformer

df_train=pd.read_csv(DF_TRAIN_ENCODED)
df_test=pd.read_csv(DF_TEST_ENCODED)

apply_power_transformer(df_train,df_test,['sqft_living','sqft_living15','sqft_liv_per_bedroom'])