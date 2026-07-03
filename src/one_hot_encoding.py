import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder

from config import (
    DF_TRAIN_ORG,
    DF_TEST_ORG,
    DF_TRAIN_ENCODED,
    DF_TEST_ENCODED,
    ENCODER_MODEL
)


def apply_one_hot_encoder(
    train_path=DF_TRAIN_ORG,
    test_path=DF_TEST_ORG,
    columns=None,
    save_train_path=DF_TRAIN_ENCODED,
    save_test_path=DF_TEST_ENCODED,
    save_model_path=ENCODER_MODEL
):
    """
    Fit OneHotEncoder on the training data and transform both
    training and test datasets.

    Parameters
    ----------
    train_path : str
        Path to training CSV.

    test_path : str
        Path to test CSV.

    columns : list
        Categorical columns to encode.

    save_train_path : str
        Output path for encoded training data.

    save_test_path : str
        Output path for encoded test data.

    save_model_path : str
        Output path for encoder (.pkl).

    Returns
    -------
    train_df
    test_df
    encoder
    """

    if columns is None:
        raise ValueError("Please provide columns to encode.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_encoded = encoder.fit_transform(train_df[columns])
    test_encoded = encoder.transform(test_df[columns])

    encoded_columns = encoder.get_feature_names_out(columns)

    train_encoded = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=train_df.index
    )

    test_encoded = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=test_df.index
    )

    train_df.drop(columns=columns, inplace=True)
    test_df.drop(columns=columns, inplace=True)

    train_df = pd.concat([train_df, train_encoded], axis=1)
    test_df = pd.concat([test_df, test_encoded], axis=1)

    train_df.to_csv(save_train_path, index=False)
    test_df.to_csv(save_test_path, index=False)

    joblib.dump(encoder, save_model_path)

    return train_df, test_df, encoder


if __name__ == "__main__":

    apply_one_hot_encoder(columns=['region'])