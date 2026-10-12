from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np

from space_titanic.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, COL_SPENT
from space_titanic.dataset import load_dataset

app = typer.Typer()

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from the dataset."""
    df = df.copy()
    ## We calculate a log1p for all the spent columns, to reduce the skewness of the data.
    for col in COL_SPENT:
        # substitute the original column with the log1p transformation of the column
        df[col] = np.log1p(df[col])

    return df

def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame, output_path_train: Path, output_path_test: Path) -> None:
    """Save the features to the specified output path."""
    train_df.to_csv(output_path_train, index=False)
    test_df.to_csv(output_path_test, index=False)

@app.command()
def main(
    input_path_train: Path = INTERIM_DATA_DIR / "train.csv",
    input_path_test: Path = INTERIM_DATA_DIR / "test.csv",
    output_path_train: Path = PROCESSED_DATA_DIR / "features_train.csv",
    output_path_test: Path = PROCESSED_DATA_DIR / "features_test.csv",
):
    """Generate features from the dataset and save them to the specified output path."""
    logger.info(f"Loading dataset from {input_path_train} and {input_path_test}")
    train_df = load_dataset(input_path_train)
    test_df = load_dataset(input_path_test)

    logger.info("Generating features")
    train_df = generate_features(train_df)
    test_df = generate_features(test_df)

    logger.info(f"Saving features to {output_path_train} and {output_path_test}")
    save_features(train_df, test_df, output_path_train, output_path_test)


if __name__ == "__main__":
    app()
