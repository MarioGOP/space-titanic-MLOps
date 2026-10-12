import typer
import pandas as pd

from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from space_titanic.config import INTERIM_DATA_DIR, RAW_DATA_DIR, SPLIT_RATIO, RANDOM_STATE, COL_SPENT

app = typer.Typer()

def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the dataset from the specified input path."""
    return pd.read_csv(input_path)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning on the dataset."""

    ## Delete irrelevant columns
    df = df.drop(columns=["PassengerId", "Name"], axis=1)

    ## Divide the "Cabin" column into two columns: "Deck" , "Num" and "Side"
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(['Cabin'], axis=1)

    ## Convert the "Num" column to numeric type
    df['Num'] = pd.to_numeric(df['Num'], errors='coerce')

    ## Fill missing CryoSleep values (False) if spent is > 0.
    ## First calculate the total spent by each passenger
    df['total_spent'] = df[COL_SPENT].sum(axis=1)
    df.loc[(df['CryoSleep'].isnull()) & (df['total_spent'] > 0), 'CryoSleep'] = False

    # Drop the total_spent column as it is no longer needed
    df = df.drop(columns=['total_spent'], axis=1)

    ## Fill missing spent values with 0 if CryoSleep is True
    for col in COL_SPENT:
        df.loc[(df['CryoSleep'] == True) & (df[col].isnull()), col] = 0

    ## On the EDA we found that duplicate rows are for different passengers, so we will keep them as they are.
    ## We are going to fill the nulls with an imputer on the pipeline, so we will not drop them here.
    return df

def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test sets."""
    train_df, test_df = train_test_split(df, test_size=SPLIT_RATIO, random_state=RANDOM_STATE, stratify=df['Transported'])

    return train_df, test_df

def save_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned dataset to the specified output path."""
    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = INTERIM_DATA_DIR,
):
    # Load the dataset
    logger.info(f"Loading dataset from: {input_path}")
    df = load_dataset(input_path)

    # Clean the dataset
    logger.info("Cleaning dataset...")
    df = clean_dataset(df)
    logger.info("Dataset cleaned successfully.")

    # Split the dataset
    logger.info("Splitting dataset...")
    train_df, test_df = split_dataset(df)
    
    # Save the datasets
    save_dataset(train_df, test_df, output_path)
    logger.success(f"Cleaned and split dataset saved to: {output_path}")

if __name__ == "__main__":
    app()
