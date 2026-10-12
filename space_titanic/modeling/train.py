import typer
import pandas as pd
import mlflow
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from loguru import logger
from tqdm import tqdm
from enum import StrEnum

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score


from space_titanic.config import MLFLOW_TRACKING_URI, MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, RANDOM_STATE,MLFLOW_EXPERIMENT_NAME, TARGET_COL
from space_titanic.dataset import load_dataset

app = typer.Typer()

class Solver(StrEnum):
    ## Validate the solver options for Logistic Regression
    lbfgs = "lbfgs"
    liblinear = "liblinear"
    newton_cg = "newton-cg"
    newton_cholesky = "newton-cholesky"
    sag = "sag"
    saga = "saga"



def build_pipeline(c: float = 1.0, solver: Solver = Solver.lbfgs, max_iter: int = 1000, seed: int = RANDOM_STATE) -> Pipeline:
    """Build the machine learning pipeline."""

    # Define the categorical and numerical columns
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']

    # Define the transformers for categorical and numerical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine the transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Create the final pipeline with the preprocessor and a classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=max_iter, C=c, solver=solver.value, random_state=seed))
    ])

    return pipeline

def train_model(X_train: pd.DataFrame, y_train: pd.Series, pipeline: Pipeline) -> Pipeline:
    """Train the machine learning model."""
    pipeline.fit(X_train, y_train)
    return pipeline

def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target."""
    X = df.drop(columns=[TARGET_COL], axis=1)
    y = df[TARGET_COL]

    return X, y

@app.command()
def main(
    features_path_train : Path = PROCESSED_DATA_DIR / "features_train.csv",
    features_path_test : Path = PROCESSED_DATA_DIR / "features_test.csv", 
    model_path: Path = MODELS_DIR / "model.pkl",
    max_iter: int = 1000,
    c : float = 1.0,
    solver: Solver = Solver.lbfgs,
    seed: int = RANDOM_STATE
):
    # Load the features dataset
    train_df = load_dataset(features_path_train)
    test_df = load_dataset(features_path_test)
    # Log the shapes of the datasets
    logger.info(f"Training set shape: {train_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")

    # Split the features and target
    X_train, y_train = split_features_and_target(train_df)
    X_test, y_test = split_features_and_target(test_df)
    
    # Build the machine learning pipeline
    pipeline = build_pipeline(c=c, solver=solver, max_iter=max_iter, seed=seed)

    # Train the model
    logger.info("Training the model...")
    pipeline = train_model(X_train, y_train, pipeline)

    # Predict on the test set
    logger.info("Making predictions on the test set...")
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # Create and save the confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Transported", "Transported"],
        cmap="Blues",
    )
    disp.ax_.set_title("Matriz de confusión")
    
    # Save the confusion matrix figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path = FIGURES_DIR / "confusion_matrix.png"
    disp.figure_.savefig(confusion_matrix_path, bbox_inches='tight')
    plt.close(disp.figure_)

    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")

    # Save the trained model
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")
    

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "max_iter": max_iter,
            "C": c,
            "solver": solver.value,
            "random_state": seed
        })

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(confusion_matrix_path))
        mlflow.sklearn.log_model(
            pipeline, 
            name="model",
            skops_trusted_types=["numpy.dtype"],
        )

    

if __name__ == "__main__":
    app()
