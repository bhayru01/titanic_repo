import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config.core import config
from src.models.pipeline import titanic_pipe
from src.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model."""

    # read train/test data
    df_train = load_dataset(filename=config.app_config.train_data_file)
    df_test = load_dataset(filename=config.app_config.test_data_file)
    y_test = load_dataset(filename=config.app_config.y_test_data_file)

    # divide train and validate
    X_train, X_val, y_train, y_val = train_test_split(
        df_train[config.model_config.features],  # predictors
        df_train[config.model_config.target],  # target
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,  # random seed for reproducibility
    )

    # Fit model
    titanic_pipe.fit(X_train, y_train)

    # Predictions
    prediction_val = titanic_pipe.predict(X_val)
    prediction_test = titanic_pipe.predict(df_test[config.model_config.features])

    # Print test/val results
    print("Validation report:")
    print(classification_report(y_val, prediction_val))
    print()
    print()
    print("Test report:")
    print(classification_report(y_test["Survived"], prediction_test))

    # Save model
    save_pipeline(titanic_pipe)


if __name__ == "__main__":
    run_training()
