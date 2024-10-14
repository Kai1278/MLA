import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

# Assuming you have these models implemented as classes or functions
from knn import knn_impute_by_user, knn_impute_by_item  # Replace with your actual imports
from irt_model import irt_predict, IRTModel  # Replace with your actual IRT prediction function and add IRTModel import
from matrix_factorization import matrix_factorization_predict, MatrixFactorizationModel  # Add MatrixFactorizationModel import
from utils import load_train_csv, load_valid_csv, load_public_test_csv
from knn_model import KNNModel  # Add this import for KNNModel

def bootstrap_samples(train_data, n_samples):
    """ Generate bootstrapped datasets. """
    bootstrapped_datasets = []
    for _ in range(n_samples):
        bootstrapped_data = resample(train_data, replace=True)
        bootstrapped_datasets.append(bootstrapped_data)
    return bootstrapped_datasets

def ensemble_predict(models, test_data):
    """ Make predictions using an ensemble of models. """
    predictions = np.zeros((len(test_data), len(models)))

    for i, model in enumerate(models):
        # Assuming each model has a predict method
        predictions[:, i] = model.predict(test_data)

    # Average predictions for regression or majority vote for classification
    final_predictions = np.mean(predictions, axis=1) >= 0.5  # Change to majority vote if needed
    return final_predictions.astype(int)

def evaluate_ensemble(models, train_data, valid_data, test_data):
    """ Evaluate the ensemble method using bagging. """
    n_samples = 3  # Number of bootstrapped datasets
    bootstrapped_datasets = bootstrap_samples(train_data, n_samples)

    # Train models on bootstrapped datasets
    trained_models = []
    for model in models:
        for bootstrapped_data in bootstrapped_datasets:
            # Train model on the bootstrapped dataset
            model.fit(bootstrapped_data)  # Define fit method for your models
            trained_models.append(model)

    # Generate predictions
    valid_predictions = ensemble_predict(trained_models, valid_data)
    test_predictions = ensemble_predict(trained_models, test_data)

    # Calculate accuracy
    valid_accuracy = accuracy_score(valid_data["is_correct"], valid_predictions)
    test_accuracy = accuracy_score(test_data["is_correct"], test_predictions)

    print(f"Validation Accuracy of Ensemble: {valid_accuracy:.4f}")
    print(f"Test Accuracy of Ensemble: {test_accuracy:.4f}")

def main():
    # Load the data
    train_data = load_train_csv("../data")
    valid_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Define your base models
    knn_model_user = KNNModel(method='user')  # Replace with your actual model initialization
    knn_model_item = KNNModel(method='item')  # Replace with your actual model initialization
    irt_model = IRTModel()  # Replace with your actual IRT model initialization
    mf_model = MatrixFactorizationModel()  # Replace with your actual MF model initialization

    models = [knn_model_user, knn_model_item, irt_model, mf_model]

    # Evaluate the ensemble
    evaluate_ensemble(models, train_data, valid_data, test_data)

if __name__ == "__main__":
    main()
