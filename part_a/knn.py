from sklearn.impute import KNNImputer
import numpy as np
import matplotlib.pyplot as plt
from utils import load_train_sparse, load_valid_csv, load_public_test_csv, sparse_matrix_evaluate

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data. """
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (User-based): {}".format(acc))
    return acc

def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data. """
    nbrs = KNNImputer(n_neighbors=k, metric='cosine')
    mat = nbrs.fit_transform(matrix.T).T  # Transpose for item-based
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy (Item-based): {}".format(acc))
    return acc

def main():
    sparse_matrix = load_train_sparse("../data/train_sparse.npz").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    k_values = [1, 6, 11, 16, 21, 26]
    user_accuracies = []
    item_accuracies = []

    # Evaluate user-based collaborative filtering
    for k in k_values:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_accuracies.append(acc)

    # Evaluate item-based collaborative filtering
    for k in k_values:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_accuracies.append(acc)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, user_accuracies, marker='o', label='User-based')
    plt.plot(k_values, item_accuracies, marker='o', label='Item-based')
    plt.title('Validation Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy')
    plt.xticks(k_values)
    plt.legend()
    plt.grid()
    plt.show()

    # Report the best k for both methods
    best_user_k = k_values[np.argmax(user_accuracies)]
    best_item_k = k_values[np.argmax(item_accuracies)]
    print(f"Best k for User-based: {best_user_k} with accuracy: {max(user_accuracies)}")
    print(f"Best k for Item-based: {best_item_k} with accuracy: {max(item_accuracies)}")

    # Evaluate the test accuracy using the best k values
    print("Evaluating test accuracy...")
    knn_impute_by_user(sparse_matrix, test_data, best_user_k)
    knn_impute_by_item(sparse_matrix, test_data, best_item_k)

if __name__ == "__main__":
    main()