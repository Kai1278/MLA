from utils import *
from scipy.linalg import sqrtm
import numpy as np
import matplotlib.pyplot as plt

def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition to reconstruct the matrix. """
    # Fill missing values (NaN) with the item mean
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Subtract item means
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]

    # Reconstruct the matrix
    reconst_matrix = np.dot(np.dot(Q, s), Ut)
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)

def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data. """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - np.sum(u[data["user_id"][i]] * z[q])) ** 2
    return 0.5 * loss

def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying stochastic gradient descent for matrix completion. """
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # Compute prediction
    prediction = np.dot(u[n], z[q])

    # Update U and Z
    u[n] += lr * (c - prediction) * z[q]
    z[q] += lr * (c - prediction) * u[n]
    
    return u, z

def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm using SGD. """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k))

    losses = []
    for iteration in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        loss = squared_error_loss(train_data, u, z)
        losses.append(loss)
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss:.4f}")

    return u, z, losses

def main():
    train_matrix = load_train_sparse("../data/train_sparse.npz").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # SVD: Try out different k values
    k_values = [5, 10, 20, 50, 100]
    svd_accuracies = []

    for k in k_values:
        reconst_matrix = svd_reconstruct(train_matrix, k)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        svd_accuracies.append(acc)
        print(f"SVD Validation Accuracy with k={k}: {acc:.4f}")

    # Plot SVD results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, svd_accuracies, marker='o', label='SVD')
    plt.title('SVD Validation Accuracy vs. k')
    plt.xlabel('k (Number of Singular Values)')
    plt.ylabel('Validation Accuracy')
    plt.xticks(k_values)
    plt.legend()
    plt.grid()
    plt.show()

    # ALS: Try out different k values
    als_accuracies = []
    lr = 0.01
    num_iterations = 100

    for k in k_values:
        u, z, losses = als(train_data, k, lr, num_iterations)
        acc = sparse_matrix_evaluate(val_data, np.dot(u, z.T))
        als_accuracies.append(acc)
        print(f"ALS Validation Accuracy with k={k}: {acc:.4f}")

    # Plot ALS results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, als_accuracies, marker='o', label='ALS')
    plt.title('ALS Validation Accuracy vs. k')
    plt.xlabel('k (Number of Latent Factors)')
    plt.ylabel('Validation Accuracy')
    plt.xticks(k_values)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()