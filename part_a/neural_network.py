from utils import *
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor. """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize the AutoEncoder. """
        super(AutoEncoder, self).__init__()
        self.g = nn.Linear(num_question, k)  # Encoder
        self.h = nn.Linear(k, num_question)   # Decoder

    def get_weight_norm(self):
        """ Return the weight norm. """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Forward pass. """
        out = torch.sigmoid(self.g(inputs))  # Encode
        out = torch.sigmoid(self.h(out))      # Decode
        return out

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network. """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(num_epoch):
        train_loss = 0.
        
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            # Add L2 regularization
            loss += lamb * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(f"Epoch: {epoch} \tTraining Cost: {train_loss:.6f}\t Valid Acc: {valid_acc:.4f}")

def evaluate(model, train_data, valid_data):
    """ Evaluate the model on the validation data. """
    model.eval()
    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    k_values = [10, 50, 100, 200, 500]
    lambdas = [0.001, 0.01, 0.1, 1]
    best_acc = 0
    best_k = None
    best_lambda = None

    for k in k_values:
        model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
        lr = 0.01  # Example learning rate
        num_epoch = 100  # Example number of epochs

        for lamb in lambdas:
            print(f"Training model with k={k} and λ={lamb}")
            train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

            # Evaluate the model
            valid_acc = evaluate(model, zero_train_matrix, valid_data)
            print(f"Validation Accuracy: {valid_acc:.4f}")

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_k = k
                best_lambda = lamb

    print(f"Best Validation Accuracy: {best_acc:.4f} with k={best_k} and λ={best_lambda}")

    # Evaluate on the test set with the best model
    final_model = AutoEncoder(num_question=train_matrix.shape[1], k=best_k)
    train(final_model, lr, best_lambda, train_matrix, zero_train_matrix, valid_data, num_epoch)
    test_acc = evaluate(final_model, zero_train_matrix, test_data)
    print(f"Test Accuracy with Best Model: {test_acc:.4f}")

if __name__ == "__main__":
    main()