import numpy as np
import matplotlib.pyplot as plt
from utils import load_train_csv, load_valid_csv, load_public_test_csv

def sigmoid(x):
    """ Apply sigmoid function. """
    return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood. """
    log_likelihood = 0.0
    for i in range(len(data['user_id'])):
        u = data['user_id'][i]
        q = data['question_id'][i]
        is_correct = data['is_correct'][i]

        p = sigmoid(theta[u] - beta[q])
        log_likelihood += is_correct * np.log(p) + (1 - is_correct) * np.log(1 - p)
    
    return -log_likelihood  # Return negative log-likelihood

def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent. """
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)

    for i in range(len(data['user_id'])):
        u = data['user_id'][i]
        q = data['question_id'][i]
        is_correct = data['is_correct'][i]
        
        p = sigmoid(theta[u] - beta[q])
        grad_theta[u] += (is_correct - p)
        grad_beta[q] += (is_correct - p)

    # Update parameters
    theta += lr * grad_theta
    beta -= lr * grad_beta  # Subtract for beta

    return theta, beta

def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy. """
    pred = []
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        p = sigmoid(theta[u] - beta[q])
        pred.append(p >= 0.5)  # Threshold at 0.5
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def irt(data, val_data, lr, iterations):
    """ Train IRT model. """
    num_users = max(data['user_id']) + 1
    num_questions = max(data['question_id']) + 1
    
    # Initialize theta and beta
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    val_acc_lst = []
    
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta, beta)
        score = evaluate(val_data, theta, beta)
        val_acc_lst.append(score)
        print(f"NLLK: {neg_lld:.4f} \t Score: {score:.4f}")
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst

def plot_probability(theta, beta, questions):
    """ Plot the probability of correct response vs. student ability. """
    abilities = np.linspace(-3, 3, 100)  # Range of student abilities
    for q in questions:
        probabilities = sigmoid(abilities - beta[q])
        plt.plot(abilities, probabilities, label=f'Question {q}')
    
    plt.title('Probability of Correct Response vs Student Ability')
    plt.xlabel('Student Ability (θ)')
    plt.ylabel('Probability of Correct Response')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    train_data = load_train_csv("../data/train_data.csv")
    val_data = load_valid_csv("../data/valid_data.csv")

    lr = 0.01  # Learning rate
    iterations = 1000  # Number of iterations

    theta, beta, val_acc_lst = irt(train_data, val_data, lr, iterations)

    # Choose three distinct questions (e.g., 0, 1, 2)
    plot_probability(theta, beta, questions=[0, 1, 2])

if __name__ == "__main__":
    main()