from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

import operator as op


def load_data(base_path="../data"):
    train_matrix = load_train_sparse(base_path).toarray()

    premium = load_premium_meta_sparse(base_path)
    subject = load_subject_meta_sparse(base_path)
    dob = load_dob_meta_sparse(base_path)

    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    zero_premium = premium.copy()
    zero_premium[np.isnan(premium)] = 0
    zero_premium = torch.FloatTensor(zero_premium)
    premium = torch.FloatTensor(premium)

    zero_subject = subject.copy()
    zero_subject[np.isnan(subject)] = 0
    zero_subject = torch.FloatTensor(zero_subject)
    subject = torch.FloatTensor(subject)

    zero_dob = dob.copy()
    zero_dob[np.isnan(dob)] = 0
    zero_dob = torch.FloatTensor(zero_dob)
    dob = torch.FloatTensor(dob)

    subj_dict = get_question_subject_dictionary()
    dob_dict = get_user_dob_dictionary()
    prem_dict = get_user_premium_dictionary()

    return zero_train_matrix, train_matrix, valid_data, test_data, premium, zero_premium, subject, zero_subject,\
        dob, zero_dob, subj_dict, dob_dict, prem_dict


class AutoEncoder(nn.Module):
    def __init__(self, num, k=100):
        super(AutoEncoder, self).__init__()
        # Define linear functions.
        self.g = nn.Linear(num, k)
        self.h = nn.Linear(k, num)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        en = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(en))
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_rows = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for row in range(num_rows):
            inputs = Variable(zero_train_data[row]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[row].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]
            reg = (lamb / 2) * model.get_weight_norm()
            loss = torch.sum((output - target) ** 2.) + reg
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        # valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t ".format(epoch, train_loss))


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
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


def get_output(model, idx, matrix):
    """
    :param model: Module
    :param matrix: FloatTensor
    """
    model.eval()
    inputs = Variable(matrix[int(idx)]).unsqueeze(0)
    output = model(inputs)
    return output


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def update_weights(w_1, w_2, w_3, w_4, y_1, y_2, y_3, y_4, lr):
    """Update weights with gradient descent rule."""
    sum = w_1*y_1 + w_2*y_2+w_3*y_3+w_4*y_4

    w_1 = w_1 - 0.25*lr*(sigmoid(y_1) - sum)*y_1
    w_2 = w_2 - 0.25*lr*(sigmoid(y_2) - sum)*y_2
    w_3 = w_3 - 0.25*lr*(sigmoid(y_3) - sum)*y_3
    w_4 = w_4 - 0.25*lr*(sigmoid(y_4) - sum)*y_4
    return w_1, w_2, w_3, w_4


def logistic_regression(c_1, c_2, c_3, c_4, lr, valid_data, iterations, threshold):
    """Precond: c_1, c_2, c_3, c_4 are matrices where the rows represent student IDs and the columns represent question IDs"""

    prediction = np.full((len(c_1), len(c_1[0])), 0)

    # store optimal weights
    W_1 = 1
    W_2 = 1
    W_3 = 1
    W_4 = 1
    sum_1 = np.sum(c_1)
    sum_2 = np.sum(c_2)
    sum_3 = np.sum(c_3)
    sum_4 = np.sum(c_4)
    # for invalid inputs, set to 0 and do not change the weight from 0
    if (np.isnan(sum_1)):
        sum_1 = 0
        W_1 = 0
    if (np.isnan(sum_2)):
        sum_2 = 0
        W_2 = 0
    if (np.isnan(sum_3)):
        sum_3 = 0
        W_3 = 0
    if (np.isnan(sum_4)):
        sum_4 = 0
        W_4 = 0

    # skip gradient descent if only one input is value and weigh it as 1
    if (op.countOf([W_1, W_2, W_3, W_4], 0) < 3):
        #perform matrix-wise gradient descent
        for i in range(iterations):
            W_1, W_2, W_3, W_4 = update_weights(W_1, W_2, W_3, W_4, sum_1, sum_2, sum_3, sum_4, lr)
    
    # make prediction
    for n in range(len(c_1)): # number of student IDs
        for m in range(len(c_1[0])): # number of questions

            c1 = c_1[n][m]
            c2 = c_2[n][m]
            c3 = c_3[n][m]
            c4 = c_4[n][m]
            if (np.isnan(c1)):
               c1=0
            if (np.isnan(c2)):
                c2 = 0
            if (np.isnan(c3)):
                c3 = 0
            if (np.isnan(c4)):
                c4 = 0

            p = sigmoid(W_1*c_1[n][m] + W_2*c_2[n][m]+W_3*c_3[n][m]+W_4*c_4[n][m])

            if p > threshold:
                p = 1
            else:
                p = 0
            prediction[n][m] = p
            # evaluate the accuracy
            #if prediction[m][n] == valid_data
    
    accuracy = accuracy_matrix_dict(prediction, valid_data, threshold)

    print("Overall accuracy: " + str(accuracy))
    return prediction


def accuracy_matrices(predict, actual):
    """Evaluate the accuracy when comparing two matrices."""
    return np.mean(predict == actual)


def accuracy_matrix_dict(predict_mat, actual_dict, threshold):
    """Evaluate the accuracy when comparing a matrix and a dict."""
    accuracy = 0
    sample_count = len(actual_dict["user_id"])
    # evaluate the accuracy
    for i in range(len(actual_dict["user_id"])):
        user_id = actual_dict["user_id"][i]
        question_id = actual_dict["question_id"][i]
        correctness = actual_dict["is_correct"][i]
        if (correctness == 1 and predict_mat[user_id][question_id] > threshold):
            accuracy +=1
        elif (correctness == 0 and predict_mat[user_id][question_id] <= threshold):
            accuracy+=1

    if accuracy == 0:
        accuracy = 0.0001

    accuracy = accuracy / sample_count
    return accuracy


def main():
    zero_train_matrix, train_matrix, valid_data, test_data, premium, zero_premium, subject, zero_subject,\
        dob, zero_dob, subj_dict, dob_dict, prem_dict = load_data()
    k = 50

    # models
    main_model = AutoEncoder(train_matrix.shape[1], k)
    model_prem = AutoEncoder(premium.shape[1], k)
    model_dob = AutoEncoder(dob.shape[1], k)
    model_subject = AutoEncoder(subject.shape[1], k)

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 45
    lamb = 0.001

    # train
    print("Training main model...")
    train(main_model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    print("Training premium data...")
    train(model_prem, lr, lamb, premium, zero_premium, valid_data, num_epoch)
    print("Training DOB data...")
    train(model_dob, lr, lamb, dob, zero_dob, valid_data, num_epoch)
    print("Training Subject data...")
    train(model_subject, lr, lamb, subject, zero_subject, valid_data, num_epoch)
    print("Done")


    main_matrix = []
    for user_id in range(542):
        main_matrix.append((get_output(main_model, user_id, train_matrix)))
    main_matrix = torch.stack(main_matrix)

    matrix_from_dob = []
    for user_id in range(542):
        dob_index = dob_dict[user_id]
        if np.isnan(dob_index):
            matrix_from_dob.append(torch.FloatTensor([np.nan] * 1774).unsqueeze(0))
        else:
            matrix_from_dob.append(get_output(model_dob, dob_index, dob))
    matrix_from_dob = torch.stack(matrix_from_dob)

    matrix_from_premium = []
    for user_id in range(542):
        prem_index = prem_dict[user_id]
        if np.isnan(prem_index):
            matrix_from_premium.append(torch.FloatTensor([np.nan] * 1774).unsqueeze(0))
        else:
            matrix_from_premium.append(get_output(model_prem, prem_index, premium))
    matrix_from_premium = torch.stack(matrix_from_premium)

    matrix_from_subject = []
    for user_id in range(542):
        matrix_from_subject.append(get_output(model_subject, 0, subject))
    matrix_from_subject = torch.stack(matrix_from_subject)

    #print('matrix sizes')
    #print(main_matrix.shape)
    #print(matrix_from_subject.shape)
    #print(matrix_from_premium.shape)
    #print(matrix_from_dob.shape)

    # remove axes of length one
    mm_np = main_matrix.detach().numpy()
    mm_np = np.squeeze(mm_np)
    ms_np = matrix_from_subject.detach().numpy()
    ms_np = np.squeeze(ms_np)
    mp_np = matrix_from_premium.detach().numpy()
    mp_np = np.squeeze(mp_np)
    md_np = matrix_from_dob.detach().numpy()
    md_np = np.squeeze(md_np)

    # evaluate accuracy on these matrices
    threshold=0.8
    acc_mm = accuracy_matrix_dict(mm_np, valid_data, threshold)
    print("Main matrix accuracy: " + str(acc_mm))
    acc_ms = accuracy_matrix_dict(ms_np, valid_data, threshold)
    print("Subject matrix accuracy: " + str(acc_ms))
    acc_mp = accuracy_matrix_dict(mp_np, valid_data, threshold)
    print("Premium matrix accuracy: " + str(acc_mp))
    acc_md = accuracy_matrix_dict(md_np, valid_data, threshold)
    print("DOB matrix accuracy: " + str(acc_md))

    p = logistic_regression(mm_np, ms_np, mp_np, md_np, 0.01, valid_data, 10, threshold)


if __name__ == "__main__":
    main()
