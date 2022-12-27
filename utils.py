from scipy.sparse import load_npz, save_npz

import numpy as np
import csv
import os
import ast
from datetime import datetime


def _load_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["user_id"].append(int(row[1]))
                data["is_correct"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def _load_question_meta_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "question_id": [],
        "subject_id": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["subject_id"].append(ast.literal_eval(row[1].strip('"')))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def _load_student_meta_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "dob": [],
        "premium_pupil": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["user_id"].append(int(row[0]))
                if row[2] != '':
                    dt = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S.%f')
                    data["dob"].append(dt.year)
                else:
                    data['dob'].append(np.nan)
                premium = row[3].replace('.', '')
                if premium.isnumeric():
                    data["premium_pupil"].append(int(premium[0]))
                else:
                    data["premium_pupil"].append(np.nan)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_train_sparse(root_dir="../data"):
    """ Load the training data as a spare matrix representation.

    :param root_dir: str
    :return: 2D sparse matrix
    """
    path = os.path.join(root_dir, "train_sparse.npz")
    if not os.path.exists(path):
        raise Exception("The specified path {} "
                        "does not exist.".format(os.path.abspath(path)))
    matrix = load_npz(path)
    return matrix


def load_question_meta_csv(root_dir="../data"):
    """ Load the training data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        question_id: a list of question id.
        subject_id: a list of list of subject ids, specific for the question_id.
    """
    path = os.path.join(root_dir, "question_meta.csv")
    return _load_question_meta_csv(path)


def load_student_meta_csv(root_dir="../data"):
    """ Load the training data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        dob: a list of the (year)date of birth, can be nan if not given.
        premium_pupil: a list of premium pupil, can be nan if not given.
    """
    path = os.path.join(root_dir, "student_meta.csv")
    return _load_student_meta_csv(path)


def load_train_csv(root_dir="../data"):
    """ Load the training data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "train_data.csv")
    return _load_csv(path)


def load_valid_csv(root_dir="../data"):
    """ Load the validation data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "valid_data.csv")
    return _load_csv(path)


def load_public_test_csv(root_dir="../data"):
    """ Load the test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "test_data.csv")
    return _load_csv(path)


def load_private_test_csv(root_dir="../data"):
    """ Load the private test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: an empty list.
    """
    path = os.path.join(root_dir, "private_test_data.csv")
    return _load_csv(path)


def save_private_test_csv(data, file_name="private_test_result.csv"):
    """ Save the private test data as a csv file.

    This should be your submission file to Kaggle.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    :param file_name: str
    :return: None
    """
    if not isinstance(data, dict):
        raise Exception("Data must be a dictionary.")
    cur_id = 1
    valid_id = ["0", "1"]
    with open(file_name, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "is_correct"])
        for i in range(len(data["user_id"])):
            if str(int(data["is_correct"][i])) not in valid_id:
                raise Exception("Your data['is_correct'] is not in a valid format.")
            writer.writerow([str(cur_id), str(int(data["is_correct"][i]))])
            cur_id += 1
    return


def evaluate(data, predictions, threshold=0.5):
    """ Return the accuracy of the predictions given the data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param predictions: list
    :param threshold: float
    :return: float
    """
    if len(data["is_correct"]) != len(predictions):
        raise Exception("Mismatch of dimensions between data and prediction.")
    if isinstance(predictions, list):
        predictions = np.array(predictions).astype(np.float64)
    return (np.sum((predictions >= threshold) == data["is_correct"])
            / float(len(data["is_correct"])))


def sparse_matrix_evaluate(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def sparse_matrix_predictions(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the predictions.

    This function can be used for submitting Kaggle competition.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: list
    """
    predictions = []
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold:
            predictions.append(1.)
        else:
            predictions.append(0.)
    return predictions


def get_question_subject_dictionary(root_dir="../data"):
    """
    A helper function to get a dictionary where the keys are the question_ids and the values are a list of subject_ids.
    :param root_dir: str, default to be ../data
    :return: output_dictionary: dictionary[int:list]
    """
    question_dictionary = load_question_meta_csv(root_dir=root_dir)
    output_dictionary = {}
    for index in range(len(question_dictionary['question_id'])):
        output_dictionary[question_dictionary['question_id'][index]] = question_dictionary['subject_id'][index]
    return output_dictionary


def get_user_dob_dictionary(root_dir="../data"):
    """
    A helper function to get a dictionary where the keys are the user_ids and the values are dobs. (the dob index, not dob value!)
    :param root_dir: str, default to be ../data
    :return: output_dictionary: dictionary[int:list]
    """
    student_dictionary = load_student_meta_csv(root_dir=root_dir)
    output_dictionary = {}
    smallest_year = min(student_dictionary['dob'])
    for index in range(len(student_dictionary['user_id'])):
        output_dictionary[student_dictionary['user_id'][index]] = student_dictionary['dob'][index] - smallest_year
    return output_dictionary


def get_user_premium_dictionary(root_dir="../data"):
    """
    A helper function to get a dictionary where the keys are the user_ids and the values are premiums.
    :param root_dir: str, default to be ../data
    :return: output_dictionary: dictionary[int:list]
    """
    student_dictionary = load_student_meta_csv(root_dir=root_dir)
    output_dictionary = {}
    for index in range(len(student_dictionary['user_id'])):
        output_dictionary[student_dictionary['user_id'][index]] = student_dictionary['premium_pupil'][index]
    return output_dictionary


def _calculate_subject_data(root_dir="../data"):
    """
    Helper function to get the subject-question matrix
    :param root_dir: str, default to be ../data
    :return: working_matrix: shape: (388(subject count), 1774(question count)). The values are the average is_correct values.
    """
    training_dictionary = load_train_csv()
    # Create a working matrix. After the calculations, the unrecorded values will be np.nan.
    working_matrix = np.zeros((388, 1774))
    # Create a counting matrix, where the initial values are zeros.
    counting_matrix = np.zeros((388, 1774))
    # Create a new dictionary where the keys are the question_ids and the values are a list of subject_ids.
    question_subject_dictionary = get_question_subject_dictionary(root_dir)
    # Calculate the working matrix values
    for index in range(len(training_dictionary['question_id'])):
        is_correct = training_dictionary['is_correct'][index]
        question_index = training_dictionary['question_id'][index]
        for subject_index in question_subject_dictionary[question_index]:
            # increment the amount of correct answers, also increment the total amount of answers by 1.
            working_matrix[subject_index][question_index] += is_correct
            counting_matrix[subject_index][question_index] += 1
    # Calculate the average values.
    # We keep the nan values, just like the training dataset.
    counting_matrix[counting_matrix == 0] = np.nan
    working_matrix /= counting_matrix
    return working_matrix


def _calculate_dob_data(root_dir="../data"):
    """
    Helper function to ge the dob-question matrix
    :param root_dir: str, default to be ../data
    :return: shape: (largest_year - smallest_year(year difference count), 1774(question count)). The values are the average is_correct values.
    """
    student_dictionary = load_student_meta_csv(root_dir=root_dir)
    training_dictionary = load_train_csv()
    smallest_year, largest_year = min(student_dictionary['dob']), max(student_dictionary['dob'])
    # Create a working matrix. After the calculations, the unrecorded values will be np.nan.
    working_matrix = np.zeros((largest_year - smallest_year + 1, 1774))
    # Create a counting matrix, where the initial values are zeros.
    counting_matrix = np.zeros((largest_year - smallest_year + 1, 1774))
    # Create a new dictionary where the keys are the user_ids and the values are dobs.
    user_dob_dictionary = get_user_dob_dictionary(root_dir)
    # Calculate the working matrix values
    for index in range(len(training_dictionary['user_id'])):
        user_id = training_dictionary['user_id'][index]
        # If the dob is not nan, record the correctness.
        if not np.isnan(user_dob_dictionary[user_id]):
            is_correct = training_dictionary['is_correct'][index]
            question_index = training_dictionary['question_id'][index]
            dob_index = user_dob_dictionary[user_id]
            # increment the amount of correct answers, also increment the total amount of answers by 1.
            working_matrix[dob_index][question_index] += is_correct
            counting_matrix[dob_index][question_index] += 1
    # Calculate the average values.
    # We keep the nan values, just like the training dataset.
    counting_matrix[counting_matrix == 0] = np.nan
    working_matrix /= counting_matrix
    return working_matrix


def _calculate_premium_data(root_dir="../data"):
    """
        Helper function to ge the premium-question matrix
        :param root_dir: str, default to be ../data
        :return: shape: (2, 1774(question count)). The values are the average is_correct values.
        """
    training_dictionary = load_train_csv()
    # Create a working matrix. After the calculations, the unrecorded values will be np.nan.
    working_matrix = np.zeros((2, 1774))
    # Create a counting matrix, where the initial values are zeros.
    counting_matrix = np.zeros((2, 1774))
    # Create a new dictionary where the keys are the user_ids and the values are premium_pupils.
    user_premium_dictionary = get_user_premium_dictionary(root_dir)
    # Calculate the working matrix values
    for index in range(len(training_dictionary['user_id'])):
        user_id = training_dictionary['user_id'][index]
        # If the premium_pupil is not nan, record the correctness.
        if not np.isnan(user_premium_dictionary[user_id]):
            is_correct = training_dictionary['is_correct'][index]
            question_index = training_dictionary['question_id'][index]
            premium_index = user_premium_dictionary[user_id]
            # increment the amount of correct answers, also increment the total amount of answers by 1.
            working_matrix[premium_index][question_index] += is_correct
            counting_matrix[premium_index][question_index] += 1
    # Calculate the average values.
    # We keep the nan values, just like the training dataset.
    counting_matrix[counting_matrix == 0] = np.nan
    working_matrix /= counting_matrix
    return working_matrix


def load_subject_meta_sparse(root_dir="../data"):
    """ Load the training data as a spare matrix representation.

    :param root_dir: str, default to be ../data
    :return: 2D sparse matrix
    """
    matrix = _calculate_subject_data(root_dir)
    return matrix


def load_dob_meta_sparse(root_dir="../data"):
    """ Load the training data as a spare matrix representation.

    :param root_dir: str, default to be ../data
    :return: 2D sparse matrix
    """
    matrix = _calculate_dob_data(root_dir)
    return matrix


def load_premium_meta_sparse(root_dir="../data"):
    """ Load the training data as a spare matrix representation.

    :param root_dir: str, default to be ../data
    :return: 2D sparse matrix
    """
    matrix = _calculate_premium_data(root_dir)
    return matrix
