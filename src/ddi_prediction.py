"""
Dự đoán tương tác thuốc–thuốc (DDI) bằng phương pháp học tổ hợp (ensemble learning) và thuật toán di truyền
"""

import os
from typing import List, Tuple

import numpy as np
import networkx as nx
import math
from numpy.linalg import inv, pinv
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
)
from sklearn.preprocessing import MinMaxScaler
import copy
from numpy import linalg as LA
import csv
import array
import random
from deap import algorithms, base, creator, tools
from sklearn import linear_model


def cross_validation(drug_drug_matrix, CV_num, seed, file_results, file_weights):
    # Khởi tạo biến lưu số lượng liên kết (links) và các vị trí của liên kết, không phải liên kết
    link_number = 0
    link_position = []
    nonLinksPosition = [] 

    # Duyệt qua ma trận drug-drug để xác định các liên kết (link) và không phải liên kết
    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[i, j] == 1:  
                link_number = link_number + 1
                link_position.append([i, j])  
            else:  
                nonLinksPosition.append([i, j]) 

    link_position = np.array(link_position)  
    random.seed(seed)  # Đặt seed cho việc tạo số ngẫu nhiên
    index = np.arange(0, link_number) 
    random.shuffle(index)  # Xáo trộn chỉ số ngẫu nhiên

    # Xác định số lượng fold cho cross-validation
    fold_num = link_number // CV_num
    print(fold_num)

    # Chạy vòng lặp cross-validation
    for CV in range(0, CV_num):
        print("*********round:" + str(CV) + "**********\n")

        # Chọn các chỉ số của các liên kết test trong fold hiện tại
        test_index = index[(CV * fold_num) : ((CV + 1) * fold_num)]
        test_index.sort() 
        testLinkPosition = link_position[test_index] 

        # Tạo bản sao của ma trận drug-drug và loại bỏ các liên kết trong ma trận train
        train_drug_drug_matrix = copy.deepcopy(drug_drug_matrix)
        for i in range(0, len(testLinkPosition)):
            train_drug_drug_matrix[testLinkPosition[i, 0], testLinkPosition[i, 1]] = 0  # Đặt giá trị của liên kết test thành 0
            train_drug_drug_matrix[testLinkPosition[i, 1], testLinkPosition[i, 0]] = 0  # Do ma trận đối xứng, xóa cả hai chiều

        # Danh sách vị trí test và non-link dùng cho việc dự đoán
        testPosition = list(testLinkPosition) + list(nonLinksPosition)

        # Xác định các tham số nội bộ (có thể là các tham số mô hình)
        weights, cf1, cf2 = internal_determine_parameter(copy.deepcopy(train_drug_drug_matrix))
        # cf1,cf2=internal_determine_parameter(copy.deepcopy(train_drug_drug_matrix))

        # Sử dụng phương pháp dự đoán kết hợp (ensemble) để dự đoán và tính toán kết quả
        [multiple_predict_matrix, multiple_predict_results] = ensemble_method(
            copy.deepcopy(drug_drug_matrix), train_drug_drug_matrix, testPosition
        )

        # Tính toán điểm số ensemble và các tham số (AUC, precision, recall...)
        ensemble_results, ensemble_results_cf1, ensemble_results_cf2 = ensemble_scoring(
            copy.deepcopy(drug_drug_matrix), multiple_predict_matrix, testPosition, weights, cf1, cf2
        )

        # Ghi kết quả dự đoán vào tệp
        for i in range(0, len(multiple_predict_results)):
            [auc_score, aupr_score, precision, recall, accuracy, f] = multiple_predict_results[i]
            file_results.write(
                auc_score + " " + aupr_score + " " + precision + " " + recall + " " + accuracy + " " + f + "\n"
            )
            file_results.flush()

        # Ghi kết quả ensemble vào tệp
        [auc_score, aupr_score, precision, recall, accuracy, f] = ensemble_results
        file_results.write(
            auc_score + " " + aupr_score + " " + precision + " " + recall + " " + accuracy + " " + f + "\n"
        )
        file_results.flush()

        [auc_score, aupr_score, precision, recall, accuracy, f] = ensemble_results_cf1
        file_results.write(
            auc_score + " " + aupr_score + " " + precision + " " + recall + " " + accuracy + " " + f + "\n"
        )
        file_results.flush()

        [auc_score, aupr_score, precision, recall, accuracy, f] = ensemble_results_cf2
        file_results.write(
            auc_score + " " + aupr_score + " " + precision + " " + recall + " " + accuracy + " " + f + "\n"
        )
        file_results.flush()

        # Ghi các trọng số vào tệp weights
        weights_str = ""
        for i in range(0, len(weights)):
            weights_str = weights_str + " " + str(weights[i])
        file_weights.write(weights_str + "\n")
        file_results.flush()
        file_weights.flush()


class Topology:
    def topology_similarity_matrix(drug_drug_matrix):
        # Chuyển đổi ma trận drug_drug_matrix thành dạng ma trận numpy
        drug_drug_matrix = np.matrix(drug_drug_matrix)

        # Số lượng nodes (thuốc)
        drug_num = len(drug_drug_matrix)

        # Khởi tạo các ma trận tương tự với kích thước (drug_num x drug_num)
        common_similarity_matrix = np.zeros(shape=(drug_num, drug_num))  # Ma trận tương tự hàng xóm chung
        AA_similarity_matrix = np.zeros(shape=(drug_num, drug_num))  # Ma trận tương tự Adamic-Adar
        RA_similarity_matrix = np.zeros(shape=(drug_num, drug_num))  # Ma trận tương tự Resource Allocation
        Katz_similarity_matrix = inv(np.identity(drug_num) - 0.05 * drug_drug_matrix) - np.identity(drug_num)
        # Ma trận tương tự Katz: (I - βA)^(-1) - I với β = 0.05

        # Tính ACT similarity dựa trên ma trận Laplacian normalized
        D = np.diag((drug_drug_matrix.sum(axis=1)).getA1())  # Ma trận đường chéo với tổng dòng của drug_drug_matrix
        L = D - drug_drug_matrix  # Ma trận Laplacian
        I = np.identity(len(drug_drug_matrix))  # Ma trận đơn vị
        D1 = np.diag(np.power(np.diag(D), (-0.5)))  # Ma trận D^(-1/2)
        LL = D1 * L * D1  # L_norm = D^(-1/2) * L * D^(-1/2)
        LL = pinv(LL)
        LL = np.matrix(LL)
        ACT_similarity_matrix = np.zeros(shape=(drug_num, drug_num))  # Khởi tạo ma trận ACT similarity

        # Tính ACT similarity
        for i in range(0, len(drug_drug_matrix)):
            ACT_similarity_matrix[i, i] = 1  # Giá trị trên đường chéo của ACT similarity là 1
            for j in range(i + 1, len(drug_drug_matrix)):
                ACT_similarity_matrix[i, j] = 1 / (
                    1 + LL[i, i] + LL[j, j] - 2 * LL[i, j]
                )  # Công thức tính ACT similarity
                ACT_similarity_matrix[j, i] = ACT_similarity_matrix[i, j]  # Do ma trận đối xứng

        # Tính Random Walk with Restart (RWR) similarity
        alpha = 0.8  # Xác suất restart
        D = np.diag((drug_drug_matrix.sum(axis=1)).getA1())  # Ma trận đường chéo với tổng dòng của drug_drug_matrix
        D = np.matrix(D)
        N = pinv(D) * drug_drug_matrix  # Ma trận chuẩn hóa của drug_drug_matrix
        RWR_similarity_matrix = (1 - alpha) * pinv(
            np.identity(drug_num) - alpha * N
        )  # Tính ma trận RWR
        RWR_similarity_matrix = RWR_similarity_matrix + np.transpose(
            RWR_similarity_matrix
        )  # Đảm bảo ma trận đối xứng

        # Trả về các ma trận tương tự
        return (
            np.matrix(common_similarity_matrix),
            np.matrix(AA_similarity_matrix),
            np.matrix(RA_similarity_matrix),
            np.matrix(Katz_similarity_matrix),
            np.matrix(ACT_similarity_matrix),
            np.matrix(RWR_similarity_matrix),
        )

def load_csv(filename, type):
    matrix_data = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row_vector in csvreader:
            if type == "int":
                matrix_data.append(list(map(int, row_vector[1:])))  
            else:
                matrix_data.append(
                    list(map(float, row_vector[1:]))
                )  
    return np.matrix(matrix_data)

def modelEvaluation(real_matrix, predict_matrix, testPosition, featurename):
    # Khởi tạo danh sách để lưu trữ nhãn thực và xác suất dự đoán
    real_labels = []
    predicted_probability = []

    # Duyệt qua các vị trí trong testPosition
    for i in range(0, len(testPosition)):
        # Lưu nhãn thực (0 hoặc 1) từ ma trận real_matrix
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

        # Lưu giá trị dự đoán từ ma trận predict_matrix
        predicted_probability.append(predict_matrix[testPosition[i][0], testPosition[i][1]])

    real_labels = np.array(real_labels)
    predicted_probability = np.array(predicted_probability)

    precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)

    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))

    # Tính F1-score cho mỗi threshold trong pr_thresholds
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + recall[k]) > 0:  # Kiểm tra tránh chia cho 0
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0

    # Tìm threshold cho F-measure tối đa
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    # Tính ROC curve và AUC cho ROC
    fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
    auc_score = auc(fpr, tpr)

    # Dự đoán nhãn dựa trên threshold tối ưu
    predicted_score = np.zeros(len(real_labels))
    predicted_score[predicted_probability > threshold] = 1  # Nếu xác suất lớn hơn threshold, dự đoán là 1

    # Tính F1-score, Accuracy, Precision, Recall cho kết quả dự đoán
    f = f1_score(real_labels, predicted_score)
    accuracy = accuracy_score(real_labels, predicted_score)
    precision = precision_score(real_labels, predicted_score)
    recall = recall_score(real_labels, predicted_score)

    # In kết quả
    print("results for feature:" + featurename)
    print("AUC score:" + str(auc_score))
    print("AUPR score:" + str(aupr_score))
    print("precision score:" + str(precision))
    print("recall score:" + str(recall))
    print("accuracy score:" + str(accuracy))
    print("F-measure score:" + str(f))

    # Tổng hợp các chỉ số đánh giá
    results = [auc_score, aupr_score, precision, recall, accuracy, f]
    return results


def fitFunction(individual, parameter1, parameter2):
    """
    Hàm fitness cho thuật toán di truyền, dùng để đánh giá chất lượng của một vector trọng số trong mô hình ensemble.
    """
    # Giả sử individual là một vector trọng số của các mô hình trong ensemble.
    # Chúng ta sẽ kết hợp các dự đoán của các mô hình này bằng cách nhân với trọng số tương ứng và cộng lại.
    real_labels = parameter1  # 'parameter1' là nhãn thực của dữ liệu.
    multiple_prediction = parameter2  # 'parameter2' là danh sách chứa các dự đoán từ các mô hình trong ensemble.

    # Khởi tạo mảng ensemble_prediction với tất cả giá trị là 0
    ensemble_prediction = np.zeros(len(real_labels))

    # Tính tổng các dự đoán từ các mô hình trong ensemble, nhân với trọng số của từng mô hình.
    for i in range(0, len(multiple_prediction)):
        ensemble_prediction = ensemble_prediction + individual[i] * multiple_prediction[i]

    # Tính Precision-Recall curve với các nhãn thực và dự đoán của ensemble.
    precision, recall, pr_thresholds = precision_recall_curve(real_labels, ensemble_prediction)

    # Tính AUPR (Area Under Precision-Recall curve)
    aupr_score = auc(recall, precision)

    # Khởi tạo mảng chứa F1 score cho mỗi threshold.
    all_F_measure = np.zeros(len(pr_thresholds))

    # Tính F1 score tại từng threshold.
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + recall[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0

    # Tìm giá trị F1 score lớn nhất (tương ứng với threshold tối ưu).
    max_index = all_F_measure.argmax()

    # Trong bài toán tối ưu hóa với DEAP, mục tiêu là _tối đa hóa_ F-measure,
    # nhưng DEAP mặc định "tối đa hóa" nếu weight là dương, "tối thiểu hóa" nếu weight là âm.
    # Ở đây, ta có thể dùng trực tiếp F-measure, hoặc dùng -F nếu muốn tối thiểu hóa.
    # Hàm trả về một tuple như yêu cầu của DEAP.
    return (all_F_measure[max_index],)


def getParamter(real_matrix, multiple_matrix, testPosition):
    """
    Xác định trọng số của từng phương pháp dự đoán trong ensemble bằng Genetic Algorithm.
    """
    # Chuẩn bị dữ liệu ground truth
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    # Chuẩn bị dữ liệu dự đoán từ các mô hình base
    predict_probability = []  # Danh sách các dự đoán từ từng mô hình
    num_model = len(multiple_matrix)  # Số lượng mô hình

    for k in range(0, num_model):
        tmp_predict = []

        for i in range(0, len(testPosition)):
            tmp_predict.append(multiple_matrix[k][testPosition[i][0], testPosition[i][1]])
        predict_probability.append(tmp_predict)

    real_labels = np.array(real_labels)  # Ground truth (0/1)
    predict_probability = np.array(predict_probability)  # Dự đoán từ các mô hình base

    # Thiết lập Genetic Algorithm cho tối ưu trọng số
    # -------------------------------------------------------
    # Tạo loại cá thể với thuộc tính 'fitness' cần tối đa hóa (weight=1.0)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax)

    # Khởi tạo toolbox (bộ công cụ cho GA)
    toolbox = base.Toolbox()

    # Đăng ký hàm khởi tạo gen: giá trị random từ [0, 1]
    toolbox.register("weight", random.random)

    # Đăng ký hàm tạo Individual: một mảng 'double' với độ dài bằng số mô hình
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.weight, n=len(predict_probability))

    # Đăng ký hàm tạo population: danh sách các cá thể
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Đăng ký hàm đánh giá (fitness function)
    toolbox.register("evaluate", fitFunction, parameter1=real_labels, parameter2=predict_probability)

    # Đăng ký toán tử lai ghép (crossover), đột biến (mutation), và chọn lọc (selection)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Khởi tạo một hạt giống ngẫu nhiên để đảm bảo tính tái lập của kết quả
    random.seed(0)

    # Khởi tạo population (100 cá thể)
    pop = toolbox.population(n=100)

    # Khởi tạo Hall of Fame (lưu trữ cá thể tốt nhất)
    hof = tools.HallOfFame(1)

    # Đăng ký các thống kê về population: trung bình, độ lệch chuẩn, min, max
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Chạy thuật toán di truyền (EA Simple) với các tham số cấu hình
    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=50,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    # Lấy individual tốt nhất (trọng số tối ưu)
    best_individual = hof[0]
    weights = np.array(best_individual)

    print("Best weights found by GA:", weights)

    return weights


class MethodHub:
    @staticmethod
    def neighbor_method(similarity_matrix, train_drug_drug_matrix):
        """
        Phương pháp dựa vào ma trận tương tự hàng xóm (neighbor-based similarity).
        """
        # Chuẩn hóa similarity_matrix 
        similarity_matrix = np.matrix(similarity_matrix)
        train_drug_drug_matrix = np.matrix(train_drug_drug_matrix)

        D = np.diag((similarity_matrix.sum(axis=1)).getA1())
        D = np.matrix(D)
        N = pinv(D) * similarity_matrix
        return_matrix = np.matrix(train_drug_drug_matrix) * N

        return return_matrix

    @staticmethod
    def Label_Propagation(similarity_matrix, train_drug_drug_matrix):
        """
        Phương pháp Label Propagation để lan truyền nhãn trên đồ thị.
        """
        similarity_matrix = np.matrix(similarity_matrix)
        train_drug_drug_matrix = np.matrix(train_drug_drug_matrix)

        D = np.diag((similarity_matrix.sum(axis=1)).getA1())
        D = np.matrix(D)
        N = pinv(D) * similarity_matrix
        alpha = 0.9  # Hệ số lan truyền, kiểm soát sự trộn lẫn thông tin

        # Tính ma trận biến đổi dựa trên công thức Label Propagation
        transform_matrix = (1 - alpha) * pinv(np.identity(len(similarity_matrix)) - alpha * N)

        # Tính ma trận dự đoán
        return_matrix = transform_matrix * train_drug_drug_matrix

        # Đảm bảo ma trận đối xứng
        return_matrix = return_matrix + np.transpose(return_matrix)

        # Trả về ma trận dự đoán
        return return_matrix

    @staticmethod
    def generate_distrub_matrix(drug_drug_matrix):
        """
        Tạo ma trận nhiễu (disturb matrix) dựa trên thay đổi nhỏ trong cấu trúc đồ thị.
        """
        A = np.matrix(drug_drug_matrix)
        eigenvalues, eigenvectors = LA.eig(A)

        # Đảm bảo eigenvalues là thực
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        num = len(drug_drug_matrix)

        # Chọn các cạnh (links) và non-links
        row_index = []
        col_index = []
        for i in range(0, num):
            for j in range(i + 1, num):
                if drug_drug_matrix[i, j] == 1:
                    row_index.append(i)
                    col_index.append(j)

        select_num = len(row_index)
        delta_A = np.zeros(shape=(num, num))
        delta_eigenvalues = np.zeros(select_num)

        for i in range(0, select_num):
            delta_A[row_index[i], col_index[i]] = 1
            delta_A[col_index[i], row_index[i]] = 1

        for k in range(0, select_num):
            vec = eigenvectors[:, k]
            tmp_sum = 0
            for i in range(0, num):
                for j in range(0, num):
                    tmp_sum += delta_A[i, j] * vec[i, 0] * vec[j, 0]
            delta_eigenvalues[k] = tmp_sum / (vec.T * vec)

        reconstructed_A = np.zeros(shape=(num, num))
        for i in range(0, num):
            reconstructed_A = reconstructed_A + (eigenvalues[i] + delta_eigenvalues[i]) * eigenvectors[:, i] * eigenvectors[
                :, i
            ].T

        return_matrix = reconstructed_A + np.transpose(reconstructed_A)
        return return_matrix

    @staticmethod
    def disturb_matrix_method(train_drug_drug_matrix):
        """
        Phương pháp disturb matrix: thêm nhiễu dựa vào eigen-decomposition.
        """
        A = np.matrix(train_drug_drug_matrix)
        delta_A, row_index, col_index, select_num = MethodHub.generate_distrub_matrix(A)
        return delta_A

    @staticmethod
    def internal_determine_parameter(drug_drug_matrix):
        """
        Hàm nội bộ dùng để xác định tham số của một số phương pháp dựa trên topology.
        """
        return internal_determine_parameter(drug_drug_matrix)


def ensemble_method(drug_drug_matrix, train_drug_drug_matrix, testPosition):
    """
    Kết hợp nhiều phương pháp dựa trên các ma trận tương tự và lan truyền nhãn để dự đoán ma trận tương tác thuốc-drug (drug-drug interaction matrix).
    """

    # Tải các ma trận tương tự (similarity matrix) từ các nguồn khác nhau
    chem_sim_similarity_matrix = load_csv("dataset/chem_Jacarrd_sim.csv", "float")  # Ma trận tương tự hóa học
    target_similarity_matrix = load_csv("dataset/target_Jacarrd_sim.csv", "float")  # Ma trận tương tự mục tiêu
    transporter_similarity_matrix = load_csv("dataset/transporter_Jacarrd_sim.csv", "float")  # Ma trận tương tự vận chuyển
    enzyme_similarity_matrix = load_csv("dataset/enzyme_Jacarrd_sim.csv", "float")  # Ma trận tương tự enzyme
    pathway_similarity_matrix = load_csv("dataset/pathway_Jacarrd_sim.csv", "float")  # Ma trận tương tự pathway
    indication_similarity_matrix = load_csv("dataset/indication_Jacarrd_sim.csv", "float")  # Ma trận tương tự indication
    label_similarity_matrix = load_csv("dataset/sideeffect_Jacarrd_sim.csv", "float")  # Ma trận tương tự side effect
    offlabel_similarity_matrix = load_csv("dataset/offsideeffect_Jacarrd_sim.csv", "float")  # Ma trận tương tự off-label

    # Khởi tạo danh sách lưu kết quả và ma trận dự đoán từ các phương pháp khác nhau
    multiple_matrix = []  # Danh sách các ma trận dự đoán
    multiple_result = []  # Danh sách các kết quả đánh giá

    print("********************************************************")

    # Áp dụng phương pháp neighbor_method với từng ma trận tương tự
    predict_matrix = MethodHub.neighbor_method(chem_sim_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "chem_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    # Tương tự với các ma trận tương tự khác (target, transporter, enzyme, pathway, indication...)
    predict_matrix = MethodHub.neighbor_method(target_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "target_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(transporter_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "transporter_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(enzyme_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "enzyme_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(pathway_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "pathway_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(indication_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "indication_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(label_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "sideeffect_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(offlabel_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "offsideeffect_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    # Áp dụng Label Propagation trên các ma trận tương tự
    predict_matrix = MethodHub.Label_Propagation(chem_sim_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "chem_label")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.Label_Propagation(target_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "target_label")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    # Áp dụng với các ma trận khác (transporter, enzyme, pathway, indication...)
    predict_matrix = MethodHub.Label_Propagation(transporter_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "transporter_label")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.Label_Propagation(enzyme_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "enzyme_label")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.Label_Propagation(pathway_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "pathway_label")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.Label_Propagation(indication_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "indication_label")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.Label_Propagation(label_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "sideeffect_label")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.Label_Propagation(offlabel_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "offsideeffect_label")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    # Tính các ma trận tương tự topology
    (
        common_similarity_matrix,
        AA_similarity_matrix,
        RA_similarity_matrix,
        Katz_similarity_matrix,
        ACT_similarity_matrix,
        RWR_similarity_matrix,
    ) = Topology.topology_similarity_matrix(train_drug_drug_matrix)

    predict_matrix = MethodHub.neighbor_method(common_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "common_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(AA_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "AA_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(RA_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "RA_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(Katz_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "Katz_similarity_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(ACT_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "ACT_similarity_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    predict_matrix = MethodHub.neighbor_method(RWR_similarity_matrix, train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "RWR_similarity_neighbor")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    # Áp dụng phương pháp disturb_matrix_method
    predict_matrix = MethodHub.disturb_matrix_method(train_drug_drug_matrix)
    results = modelEvaluation(drug_drug_matrix, predict_matrix, testPosition, "disturb_matrix_label")
    multiple_result.append(results)
    multiple_matrix.append(predict_matrix)

    # Trả về danh sách các ma trận dự đoán và kết quả đánh giá
    return multiple_matrix, multiple_result


def internal_determine_parameter(drug_drug_matrix):
    """
    Tìm các tham số tối ưu bằng cách chạy hold-out validation và tối ưu logistic regression / trọng số.
    """
    # Dùng holdout_by_link để chia dữ liệu thành train/test
    train_matrix, test_matrix, test_position = holdout_by_link(drug_drug_matrix, ratio=0.2, seed=0)

    # Lấy các dự đoán từ nhiều phương pháp
    multiple_matrix, _ = ensemble_method(drug_drug_matrix, train_matrix, test_position)

    # Tính trọng số ensemble bằng getParamter (Genetic Algorithm)
    weights = getParamter(drug_drug_matrix, multiple_matrix, test_position)

    # Ở đây, bạn có thể thêm logistic regression (cf1, cf2) nếu muốn.
    cf1 = None
    cf2 = None

    return weights, cf1, cf2


def holdout_by_link(drug_drug_matrix, ratio, seed):
    """
    Chia dữ liệu theo các liên kết (links) thành tập train/test với tỉ lệ 'ratio'.
    """
    link_number = 0
    link_position = []

    # Xác định tất cả các liên kết trong ma trận
    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[i, j] == 1:
                link_number += 1
                link_position.append([i, j])

    link_position = np.array(link_position)

    random.seed(seed)
    index = np.arange(0, link_number)
    random.shuffle(index)

    test_link_num = int(link_number * ratio)
    test_index = index[0:test_link_num]
    test_index.sort()

    testLinkPosition = link_position[test_index]
    train_drug_drug_matrix = copy.deepcopy(drug_drug_matrix)

    for i in range(0, len(testLinkPosition)):
        train_drug_drug_matrix[testLinkPosition[i][0], testLinkPosition[i][1]] = 0
        train_drug_drug_matrix[testLinkPosition[i][1], testLinkPosition[i][0]] = 0

    testPosition = list(testLinkPosition)

    return train_drug_drug_matrix, drug_drug_matrix, testPosition


def ensemble_scoring(real_matrix, multiple_matrix, testPosition, weights, cf1=None, cf2=None):
    """
    Tính điểm số ensemble (và logistic regression nếu có) trên tập testPosition.
    """
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    real_labels = np.array(real_labels)

    multiple_predict = []
    for k in range(0, len(multiple_matrix)):
        tmp_predict = []
        for i in range(0, len(testPosition)):
            tmp_predict.append(multiple_matrix[k][testPosition[i][0], testPosition[i][1]])
        multiple_predict.append(tmp_predict)

    multiple_predict = np.array(multiple_predict)

    # Ensemble tuyến tính với trọng số
    ensemble_prediction = np.zeros(len(real_labels))
    for i in range(0, len(multiple_predict)):
        ensemble_prediction = ensemble_prediction + weights[i] * multiple_predict[i]

    # Chuẩn hóa về [0,1] để ổn định
    min_val = ensemble_prediction.min()
    max_val = ensemble_prediction.max()
    if max_val != min_val:
        ensemble_prediction = (ensemble_prediction - min_val) / (max_val - min_val)
    else:
        ensemble_prediction = np.zeros_like(ensemble_prediction)

    # Tính các chỉ số đánh giá dựa trên nhãn thực tế và dự đoán
    result = calculate_metric_score(real_labels, ensemble_prediction)  # Ensemble với trọng số

    # Placeholder cho logistic regression nếu muốn triển khai
    result_cf1 = calculate_metric_score(real_labels, ensemble_prediction)  # Logistic regression L1 (chưa triển khai riêng)
    result_cf2 = calculate_metric_score(real_labels, ensemble_prediction)  # Logistic regression L2 (chưa triển khai riêng)

    # Trả về kết quả
    return result, result_cf1, result_cf2


def calculate_metric_score(real_labels, predicted_probability):
    """
    Tính các chỉ số AUC, AUPR, Precision, Recall, Accuracy, F1 cho một vector dự đoán.
    """
    real_labels = np.array(real_labels)
    predicted_probability = np.array(predicted_probability)

    precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + recall[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0

    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
    auc_score = auc(fpr, tpr)

    predicted_score = np.zeros(len(real_labels))
    predicted_score[predicted_probability > threshold] = 1

    f = f1_score(real_labels, predicted_score)
    accuracy = accuracy_score(real_labels, predicted_score)
    precision_val = precision_score(real_labels, predicted_score)
    recall_val = recall_score(real_labels, predicted_score)

    print("results for feature: weighted_scoring")
    print("AUC score:" + str(auc_score))
    print("AUPR score:" + str(aupr_score))
    print("precision score:" + str(precision_val))
    print("recall score:" + str(recall_val))
    print("accuracy score:" + str(accuracy))
    print("F-measure score:" + str(f))

    results = [auc_score, aupr_score, precision_val, recall_val, accuracy, f]
    return results


def main(
    runtimes: int = 20,
    cv_num: int = 3,
    drug_drug_matrix_path: str = "dataset/drug_drug_matrix.csv",
    results_prefix: str = "result/result_on_our_dataset_3CV",
    weights_prefix: str = "result/weights_on_our_dataset_3CV",
) -> None:
    """
    Chạy quy trình cross-validation và ghi các chỉ số đánh giá theo từng fold cùng với trọng số ra ổ đĩa.

    Tham số:
    - runtimes: Số lần chạy lặp lại / số random seed cần thực hiện.
    - cv_num: Số lượng fold cho cross-validation (ví dụ: 3 cho 3-fold CV).
    - drug_drug_matrix_path: Đường dẫn đến file CSV chứa ma trận tương tác thuốc–thuốc dạng nhị phân.
    - results_prefix: Tiền tố (prefix) dùng để tạo tên cho các file output lưu kết quả đánh giá.
    - weights_prefix: Tiền tố cho các file output lưu trọng số được học bởi mô hình ensemble.
    """
    drug_drug_matrix = load_csv(drug_drug_matrix_path, "int")

    # Ensure output directories exist
    results_dir = os.path.dirname(results_prefix)
    weights_dir = os.path.dirname(weights_prefix)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)

    for seed in range(runtimes):
        file_results_path = f"{results_prefix}_{seed}.txt"
        weights_results_path = f"{weights_prefix}_{seed}.txt"
        with open(file_results_path, "w") as file_results, open(weights_results_path, "w") as file_weights:
            cross_validation(drug_drug_matrix, cv_num, seed, file_results, file_weights)


if __name__ == "__main__":
    main()
