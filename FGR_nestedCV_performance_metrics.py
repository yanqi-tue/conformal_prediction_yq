from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from sklearn import metrics
import pandas as pd
import csv
import xlrd
import random
from numpy.random import shuffle
# from hyperimpute.plugins.imputers import Imputers
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from operator import itemgetter
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn import svm
from sklearn.svm import OneClassSVM
import scipy.stats as st
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#from hyperimpute.plugins.imputers import Imputers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from scipy.stats import mannwhitneyu
import seaborn as sns

from imblearn.over_sampling import SMOTE
#from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
#from sklearn.datasets import load_breast_cancer
#from tabpfn import TabPFNClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
import statsmodels.api as sm


try:
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:
    ExplainableBoostingClassifier = None



# function to read the data and fill it to the same size as the GDM sample# function to read the data and fill it to the same size as the GDM sample
def data_read_and_fill_cat(dataset_num, variable_name, label_ID, factor_all):
    # dataset_num is the number of the dataset where factor is located, e.g. the dataset that having risk factor 'Age' is s01, and the dataset number of s01 is 41;
    # variable_name is the name of risk factor in dataset, e.g, the variable name of risk factor 'Age' in s01 is 'S01B01'
    # Taking label_ID as the standard, by comparing label_ID with current list's ID, add elements or delete elements to current list
    factor = []  # a list for risk factor value
    factor_ID = []  # a list for the ID of risk factor, such as '00001U'
    k = 0  # k is the variable number in the dataset
    for i in range(1, len(full_variable_name[dataset_num])):
        if str(full_variable_name[dataset_num][i]) == variable_name:  # located the position of risk factor
            k = i  # k is the number that the
    for i in range(1, len(full_data[dataset_num])):
        if (full_data[dataset_num][i][k] == 'D') or (full_data[dataset_num][i][k] == '') or (
                full_data[dataset_num][i][k] == 'S') or (full_data[dataset_num][i][k] == 'R') or (
                full_data[dataset_num][i][k] == 'M') or (
                full_data[dataset_num][i][k] == 'N') or (full_data[dataset_num][i][k] == 'E')or (
                full_data[dataset_num][i][k] == 'P') :  # 'D' '' 'S' means the original data is missing
            factor.append(np.nan)  # set -1 for the missing data
        else:
            factor.append((full_data[dataset_num][i][k]))  # read risk factor value
        factor_ID.append(full_data[dataset_num][i][0])  # read risk factor ID
    factor_fill = []  # list to hold the value after comparison with label_ID
    for i in range(len(label_ID)):
        if label_ID[i] not in factor_ID:
            factor_fill.append(np.nan)  # label_ID[i] not in factor_ID means the current risk factor ID list don't have the label_ID[i], choose to fill '0' for the missing value in the factor_fill list
        else:
            position = factor_ID.index(label_ID[i])  # If existed, find it and then fill it into factor_fill list
            factor_fill.append((factor[position]))
    factor_all.append(factor_fill)
    return factor_all

def data_read_and_fill_fornum(dataset_num, variable_name, label_ID):
    # dataset_num is the number of the dataset where factor is located, e.g. the dataset that having risk factor 'Age' is s01, and the dataset number of s01 is 41;
    # variable_name is the name of risk factor in dataset, e.g, the variable name of risk factor 'Age' in s01 is 'S01B01'
    # Taking label_ID as the standard, by comparing label_ID with current list's ID, add elements or delete elements to current list
    factor = []  # a list for risk factor value
    factor_ID = []  # a list for the ID of risk factor, such as '00001U'
    k = 0  # k is the variable number in the dataset
    for i in range(1, len(full_variable_name[dataset_num])):
        if str(full_variable_name[dataset_num][i]) == variable_name:  # located the position of risk factor
            k = i  # k is the number that the
    for i in range(1, len(full_data[dataset_num])):
        if (full_data[dataset_num][i][k] == 'D') or (full_data[dataset_num][i][k] == '') or (
                full_data[dataset_num][i][k] == 'R') or (
                full_data[dataset_num][i][k] == 'S') or (full_data[dataset_num][i][k] == 'M') or (
                full_data[dataset_num][i][k] == 'N') or (
                full_data[dataset_num][i][k] == 'P') or (full_data[dataset_num][i][k] == 'E'):  # 'D' '' 'S' means the original data is missing
            factor.append(np.nan)  # set 0 for the missing data
        else:
            factor.append(full_data[dataset_num][i][k])  # read risk factor value
        factor_ID.append(full_data[dataset_num][i][0])  # read risk factor ID
    factor_fill = []  # list to hold the value after comparison with label_ID
    for i in range(len(label_ID)):
        if label_ID[i] not in factor_ID:
            factor_fill.append(np.nan)  # label_ID[i] not in factor_ID means the current risk factor ID list don't have the label_ID[i], choose to fill '0' for the missing value in the factor_fill list
        else:
            position = factor_ID.index(label_ID[i])  # If existed, find it and then fill it into factor_fill list
            factor_fill.append(float(factor[position]))
    #    factor_all.append(factor_fill)
    return factor_fill

def data_read_and_fill_fornum_placanta(dataset_num, variable_name, label_ID):
    # dataset_num is the number of the dataset where factor is located, e.g. the dataset that having risk factor 'Age' is s01, and the dataset number of s01 is 41;
    # variable_name is the name of risk factor in dataset, e.g, the variable name of risk factor 'Age' in s01 is 'S01B01'
    # Taking label_ID as the standard, by comparing label_ID with current list's ID, add elements or delete elements to current list
    dataset_num = 39
    factor = []  # a list for risk factor value
    factor_ID = []  # a list for the ID of risk factor, such as '00001U'
    k = 0  # k is the variable number in the dataset
    Visit_num = 0
    ID_PLACENTA_ANALYSIS = []
    for i in range(1, len(full_variable_name[dataset_num])):
        if str(full_variable_name[dataset_num][i]) == variable_name:  # located the position of risk factor
            k = i  # k is the number that the
    for i in range(1, len(full_variable_name[dataset_num])):
        if str(full_variable_name[dataset_num][i]) == 'Visit':  # located the position of risk factor
            Visit_num = i  # k is the number that the
    for i in range(1, len(full_data[dataset_num])):
        if (full_data[dataset_num][i][Visit_num] == '1'):
            factor_ID.append(full_data[dataset_num][i][0])
            if (full_data[dataset_num][i][k] == 'D') or (full_data[dataset_num][i][k] == '') or (
                    full_data[dataset_num][i][k] == 'R') or (
                    full_data[dataset_num][i][k] == 'S') or (full_data[dataset_num][i][k] == 'M') or (
                    full_data[dataset_num][i][k] == 'N'):  # 'D' '' 'S' means the original data is missing
                factor.append(np.nan)  # set 0 for the missing data
            else:
                factor.append(float(full_data[dataset_num][i][k]))

            # read risk factor value
        # read risk factor ID
    MEAN=np.nanmedian(factor)
    factor_fill = []  # list to hold the value after comparison with label_ID
    for i in range(len(label_ID)):
        if label_ID[i] not in factor_ID:
            factor_fill.append(
                np.nan)  # label_ID[i] not in factor_ID means the current risk factor ID list don't have the label_ID[i], choose to fill '0' for the missing value in the factor_fill list
        else:
            position = factor_ID.index(label_ID[i])  # If existed, find it and then fill it into factor_fill list
            factor_fill.append(float(factor[position])/MEAN)
    #    factor_all.append(factor_fill)
    return factor_fill

def placanta_ID(label_ID):
    # dataset_num is the number of the dataset where factor is located, e.g. the dataset that having risk factor 'Age' is s01, and the dataset number of s01 is 41;
    # variable_name is the name of risk factor in dataset, e.g, the variable name of risk factor 'Age' in s01 is 'S01B01'
    # Taking label_ID as the standard, by comparing label_ID with current list's ID, add elements or delete elements to current list
    variable_name = 'AMAD12'
    dataset_num = 39
    factor = []  # a list for risk factor value
    factor_ID = []  # a list for the ID of risk factor, such as '00001U'
    k = 0  # k is the variable number in the dataset
    Visit_num = 0
    ID_PLACENTA_ANALYSIS = []
    # for i in range(1, len(full_variable_name[dataset_num])):
    #     if str(full_variable_name[dataset_num][i]) == variable_name:  # located the position of risk factor
    #         k = i  # k is the number that the
    for i in range(1, len(full_variable_name[dataset_num])):
        if str(full_variable_name[dataset_num][i]) == 'Visit':  # located the position of risk factor
            Visit_num = i  # k is the number that the
    for i in range(1, len(full_data[dataset_num])):
        if (full_data[dataset_num][i][Visit_num] == '1') and (full_data[dataset_num][i][0]) in label_ID:
            factor_ID.append(full_data[dataset_num][i][0])

    #    factor_all.append(factor_fill)
    return factor_ID

def feature_select(full_variable_name, full_variable_type, dataset_num, full_variable_code_list, factor_all_num,
                   factor_all_cat):
    LENGTH = len(full_variable_name[dataset_num])

    for i in range(6, LENGTH):
        if (full_variable_type[dataset_num][i] == 'Numeric'):
            factor_all_num.append(data_read_and_fill_fornum(dataset_num, full_variable_name[dataset_num][i], new_ID))

        #            featurenum1.append(data_read_and_fill_fornum(dataset_num, full_variable_name[dataset_num][i], new_ID)[1])
        elif (full_variable_type[dataset_num][i] == 'Coded'):
            factor_all_cat = (
                data_read_and_fill_cat(dataset_num, full_variable_name[dataset_num][i], new_ID, factor_all_cat))

    return factor_all_num, factor_all_cat

def standardlize(data):
    MEDIAN=np.median(data)
    STD=3.0*np.std(data)
    new_data=[]
    for i in range(len(data)):
        if (data[i]<=MEDIAN):
            if (MEDIAN-data[i]<STD):
                new_data.append(0.5-0.5*((MEDIAN-data[i])/STD))
            else:
                new_data.append(0)
        else:
            if (data[i]-MEDIAN<STD):
                new_data.append(0.5+0.5*(data[i]-MEDIAN)/STD)
            else:
                new_data.append(1.0)
    return new_data

# dataset num decide where to load the data

def Reshape(datafile, ID_datafile, ID):
    Ldata = len(datafile)
    LID = len(ID)
    new_data_file = []
    for i in range(LID):
        if ID[i] in ID_datafile:
            INDEX = ID_datafile.index(ID[i])
            new_data_file.append(datafile[INDEX])
        else:
            new_data_file.append(np.nan)
    return new_data_file


def COUNTNON0(datafile):
    s = 0
    Ldata = len(datafile)
    for i in range(Ldata):
        if (np.isnan(datafile[i]) == False):
            s = s + 1
    return s

def bio_plt(label, marker):
    new_bio1 = []
    new_bio0 = []
    for i in range(len(marker)):
        if (np.isnan(marker[i]) != True) and label[i] == 1:
            new_bio1.append(marker[i])
        if (np.isnan(marker[i]) != True) and label[i] == 0:
            new_bio0.append(marker[i])
    cat_bio0 = []
    cat_bio1 = []
    for i in range(9):
        cat_bio0.append(0)
        cat_bio1.append(0)
    for s in range(len(new_bio0)):
        if (new_bio0[s] < 0.25):
            cat_bio0[0] = cat_bio0[0] + 1
        elif (new_bio0[s] <= 0.5) and (new_bio0[s] > 0.25):
            cat_bio0[1] = cat_bio0[1] + 1
        elif (new_bio0[s] <= 0.75) and (new_bio0[s] > 0.5):
            cat_bio0[2] = cat_bio0[2] + 1
        elif (new_bio0[s] <= 1) and (new_bio0[s] > 0.75):
            cat_bio0[3] = cat_bio0[3] + 1
        elif (new_bio0[s] <= 1.25) and (new_bio0[s] > 1):
            cat_bio0[4] = cat_bio0[4] + 1
        elif (new_bio0[s] <= 1.5) and (new_bio0[s] > 1.25):
            cat_bio0[5] = cat_bio0[5] + 1
        elif (new_bio0[s] <= 1.75) and (new_bio0[s] > 1.5):
            cat_bio0[6] = cat_bio0[6] + 1
        elif (new_bio0[s] <= 2) and (new_bio0[s] > 1.75):
            cat_bio0[7] = cat_bio0[7] + 1
        elif (new_bio0[s] > 2):
            cat_bio0[8] = cat_bio0[8] + 1

    for s in range(len(new_bio1)):
        if (new_bio1[s] < 0.25):
            cat_bio1[0] = cat_bio1[0] + 1
        elif (new_bio1[s] <= 0.5) and (new_bio1[s] > 0.25):
            cat_bio1[1] = cat_bio1[1] + 1
        elif (new_bio1[s] <= 0.75) and (new_bio1[s] > 0.5):
            cat_bio1[2] = cat_bio1[2] + 1
        elif (new_bio1[s] <= 1) and (new_bio1[s] > 0.75):
            cat_bio1[3] = cat_bio1[3] + 1
        elif (new_bio1[s] <= 1.25) and (new_bio1[s] > 1):
            cat_bio1[4] = cat_bio1[4] + 1
        elif (new_bio1[s] <= 1.5) and (new_bio1[s] > 1.25):
            cat_bio1[5] = cat_bio1[5] + 1
        elif (new_bio1[s] <= 1.75) and (new_bio1[s] > 1.5):
            cat_bio1[6] = cat_bio1[6] + 1
        elif (new_bio1[s] <= 2) and (new_bio1[s] > 1.75):
            cat_bio1[7] = cat_bio1[7] + 1
        elif (new_bio1[s] > 2):
            cat_bio1[8] = cat_bio1[8] + 1

    for i in range(9):
        cat_bio0[i] = cat_bio0[i] / (len(new_bio0))
        cat_bio1[i] = cat_bio1[i] / (len(new_bio1))

    plt.xlim(0, 10)
    plt.ylim(0, 0.5)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x1 = [1 + 0.35, 2 + 0.35, 3 + 0.35, 4 + 0.35, 5 + 0.35, 6 + 0.35, 7 + 0.35, 8 + 0.35, 9 + 0.35]
    index_ls = ['<0.25', '0.25-0.5', '0.5-0.75', '0.75-1', '1-1.25', '1.25-1.5', '1.5-1.75', '1.75-2', '>2']
    plt.xticks(x, index_ls)
    # plt.ylabel('Family income per year($)',fontname="Times New Roman",fontsize=20)
    # plt.ylabel('Frequency')

    plt.title('1st trimester Nuchal translucency', fontname="Times New Roman", fontsize=20)
    plt.bar(x, cat_bio0, alpha=0.9, width=0.35, edgecolor='white', label='W/o GDM', lw=1)
    plt.bar(x1, cat_bio1, alpha=0.9, width=0.35, edgecolor='white', label='With GDM', lw=1)
    plt.xticks(fontname="Times New Roman", fontsize=18)
    ##plt.yticks(fontname='Times New Roman',fontsize=20)

    plt.yticks(fontname="Times New Roman", fontsize=20)
    font = {'family': 'Times New Roman',
            'size': 20,
            }
    plt.ylabel("Possibility", fontdict=font)
    plt.xlabel("MoM", fontdict=font)
    plt.legend(loc='upper right', prop={'size': 20,
                                        'family': 'Times New Roman',
                                        })
    plt.show()

    print()


def p_value(arrA, label):
    XA = len(arrA)
    a = []
    b = []
    for i in range(XA):
        if pd.isnull(arrA[i]) == False:
            if label[i] == 1:
                a.append(float(arrA[i]))
            else:
                b.append(float(arrA[i]))

    rate = len(a) / (len(b) + len(a))
    a = np.array(a)
    b = np.array(b)
    if len(a)>0 and len(b)>0:
        ma = np.nanmean(a)
        sa = np.nanstd(a)
        mb = np.nanmean(b)
        sb = np.nanstd(b)
        # plt.figure(figsize=(10,6))
        # plt.boxplot([a,b],showfliers=False)
        # plt.show()
        t, p = mannwhitneyu(a, b)
        D = ((abs(ma - mb)) / (sa + sb))
        return p, D
    else:
        return -999, -999


def data_read_and_fill_fornum1(dataset_num, variable_name, label_ID):
    # dataset_num is the number of the dataset where factor is located, e.g. the dataset that having risk factor 'Age' is s01, and the dataset number of s01 is 41;
    # variable_name is the name of risk factor in dataset, e.g, the variable name of risk factor 'Age' in s01 is 'S01B01'
    # Taking label_ID as the standard, by comparing label_ID with current list's ID, add elements or delete elements to current list
    factor = []  # a list for risk factor value
    factor_ID = []  # a list for the ID of risk factor, such as '00001U'
    k = 0  # k is the variable number in the dataset
    for i in range(1, len(full_variable_name[dataset_num])):
        if str(full_variable_name[dataset_num][i]) == variable_name:  # located the position of risk factor
            k = i  # k is the number that the
    for i in range(1, len(full_data[dataset_num])):
        if (full_data[dataset_num][i][k] == 'D') or (full_data[dataset_num][i][k] == '') or (
                full_data[dataset_num][i][k] == 'R') or (
                full_data[dataset_num][i][k] == 'S') or (full_data[dataset_num][i][k] == 'M') or (
                full_data[dataset_num][i][k] == 'N') or (
                full_data[dataset_num][i][k] == 'P') or (full_data[dataset_num][i][k] == 'E'):  # 'D' '' 'S' means the original data is missing
            factor.append(np.nan)  # set 0 for the missing data
        else:
            factor.append(full_data[dataset_num][i][k])  # read risk factor value
        factor_ID.append(full_data[dataset_num][i][0])  # read risk factor ID
    factor_fill = []  # list to hold the value after comparison with label_ID
    for i in range(len(label_ID)):
        if label_ID[i] not in factor_ID:
            factor_fill.append(np.nan)  # label_ID[i] not in factor_ID means the current risk factor ID list don't have the label_ID[i], choose to fill '0' for the missing value in the factor_fill list
        else:
            position = factor_ID.index(label_ID[i])  # If existed, find it and then fill it into factor_fill list
            factor_fill.append(float(factor[position]))
    #    factor_all.append(factor_fill)
    return factor_fill


def remove_correlated_features(df, threshold=0.9):
    """
    移除高度相关的特征，保留DataFrame的列名

    参数:
    df -- 输入的DataFrame
    threshold -- 相关性阈值，默认0.9

    返回:
    去除高相关特征后的DataFrame
    """
    # 确保输入是DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是pandas DataFrame")

    # 计算绝对值相关系数矩阵
    corr_matrix = df.corr().abs()

    # 创建上三角矩阵（不包括对角线）
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # 找出相关性高于阈值的列
    to_drop = [column for column in upper.columns
               if any(upper[column] > threshold)]

    # 打印将被移除的列（可选）
    if to_drop:
        print(f"将移除以下高相关特征: {to_drop}")
    else:
        print("没有发现高相关特征")

    # 返回去除高相关特征后的DataFrame
    return df.drop(columns=to_drop)

def p_value_label(a, b):
    A = []
    B = []
    # ranks_a = stats.rankdata(a)
    # a_transformed = stats.norm.ppf((ranks_a - 0.5) / len(a))
    MEAN_ALL = np.nanmean(a)
    for k in range(len(b)):
        if b[k] == 1 and pd.isnull(a[k])==False:
            A.append(a[k])
        elif b[k] == 0 and pd.isnull(a[k])==False:
            B.append(a[k])

    ma = float(np.nanmean(A))
    sa = float(np.nanstd(A))
    mb = float(np.nanmean(B))
    sb = float(np.nanstd(B))
    SD=float(np.nanstd(a))
    # plt.figure(figsize=(10,6))
    # plt.boxplot([a,b],showfliers=False)
    # plt.show()
    #t, p = stats.ttest_ind(A, B)
    if SD==0:
        return 0
    else:
        div = (ma - mb) / (SD)

        return div

def p_value_label_inside(a, b):
    A = []
    B = []
    ranks_a = stats.rankdata(a)

    MEAN_ALL = np.nanmean(a)
    non_nan_a=[]
    #non_nan_label=[]
    for k in range(len(b)):
        if pd.isnull(a[k])==False:
            non_nan_a.append(a)
        if b[k] == 1 and pd.isnull(a[k])==False:
            A.append(a[k])
        elif b[k] == 0 and pd.isnull(a[k])==False:
            B.append(a[k])
    #a_transformed = stats.norm.ppf((ranks_a - 0.5) / len(a))
    ma = float(np.nanmean(A))
    sa = float(np.nanstd(A))
    mb = float(np.nanmean(B))
    sb = float(np.nanstd(B))
    SD=float(np.nanstd(non_nan_a))
    # plt.figure(figsize=(10,6))
    # plt.boxplot([a,b],showfliers=False)
    # plt.show()
    #t, p = stats.ttest_ind(A, B)
    if SD==0:
        return 0
    else:
        div = (ma - mb) / (SD)

        return div

def label_top_10_percent(arr):
    """
    将一维数组中前10%最大值标记为1，其余为0

    参数:
    arr (np.ndarray): 输入的一维数组

    返回:
    np.ndarray: 标记数组，1表示前10%，其余为0
    """
    if arr.ndim != 1:
        raise ValueError("输入必须是一维长度为1000的数组")

    top_k = int(0.03 * len(arr))
    threshold = np.partition(arr, -top_k)[-top_k]
    labels = (arr >= threshold).astype(int)
    return labels

def label_bottom_10_percent(arr):
    """
    将一维数组中后10%最小值标记为1，其余为0

    参数:
    arr (np.ndarray): 输入的一维数组

    返回:
    np.ndarray: 标记数组，1表示后10%，其余为0
    """
    if arr.ndim != 1:
        raise ValueError("输入必须是一维数组")

    bottom_k = int(0.03 * len(arr))
    threshold = np.partition(arr, bottom_k - 1)[bottom_k - 1]
    labels = (arr <= threshold).astype(int)
    return labels

def find_threshold_for_positive_accuracy(y_prob, y_true, target_pos_accuracy, step=0.001):
    """
    在给定预测概率和标签下，寻找一个阈值，使得预测为正类的准确率接近 target_pos_accuracy。

    参数：
        y_prob: ndarray, 模型对正类的预测概率
        y_true: ndarray, 真实标签（0 或 1）
        target_pos_accuracy: float, 目标正类预测准确率（如 0.85）
        step: float, 搜索步长，默认0.001

    返回：
        best_thresh: 最佳阈值
        actual_pos_precision: 该阈值下预测为正类的准确率
        neg_class_accuracy: 负类样本中被正确分类为负类的比例
    """
    best_thresh = None
    best_diff = float('inf')
    actual_pos_precision = 0
    neg_class_accuracy = 0

    thresholds = np.arange(0.01, 0.99, step)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tn = np.sum((preds == 0) & (y_true == 0))

        if tp + fp == 0:
            continue  # 避免除以0

        pos_precision = tp / (tp + fp)
        diff = abs(pos_precision - target_pos_accuracy)

        if diff < best_diff:
            best_diff = diff
            best_thresh = t
            actual_pos_precision = pos_precision
            neg_class_accuracy = tn / np.sum(y_true == 0)

    return 1-neg_class_accuracy


def calib_curve_tail_merge(y_true, y_prob, cutoff=0.4, n_bins=10):
    """
    先对 y_prob<=cutoff 做等频分箱，再把 >cutoff 的样本合并为最后一组。

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_prob : array-like, shape (n_samples,)
    cutoff : float,  分界阈值，>cutoff 的预测会被合并
    n_bins : int,    cutoff 左侧想要的等频箱数
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # 1️⃣ 先取出≤cutoff部分
    mask_low = y_prob <= cutoff
    mask_high = y_prob > cutoff

    # --- 等频边界(不含最右边) ---
    quantiles = np.linspace(0, 1, n_bins + 1)  # n_bins等频 -> n_bins+1条边
    # 仅在低段做分位数
    bin_edges_low = np.quantile(y_prob[mask_low], quantiles)
    bin_edges_low[-1] = cutoff  # 把最后一条边改成 cutoff
    # 去重（极端情况可能出现重复边）
    bin_edges_low = np.unique(bin_edges_low)

    # 2️⃣ 构造完整边界，最后一条为 1.0
    bin_edges = np.concatenate([bin_edges_low, [1.0]])

    # 3️⃣ 计算每个 bin 的指标
    prob_pred, prob_true = [], []
    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:  # 正常 bin，左闭右开
            idx = (y_prob >= left) & (y_prob < right)
        else:  # 最后一个 bin：>=right_left
            idx = (y_prob >= left) & (y_prob <= right)

        if idx.sum() == 0:  # 可能出现空 bin，跳过
            continue
        prob_pred.append(y_prob[idx].mean())
        prob_true.append(y_true[idx].mean())

    return np.array(prob_pred), np.array(prob_true)

def calib_curve_custom_bins(y_true, y_prob, cutoff=0.4, bin_ratios=None):
    """
    自定义 bin 比例下的校准曲线计算（0-cutoff 区间内分箱 + 超过cutoff合并为一组）。

    Parameters
    ----------
    y_true : array-like
        真实标签。
    y_prob : array-like
        模型预测概率。
    cutoff : float
        分界阈值，大于此值的样本将被合并为一组。
    bin_ratios : list of float
        在 0~cutoff 范围内各 bin 占比，应加总为 1。

    Returns
    -------
    prob_pred : ndarray
        每个 bin 中的平均预测概率。
    prob_true : ndarray
        每个 bin 中的真实概率。
    bin_info : pd.DataFrame
        每个 bin 的边界、样本数、平均预测值、平均真实值。
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if bin_ratios is None:
        bin_ratios = [0.25, 0.25, 0.2, 0.10, 0.06, 0.05, 0.04, 0.03, 0.02]

    bin_ratios = np.array(bin_ratios)
    assert np.isclose(bin_ratios.sum(), 1.0), "bin_ratios must sum to 1.0"

    mask_low = y_prob <= cutoff
    mask_high = y_prob > cutoff

    low_probs = y_prob[mask_low]
    low_trues = y_true[mask_low]

    quantiles = np.cumsum(bin_ratios)
    low_edges = np.quantile(low_probs, quantiles[:-1])
    bin_edges = [low_probs.min()] + list(low_edges) + [cutoff, 1.0]
    bin_edges = np.unique(bin_edges)

    prob_pred, prob_true, sizes = [], [], []
    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i + 1]
        idx = (y_prob >= left) & (y_prob < right) if i < len(bin_edges) - 2 else (y_prob >= left) & (y_prob <= right)
        if idx.sum() == 0:
            continue
        prob_pred.append(y_prob[idx].mean())
        prob_true.append(y_true[idx].mean())
        sizes.append(idx.sum())

    bin_info = pd.DataFrame({
        "Bin Start": bin_edges[:-1],
        "Bin End": bin_edges[1:],
        "Size": sizes,
        "Pred Prob (mean)": prob_pred,
        "True Prob (mean)": prob_true
    })

    return np.array(prob_pred), np.array(prob_true), bin_edges


def calib_curve_uniform_tail_merge(y_true, y_prob, cutoff=0.4, n_bins=10):
    """
    在 0 ~ cutoff 范围内做等长度分箱，cutoff 以上的预测概率统一合并为最后一组。

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels.

    y_prob : array-like of shape (n_samples,)
        Predicted probabilities.

    cutoff : float
        分割阈值，大于该值的预测概率会被合并到最后一组。

    n_bins : int
        cutoff 之前划分的等长度箱数。

    Returns
    -------
    prob_pred : np.ndarray
        每个 bin 的平均预测概率。

    prob_true : np.ndarray
        每个 bin 的真实正类比例。
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # === Step 1: 构造等长度边界 + 最后一个 merge bin ===
    bin_edges = np.linspace(0.0, cutoff, n_bins + 1)  # n_bins 个箱子 => n_bins+1 个边界
    bin_edges = np.append(bin_edges, [1.0])  # 加入最后一组 [cutoff, 1.0]

    prob_pred, prob_true = [], []

    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i + 1]

        if i < len(bin_edges) - 2:
            idx = (y_prob >= left) & (y_prob < right)  # 左闭右开
        else:
            idx = (y_prob >= left) & (y_prob <= right)  # 最后一组包括 right=1.0

        if idx.sum() == 0:
            continue  # 跳过空 bin

        prob_pred.append(y_prob[idx].mean())
        prob_true.append(y_true[idx].mean())

    return np.array(prob_pred), np.array(prob_true)

def compute_calibration_curve(y_true, y_prob, bin_edges):
    """
    使用指定的 bin 边界计算校准曲线，并返回 Brier Score。

    参数:
    -------
    y_true : array-like
        真实标签 (0/1)。
    y_prob : array-like
        模型预测概率。
    bin_edges : array-like
        分箱的边界值（长度为 n+1，表示 n 个 bin）。

    返回:
    -------
    prob_pred : ndarray
        每个 bin 中的平均预测概率（x 轴）。
    prob_true : ndarray
        每个 bin 中的平均真实概率（y 轴）。
    brier : float
        Brier Score（整体预测概率的均方误差）。
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.asarray(bin_edges)

    prob_pred = []
    prob_true = []

    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            idx = (y_prob >= left) & (y_prob < right)
        else:
            idx = (y_prob >= left) & (y_prob <= right)

        if np.sum(idx) == 0:
            continue

        prob_pred.append(y_prob[idx].mean())
        prob_true.append(y_true[idx].mean())

    brier = brier_score_loss(y_true, y_prob)

    return np.array(prob_pred), np.array(prob_true), brier
from openpyxl import load_workbook


class WorksheetAdapter:
    """
    将 openpyxl 工作表包装成类似 xlrd worksheet 的对象，
    继续支持：
        worksheet.nrows
        worksheet.row_values(i)
    """

    def __init__(self, openpyxl_worksheet):
        self._rows = list(
            openpyxl_worksheet.iter_rows(values_only=True)
        )

    @property
    def nrows(self):
        return len(self._rows)

    def row_values(self, row_index):
        return list(self._rows[row_index])


def load_first_worksheet(file_path):
    """
    使用 openpyxl 读取第一个工作表，
    返回支持 xlrd 风格调用的 WorksheetAdapter。
    """
    workbook = load_workbook(
        file_path,
        data_only=True,
        read_only=True
    )

    openpyxl_worksheet = workbook.worksheets[0]
    worksheet_adapter = WorksheetAdapter(openpyxl_worksheet)

    workbook.close()

    return worksheet_adapter


# =============================================================================
# macOS project paths
# =============================================================================
# NOTE: In Python strings, the space in "conformal prediction" is written
# normally. The backslash is only needed when typing the path in a shell.
PROJECT_ROOT = r"/Users/ikaros/Project/conformal prediction"
DATA_ROOT = os.path.join(PROJECT_ROOT, "numom2b_new")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "numom2b_matched_score_analysis")
INTERMEDIATE_OUTPUT_DIR = os.path.join(RESULTS_ROOT, "intermediate_features")
os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(INTERMEDIATE_OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load variable information
# =============================================================================

worksheet = load_first_worksheet(
    os.path.join(DATA_ROOT, "nuMoM2b Variable Information.xlsx")
)

nrow = worksheet.nrows


# =============================================================================
# Load dataset names
# =============================================================================

dataset_name = []

dataset_name.append(
    str(worksheet.row_values(1)[0]).upper()
)

for i in range(2, worksheet.nrows):

    current_dataset = worksheet.row_values(i)[0]
    previous_dataset = worksheet.row_values(i - 1)[0]

    if current_dataset != previous_dataset:
        dataset_name.append(
            str(current_dataset).upper()
        )


# =============================================================================
# Load all variable names, types and code lists
# =============================================================================

full_variable_name = []
full_variable_type = []
full_variable_code_list = []

for i in range(len(dataset_name)):

    variable_name_middle = []
    variable_type_middle = []
    variable_code_list_middle = []

    for j in range(nrow):

        row = worksheet.row_values(j)

        if row[0] == dataset_name[i].lower():

            variable_name_middle.append(
                row[1]
            )

            variable_type_middle.append(
                row[3]
            )

            variable_code_list_middle.append(
                row[5]
            )

    full_variable_name.append(
        variable_name_middle
    )

    full_variable_type.append(
        variable_type_middle
    )

    full_variable_code_list.append(
        variable_code_list_middle
    )


# =============================================================================
# Load dataset descriptions
# =============================================================================

worksheet_dataset = load_first_worksheet(
    os.path.join(DATA_ROOT, "nuMoM2b Dataset Information.xlsx")
)

dataset_description = []

dataset_description.append(
    worksheet_dataset.row_values(3)[2]
)

for i in range(4, worksheet_dataset.nrows):

    current_description = worksheet_dataset.row_values(i)[2]
    previous_description = worksheet_dataset.row_values(i - 1)[2]

    if current_description != previous_description:
        dataset_description.append(
            current_description
        )


# =============================================================================
# Checks
# =============================================================================

print("Number of Excel rows:", worksheet.nrows)
print("Number of datasets:", len(dataset_name))
print("Number of variable groups:", len(full_variable_name))
print("Number of dataset descriptions:", len(dataset_description))
# -----------------------------------------------------------------------------------------------------------------------
#  input all data
full_data = []
for i in range(79):
    file_path = os.path.join(DATA_ROOT, "CSV", dataset_name[i] + ".CSV")
    with open(file_path, encoding='ISO-8859-1') as csv_file:
        csv_file_middle = []
        for row in csv.reader(csv_file):
            csv_file_middle.append(row)  # load every row in the target path and transfer it into fill_data
        full_data.append(csv_file_middle)

# -----------------------------------------------------------------------------------------------------------------------
# read diabetes information from dataset:'pregnancy_outcomes', number 40, variable name is 'oDM'
GDM = []
GDM_ID = []  # ID for patient in GDM list
for i in range(len(full_variable_name[40])):
    if (str(full_variable_name[40][i]) == 'oDM'):
        GDM = []
        for j in range(1, len(full_data[40][:])):
            GDM.append(full_data[40][j][i])
            GDM_ID.append(full_data[40][j][0])

#label1=np.load('D:\\numom2b\\hrv_93\\label1.npy')
#HRV_ID=np.load ('D:\\numom2b\\ECG510\\ID_all.npy')

GDM_relabel0 = []
GDM_ID_relabel0 = []
# GDM[i] == '3' means no GDM
# GDM[i] == '2' means GDM
# GDM[i] == '1' means preDM
# the meaning for relabel is to delete the the data without diagnosis which corresponding to '', and the preDM data,which corresponding to '1'
GDM_glm=[]
DM_glm=[]
for i in range(len(GDM)):
    if GDM[i] == '2':
        GDM_glm.append(1.0)
    else:
        GDM_glm.append(0.0)
    if GDM[i] == '1':
        DM_glm.append(1.0)
    else:
        DM_glm.append(0.0)
# GDM_index=[]
# for i in range(len(GDM)):
#     if GDM[i] == '3':
#         GDM_relabel0.append(0.0)
#         GDM_ID_relabel0.append(GDM_ID[i])
#         GDM_index.append(i)
#     if GDM[i] == '2':
#         GDM_relabel0.append(1.0)
#         GDM_ID_relabel0.append(GDM_ID[i])
#         GDM_index.append(i)
DM_ID=[]
DM_G_ID=[]
for i in range(len(GDM)):
    if GDM[i] == '1':
        DM_ID.append(GDM_ID[i])
    elif GDM[i] == '2':
        DM_G_ID.append(GDM_ID[i])

PEgHTN = (data_read_and_fill_fornum(40, 'PEgHTN', GDM_ID))  # Chronic hypertension based on CMDA01 & CMAE01
SGA=(data_read_and_fill_fornum(40, 'SGA_alex', GDM_ID))
SGA_2=(data_read_and_fill_fornum(40, 'SGA_oken', GDM_ID))
PTB=(data_read_and_fill_fornum(4, 'A09A03a', GDM_ID))
LGA=(data_read_and_fill_fornum(21, 'CBAA05a_Gm', GDM_ID))
# GAwksCA
# GAdysCA
# GAwksA09
# GAdysA09
# GA1=(data_read_and_fill_fornum(40, 'GAwksCA', GDM_ID))
# GA2=(data_read_and_fill_fornum(40, 'GAdysCA', GDM_ID))
# GA3=(data_read_and_fill_fornum(40, 'GAwksA09', GDM_ID))
# GA4=(data_read_and_fill_fornum(40, 'GAdysA09', GDM_ID))
# GA_wks_dys = np.array(GA1) - np.array(GA2) / 7





birth_weight=[]
birth_weight_test=[]
GA_wks_dys_LGA=[]
GDM_GA=[]
C_DM=[]
nonGDM_GA=[]
C_nonDM=[]
BW_DM=[]
BW_GDM=[]
BW_nonDM=[]
BW_DM_per=[]
BW_nonDM_per=[]
BW_per=[]
DM_label_raw=[]
DM_label=[]
GDM_DM_label=[]
BW_DM_before=[]
BW_nonDM_before=[]
DM_index=[]
DM_order=0
GA_save=[]
earlycsection=[]
LGA_macro=[]
LGA_nor=[]
GA_macro=[]
GA_nor=[]
Term_birth_index=[]
Non_indicated_PTB_mid_term=[]
Non_indicated_PTB_index=[]


# for i in range(len(GDM_ID)):
#     if DM_label_raw[i]=='2':
#         GDM_DM_label.append(1)
#     else:
#         GDM_DM_label.append(0)
#     if DM_label_raw[i]=='1':
#         DM_label.append(1)
#     else:
#         DM_label.append(0)
# a=0
# b=0
# c=0
# d=0
# for k in range(len(BW_DM)):
#     if BW_DM[k]==1 and LGA_label[k]==1:
#         a=a+1
#     elif BW_DM[k]==0 and LGA_label[k]==0:
#         d=d+1
#     elif BW_DM[k] == 1 and LGA_label[k] == 0:
#         c=c+1
#     else:
#         b=b+1
# odds_ratio=(a*d)/(b*c)
# P_val=stats.ttest_ind(BW_DM, BW_nonDM)
# P_val_per=stats.ttest_ind(BW_DM_per, BW_nonDM_per)
# percantage_fetal_weight = []
# for i in range(len(birth_weight)):
#     percantage_fetal_weight.append(EFW_percentage(birth_weight[i], GA_wks_dys_LGA[i]))

    # else:
    #     LGA_label.append(np.nan)
    #     LGA_ID.append(GDM_ID[i])

percantage_fetal_weight_1=[]


HTN_relabel = []
HTN_ID_relabel = []
PE_label=[]
GIH_label=[]
GA_PTB=[]
GA_PTB_label=[]
GA=np.array((data_read_and_fill_fornum(40, 'GAwksA09', GDM_ID)))
for i in range(len(GDM_ID)):
    if GA[i]<=36:
        GA_PTB.append(GDM_ID[i])
        GA_PTB_label.append(1)
    elif GA[i]>36:
        GA_PTB.append(GDM_ID[i])
        GA_PTB_label.append(0)
    if (PEgHTN[i] == 1.0) or  (PEgHTN[i] == 3.0) or (PEgHTN[i] == 2.0) or(PEgHTN[i] == 4.0):
        HTN_relabel.append(1.0)
        PE_label.append(1.0)
        GIH_label.append(0.0)
        HTN_ID_relabel.append(GDM_ID[i])
    elif (PEgHTN[i] == 5.0) or (PEgHTN[i] == 6.0):
        PE_label.append(0.0)
        GIH_label.append(1.0)
        HTN_relabel.append(1.0)
        HTN_ID_relabel.append(GDM_ID[i])
    else:
        PE_label.append(0.0)
        GIH_label.append(0.0)
        HTN_relabel.append(0.0)
        HTN_ID_relabel.append(GDM_ID[i])
#        GDM_ID_relabel.append(GDM_ID[i])
#     elif (PEgHTN[i] == 7.0) or (PEgHTN[i] == 5.0) or (PEgHTN[i] == 6.0):#or (PEgHTN[i] == 3.0):
#         HTN_relabel.append(0.0)
#         HTN_ID_relabel.append(GDM_ID[i])
# HRV_data=np.load('C:\\Users\\20204374\\Desktop\\phd\\lapdata\\hrv_93\\df_52_feature_quarter_test.npy',allow_pickle=True)
# HRV_name=np.load('C:\\Users\\20204374\\Desktop\\phd\\lapdata\\hrv_93\\feature_name_52_mean_sd.npy',allow_pickle=True)
#HRV_id=np.load('C:\\Users\\20204374\\Desktop\\phd\\lapdata\\ECG510\\ID_all.npy',allow_pickle=True)
#HRV_id=GDM_ID_relabel
odm=np.array((data_read_and_fill_fornum(40, 'oDM', GDM_ID)))
bw=np.array((data_read_and_fill_fornum(40, 'bw', GDM_ID)))
GA=np.array((data_read_and_fill_fornum(40, 'GAwksA09', GDM_ID)))
non_missing_index_GA=[]
newGA=[]
for i in range(len(GA)):
    if GA[i] <= 36:
        newGA.append(1)
        non_missing_index_GA.append(i)
    elif GA[i] > 36:
        newGA.append(0)
        non_missing_index_GA.append(i)

for i in range(len(odm)):
    if odm[i] == 2:
        odm[i]=1
    else:
        odm[i]=0
index_bw_non_missing=[]
new_bw=[]
new_bw_4500=[]
non_missing_index_bw=[]
for i in range(len(bw)):
    if pd.isnull(bw[i]) == False:
        index_bw_non_missing.append(i)
        new_bw.append(bw[i])
        non_missing_index_bw.append(i)
nn_4500=np.zeros(len(new_bw))

for X in range(len(new_bw)):
    if new_bw[X]>=4500:
        nn_4500[X]=1

    else:
        nn_4500[X] = 0
n_10=label_top_10_percent(np.array(new_bw))
#pee=np.array((data_read_and_fill_fornum(40, 'acog_PEgHTN', HRV_id)))
fgr=np.array((data_read_and_fill_fornum(26, 'CMAE08', GDM_ID)))
# for i in range(len(fgr)):
#     if fgr[i] == 2:
#         fgr[i]=0
fgr_relabel0=[]
fgr_ID_relabel0=[]
fgr_index=[]
for i in range(len(fgr)):
    if fgr[i] == 1:
        fgr_relabel0.append(1.0)
        fgr_ID_relabel0.append(GDM_ID[i])
        fgr_index.append(i)
    if fgr[i] == 2:
        fgr_relabel0.append(0.0)
        fgr_ID_relabel0.append(GDM_ID[i])
        fgr_index.append(i)
pee=np.array((data_read_and_fill_fornum(29, 'CMDA08a1', GDM_ID)))
pee_all=np.zeros(len(pee))
pee_time=np.array((data_read_and_fill_fornum(29, 'CMDA08A3_INT', GDM_ID)))
PTB=np.array((data_read_and_fill_fornum(4, 'A09A03a', GDM_ID)))
for i in range(len(PTB)):
    if PTB[i] != 1:
        PTB[i]=0

for i in range(len(pee)):
    if pee[i]>=2:
        pee_all[i]=1
    if pee[i] > 1 and pee_time[i]<=-21:
        pee[i]=1
    else:
        pee[i]=0
#PTB=np.array((data_read_and_fill_fornum(4, 'A09A03a', HRV_id)))
for i in range(len(PTB)):
    if PTB[i] != 1:
        PTB[i]=0

# for i in range(len(pee)):
#     if pee[i] <= 4:
#         pee[i]=1
#     else:
#         pee[i]=0
# sepsis=np.array((data_read_and_fill_fornum(21, 'CBAB02', HRV_id)))
# sepsis1=np.array((data_read_and_fill_fornum(30, 'CMEA02', HRV_id)))
# sepsis2=np.array((data_read_and_fill_fornum(27, 'CMBJ03', HRV_id)))

#new_ID = GDM_ID_relabel  # choose the GDM's label as the standard to select the all risk factor data
# new_ID = PTB_ID_relabel
# GDM_relabel=PTB_relabel
# LGA_label=GDM_relabel
#new_ID = HTN_ID_relabel
#new_ID=HRV_ID
new_ID=GDM_ID
#GDM_relabel=HTN_relabel
#GDM_relabel = np.array(GDM_relabel).T  # transfer to nparray, no necessary

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
factor_all_num = []  # list to hold all risk factor
factor_all_cat = []
featurenum = []
R_num=[]
R_cat=[]
# test=factor_all[1]


# test = pd.DataFrame(test, dtype=int)
# test=test.astype("category")
factor_all = []
variable_SDBname_cat=[]
variable_SDBname_num=[]

# (SDB_ID,SDB_num,SDB_cat,variable_SDBname_cat,variable_SDBname_num)=feature_SDB_all(full_data,full_variable_name,full_variable_type,new_ID,variable_SDBname_cat,variable_SDBname_num)
# factor_all_num=factor_all_num+SDB_num

print()
# SDB_label_HTN=[]
#
# SDB_label_GDM=[]
# for i in range(len(SDB_ID)):
#     INDEX=new_ID.index(SDB_ID[i])
#     SDB_label_GDM.append(GDM_relabel[INDEX])
# #    SDB_label_LGA.append(LGA_label[INDEX])
# GDM_relabel=SDB_label_GDM
# new_ID=SDB_ID
#(anxiety_num,non_anxiety_num, co_anxiety_num,EPDS_num,difficulty_num)=mental_read(full_data,full_variable_name,new_ID)
# (PAPP, HCG, HGBA1c,Hemoglobin,Hematocrit,MCV,Platelet,Nuchal_translucency,Trisomy_21,Trisomy_18,Hemoglobin_electrophoresis,Blood_type,Rh_factor,Antibody,Urine_culture) =feature_CLA_read(full_data,full_variable_name,full_variable_type,new_ID,variable_SDBname_cat,variable_SDBname_num)
# (SDB_ID,SDB_data)=feature_SDB(full_data,full_variable_name,new_ID)
# rate10,p10=p_value(Hemoglobin, GDM_relabel)
# rate11,p11=p_value(Hematocrit, GDM_relabel)
# rate12,p12=p_value(MCV, GDM_relabel)
# rate13,p13=p_value(HGBA1c, GDM_relabel)
# rate14,p14=p_value(Platelet, GDM_relabel)
# rate15,p15=p_value(Nuchal_translucency, GDM_relabel)
# new_ID=SDB_ID

# factor_all.append(data_read_and_fill(57, 'V1BA01_KG', new_ID))  # Weight - kg
# factor_all_num.append(data_read_and_fill_fornum(57, 'V1BA02a', new_ID)) #height

# factor_all_num.append(data_read_and_fill_fornum(24, 'CLAE05a1', new_ID))  # HgbA1c test - Result % (a)
# factor_all_num.append(data_read_and_fill_fornum(24, 'CLAA01a', new_ID))  # Complete blood count - Hemoglobin g/dL
# factor_all_num.append(data_read_and_fill_fornum(39, 'PLGF', new_ID))  # PLGF
# factor_all.append(data_read_and_fill_fornum(56, 'V1AD01a', new_ID)[0])# How much did you weigh before you got pregnant?
# factor_all_num.append(data_read_and_fill_fornum(56, 'V1AE01', new_ID))  # Including this pregnancy, how  many times have you been pregnant - Number of pregnancies
# variable_name_num.append('Gravidity')

# CLAC01c2
# ID_placenta = placanta_ID(new_ID)
# PLA_label = []
# record_INDEX = []
# for i in range(len(ID_placenta)):
#    INDEX = new_ID.index(ID_placenta[i])
#    record_INDEX.append(INDEX)
#    PLA_label.append(PTB_relabel[INDEX])
# t_val = 0
# for i in range(len(new_ID)):
#    if new_ID[i] in ID_placenta:
#        posi = ID_placenta.index(new_ID[i])
#        if PTB_relabel[i] != PLA_label[posi]:
#            t_val = t_val + 1


# (papp, hcg, A1c, Hemoglobin, Hematocrit, MCV, Platelet, Nuchal_translucency, Trisomy_21, Trisomy_18,Hemoglobin_electrophoresis, Blood_type, Rh_factor, Antibody, Urine_culture) = feature_CLA_read(full_data,
#                                                                                                 full_variable_name,
#                                                                                                 ID_placenta)
# A1c_1=p_value(A1c, PLA_label)
# Hemoglobin_1=p_value(Hemoglobin, PLA_label)
# Hematocrit_1=p_value(Hematocrit, PLA_label)
# MCV_1=p_value(MCV, PLA_label)
# Platelet_1=p_value(Platelet, PLA_label)
# Nuchal_translucency_1=p_value(Nuchal_translucency, PLA_label)
# ADAM12 = (data_read_and_fill_fornum_placanta(39, 'ADAM12', new_ID))
# ADAM12_1 = p_value(ADAM12, GDM_relabel)
#
# ENDOGLIN = (data_read_and_fill_fornum_placanta(39, 'ENDOGLIN', new_ID))
# ENDOGLIN_1 = p_value(ENDOGLIN, GDM_relabel)
#
# SFLT1 = (data_read_and_fill_fornum_placanta(39, 'SFLT1', new_ID))
# SFLT1_1 = p_value(SFLT1, GDM_relabel)
# VEGF = (data_read_and_fill_fornum_placanta(39, 'VEGF', new_ID))
# VEGF_1 = p_value(VEGF, GDM_relabel)
# AFP = (data_read_and_fill_fornum_placanta(39, 'AFP', new_ID))
# AFP_1= p_value(AFP, GDM_relabel)
# fbHCG = (data_read_and_fill_fornum_placanta(39, 'fbHCG', new_ID))
# fbHCG_1 = p_value(fbHCG, GDM_relabel)
# INHIBINA = (data_read_and_fill_fornum_placanta(39, 'INHIBINA', new_ID))
# INHIBINA_1 = p_value(INHIBINA, GDM_relabel)
# PAPPA = (data_read_and_fill_fornum_placanta(39, 'PAPPA', new_ID))
# PAPPA_1 = p_value(PAPPA, GDM_relabel)
# PLGF = (data_read_and_fill_fornum_placanta(39, 'PLGF', new_ID))
# PLGF_1 = p_value(PLGF, GDM_relabel)

# V1AD17 = data_read_and_fill_fornum(56, 'V1AD17', new_ID)
# #R_num.append(p_value(V1AD17, LGA_label))
# V1AD18 = data_read_and_fill_fornum(56, 'V1AD18', new_ID)
# #R_num.append(p_value(V1AD18, LGA_label))
# V1AD17a1 = data_read_and_fill_fornum(56, 'V1AD17a1', new_ID)
# V1AD17a2 = data_read_and_fill_fornum(56, 'V1AD17a2', new_ID)
# V1AD17a3 = data_read_and_fill_fornum(56, 'V1AD17a3', new_ID)
# V1AD17a1_1 = data_read_and_fill_fornum(56, 'V1AD17a1_1', new_ID)
# V1AD17a3_1 = data_read_and_fill_fornum(56, 'V1AD17a3_1', new_ID)
#
# variable_name_num = []
# variable_name_cat = []
# # variable_name_num=variable_name_num+variable_SDBname_num
# # Age = data_read_and_fill_fornum(41, 'S01B01', new_ID)
# # factor_all_num.append(Age)  # Age
# # variable_name_num.append('Age')
# # R_num.append(p_value(Age,LGA_label))
# # BMI = data_read_and_fill_fornum(34, 'BMI', new_ID)
# # factor_all_num.append(BMI)
# # R_num.append(p_value(BMI, LGA_label))
# # # factor_all_num.append(data_read_and_fill_fornum(34, 'BMI', ID_placenta))  # BMI
# # variable_name_num.append('BMI')
#
# # V1AD17
# # (factor_all_num,variable_name_num,R_num)=feature_select_vxx(full_variable_name, full_variable_type, 78, full_variable_code_list, factor_all_num,
# #                    new_ID,LGA_label,variable_name_num,R_num)
# # feature_select_v1l(full_variable_name, full_variable_type, 64, full_variable_code_list, factor_all_num,factor_all_cat,LGA_label,LGA_ID)
#
#
# CLAE05A_INT=(data_read_and_fill_fornum(24, 'CLAE05A_INT', new_ID))  # BMI
# CLAE05a1=(data_read_and_fill_fornum(24, 'CLAE05a1', new_ID))
# CLAE06=(data_read_and_fill_fornum(24, 'CLAE06', new_ID))
# CLAE06B1_INT=(data_read_and_fill_fornum(24, 'CLAE06B1_INT', new_ID))
# CLAE07=(data_read_and_fill_fornum(24, 'CLAE07', new_ID))
# CLAE07B1_INT=(data_read_and_fill_fornum(24, 'CLAE07B1_INT', new_ID))
# suspect_GDM=np.zeros(2748)
# for T in range(len(CLAE05A_INT)):
#     if CLAE05A_INT[T]<-170 and CLAE05a1[T]>=5.7:
#         suspect_GDM[T]=1
#     if CLAE06[T]==2 and CLAE06B1_INT[T]<-170:
#         suspect_GDM[T]=1
#     if CLAE07[T]==1 and CLAE07B1_INT[T]<-170:
#         suspect_GDM[T]=1
# factor_all_num.append(suspect_GDM)
# variable_name_num.append('suspect_GDM')
# # feature_select_cla_diabetes(full_variable_name, full_variable_type, 24, factor_all_num,factor_all_cat,
# #                        new_ID, variable_name_num, variable_name_cat)
# factor_all_num.append(data_read_and_fill_fornum(41, 'S01B01', new_ID))  # Age
# variable_name_num.append('Age')
# factor_all_num.append(data_read_and_fill_fornum1(34, 'BMI', new_ID))  # BMI
# variable_name_num.append('BMI')
# feature_select_cla(full_variable_name, full_variable_type, 24, factor_all_num,factor_all_cat,
#                        new_ID, variable_name_num, variable_name_cat)
# #EPDS_num=mental_read_EDPS(58,full_data, full_variable_name, new_ID)
# #re=p_value_label(EPDS_num, label1)
# # factor_all_num.append(EPDS_num)
# # variable_name_num.append('EPDS')
# sum_difficult=feature_select_v1e(full_variable_name, full_variable_type, 59, factor_all_num,
#                   new_ID,variable_name_num)
# su_social=feature_select_v1g(full_variable_name, full_variable_type, 61, factor_all_num,
#                   new_ID,variable_name_num)
# sum_anxiety=feature_select_v1h(full_variable_name, full_variable_type, 62, factor_all_num,
#                   new_ID,variable_name_num)
#
# #re=p_value_label(sum_anxiety, label1)
# # feature_select_v1a(full_variable_name, full_variable_type, 56, factor_all_num,
# #                    variable_name_num,new_ID)
# feature_select_vxx(full_variable_name, full_variable_type, 78, full_variable_code_list, factor_all_num,
#                    new_ID,variable_name_num)
# # feature_select_v1l(full_variable_name, full_variable_type, 64, factor_all_num,
# #                        factor_all_cat,new_ID,variable_name_num,variable_name_cat)
# RACE_ori=(data_read_and_fill_fornum(34, 'Race', new_ID))  # Race/ethnicity category derived from V1AF05 and V1AF07a-g
# for l in range (len(RACE_ori)):
#     if RACE_ori[l]>=4 and RACE_ori[l]<=6:
#         RACE_ori[l]=4.0
#     elif RACE_ori[l]==7:
#         RACE_ori[l] = 5.0
#     elif RACE_ori[l]==8:
#         RACE_ori[l] = 6.0
#
# factor_all_num.append(RACE_ori)
# variable_name_num.append('RACE_ori')
# RACE=pd.get_dummies(RACE_ori)
# factor_all_num.append(RACE[1])
# variable_name_num.append('Non-hispanic_white')
# factor_all_num.append(RACE[2])
# variable_name_num.append('Non-hispanic_black')
# factor_all_num.append(RACE[3])
# variable_name_num.append('hispanic')
# factor_all_num.append(RACE[4])
# variable_name_num.append('American_Indian_asian_hawaii_other')
# factor_all_num.append(RACE[5])
# variable_name_num.append('Other')
# factor_all_num.append(RACE[6])
# variable_name_num.append('Multiracial')
#
# #S02E05
# # CRL=(data_read_and_fill_fornum(42, 'S02E05', new_ID))  # Age
# # testcrl=p_value(CRL, GDM_relabel)
# # fmhxDM=data_read_and_fill_fornum(65, 'V2AE04', new_ID)
# # for l in range (len(fmhxDM)):
# #     if fmhxDM[l]==1.0:
# #         fmhxDM[l]=1.0
# #     else:
# #         fmhxDM[l]=0.0
# # factor_all_num.append(fmhxDM)  # BMI
# # variable_name_num.append('fmhxdiabetes')
# # factor_all_num.append(data_read_and_fill_fornum1(34, 'PctFedPoverty', new_ID))  # BMI
# # variable_name_num.append('income')
# #
# # education=(data_read_and_fill_fornum(34, 'Education', new_ID))
# # for l in range(len(education)):
# #     if education[l] == 1.0 or education[l]==2.0:
# #         education[l] = 1.0
# #     elif education[l] == 3.0 or education[l]==4.0:
# #         education[l] = 2.0
# #     elif education[l] == 5.0:
# #         education[l]=3.0
# #     else:
# #         education[l] = 4.0
# # # BMI
# # factor_all_num.append(education)
# # variable_name_num.append('education_ori')
# # EDU=pd.get_dummies(education)
# # factor_all_num.append(EDU[1])
# # variable_name_num.append('Less than HS or HS')
# # factor_all_num.append(EDU[2])
# # variable_name_num.append('Some college or Assoc/Tech degree')
# # factor_all_num.append(EDU[3])
# # variable_name_num.append('Completed college')
# # factor_all_num.append(EDU[4])
# # variable_name_num.append('Beyond college')
# #variable_name_num.append('education')
# SmokeCat1=(data_read_and_fill_fornum(34, 'SmokeCat1', new_ID))  # BMI
# SmokeCat2=(data_read_and_fill_fornum(34, 'SmokeCat2', new_ID))  # BMI
#
# SmokeCat3=(data_read_and_fill_fornum(34, 'SmokeCat3', new_ID))  # BMI
# for T in range(len(SmokeCat3)):
#     if pd.isnull(SmokeCat3[T])==True:
#         SmokeCat3[T]=0.0
#
# factor_all_num.append(SmokeCat3)
# variable_name_num.append('smoke-3-months')
#
# GravCat=(data_read_and_fill_fornum(34, 'GravCat', new_ID))
# for T in range(len(GravCat)):
#     if pd.isnull(GravCat[T])==True:
#         GravCat[T]=1.0
# # BMI
# #PctFedPoverty
# PCOS=(data_read_and_fill_fornum(78, 'VXXB01bc3_V1a', new_ID))
# for T in range(len(PCOS)):
#     if PCOS[T]!=1:
#         PCOS[T]=0
# factor_all_num.append(PCOS)
# variable_name_num.append('PCOS')
# fmhxdm=(data_read_and_fill_fornum(65, 'V2AE04', new_ID))
# for T in range(len(fmhxdm)):
#     if fmhxdm[T]!=1:
#         fmhxdm[T]=0
# factor_all_num.append(fmhxdm)
# variable_name_num.append('fmhxdm')
# #VXXB01bc3_V1a
# # alco1=(data_read_and_fill_fornum(56, 'V1AG01', new_ID))
# # alco2=(data_read_and_fill_fornum(56, 'V1AG02', new_ID))
# # for T in range(len(alco2)):
# #     if alco2[T]!=1:
# #         alco2[T]=0
# # factor_all_num.append(alco2)
# # variable_name_num.append('drink_3_months')
# # alco2a=(data_read_and_fill_fornum(56, 'V1AG02a', new_ID))
# # alco2b=(data_read_and_fill_fornum(56, 'V1AG02b', new_ID))
# # alco2c=(data_read_and_fill_fornum(56, 'V1AG02c', new_ID))
# # alco3=(data_read_and_fill_fornum(56, 'V1AG03', new_ID))
# # for T in range(len(alco3)):
# #     if alco3[T]!=1:
# #         alco3[T]=0
# # factor_all_num.append(alco3)
# # variable_name_num.append('drink_last_months')
# # alco3a=(data_read_and_fill_fornum(56, 'V1AG03a', new_ID))
# # alco3b=(data_read_and_fill_fornum(56, 'V1AG03b', new_ID))
# # alco3c=(data_read_and_fill_fornum(56, 'V1AG03c', new_ID))
# income=(data_read_and_fill_fornum(34, 'PctFedPoverty', new_ID))
# factor_all_num.append(income)
# variable_name_num.append('income')
# income=(data_read_and_fill_fornum(34, 'Ins_Govt', new_ID))
# factor_all_num.append(income)
# variable_name_num.append('Ins_Govt')
# income=(data_read_and_fill_fornum(34, 'Ins_Mil', new_ID))
# factor_all_num.append(income)
# variable_name_num.append('Ins_Mil')
# income=(data_read_and_fill_fornum(34, 'Ins_Comm', new_ID))
# factor_all_num.append(income)
# variable_name_num.append('Ins_Comm')
# income=(data_read_and_fill_fornum(34, 'Ins_Pers', new_ID))
# factor_all_num.append(income)
# variable_name_num.append('Ins_Pers')
#
# factor_all_cat = (
#                 data_read_and_fill_cat(34, 'Education', new_ID, factor_all_cat))
# variable_name_cat.append('Education')
#
#
# SBP=(data_read_and_fill_fornum1(57, 'V1BA06a1', new_ID))
# DBP=(data_read_and_fill_fornum1(57, 'V1BA06b1', new_ID))
# factor_all_num.append(SBP)
# variable_name_num.append('SBP')
# factor_all_num.append(DBP)
# variable_name_num.append('DBP')
#
# snore1=(data_read_and_fill_fornum(64, 'V1LE01', new_ID))
# for T in range(len(snore1)):
#     if snore1[T]!=1:
#         snore1[T]=0
# factor_all_num.append(snore1)
# variable_name_num.append('snore_before')
#
# #V1BA03a
# v1b_waist = (data_read_and_fill_fornum(57, 'V1BA03a', new_ID))
# #R_num.append(p_value(v1b_waist, LGA_label))
# factor_all_num.append(v1b_waist)
# variable_name_num.append('v1b_waist')
# #
# v1b_waist_over_iliac_crest = (data_read_and_fill_fornum(57, 'V1BA04a', new_ID))
# #R_num.append(p_value(v1b_waist_over_iliac_crest, LGA_label))
# factor_all_num.append(v1b_waist_over_iliac_crest)
# variable_name_num.append('v1b_waist_over_iliac_crest')
# #
# Hip_circumference = (data_read_and_fill_fornum(57, 'V1BA05a', new_ID))
# #R_num.append(p_value(Hip_circumference, LGA_label))
# factor_all_num.append(Hip_circumference)
# variable_name_num.append('Hip_circumference')
#
# Neck=(data_read_and_fill_fornum(57, 'V1BA07a', new_ID))
# #R_num.append(p_value(Neck, GDM_relabel))
# factor_all_num.append(Neck)
# variable_name_num.append('Neck')
#
# income=(data_read_and_fill_fornum(34, 'PctFedPoverty', new_ID))
# factor_all_num.append(income)
# variable_name_num.append('income')
#
#
# # snore2=(data_read_and_fill_fornum(64, 'V1LE01a', new_ID))
# # snore3=(data_read_and_fill_fornum(64, 'V1LE01b', new_ID))
# # snore4=(data_read_and_fill_fornum(64, 'V1LE01c', new_ID))
# #
# # snore5=(data_read_and_fill_fornum(64, 'V1LE06', new_ID))
# # for T in range(len(snore5)):
# #     if snore5[T]!=1:
# #         snore5[T]=0
# # factor_all_num.append(snore5)
# # variable_name_num.append('snore_recent')
# # snore6=(data_read_and_fill_fornum(64, 'V1LE06a', new_ID))
# # snore7=(data_read_and_fill_fornum(64, 'V1LE06b', new_ID))
# # snore8=(data_read_and_fill_fornum(64, 'V1LE06c', new_ID))
# #
# #
# # sleep_duration_wk=(data_read_and_fill_fornum(64, 'V1LA02a', new_ID))
# # sleep_duration_wkn=(data_read_and_fill_fornum(64, 'V1LA02b', new_ID))
# #
# # weight_plan=(data_read_and_fill_fornum(56, 'V1AD17', new_ID))
# # weight_plan1=(data_read_and_fill_fornum(56, 'V1AD17a1', new_ID))
# # weight_plan2=(data_read_and_fill_fornum(56, 'V1AD17a1_1', new_ID))
# # weight_plan3=(data_read_and_fill_fornum(56, 'V1AD17a2', new_ID))
# # weight_plan4=(data_read_and_fill_fornum(56, 'V1AD17a3', new_ID))
# # weight_plan5=(data_read_and_fill_fornum(56, 'V1AD17a3_1', new_ID))
# # for T in range(len(weight_plan5)):
# #     if pd.isnull(weight_plan5[T])==True:
# #         if pd.isnull(weight_plan2[T])==False:
# #             weight_plan5[T]=weight_plan2[T]
# #         else:
# #             weight_plan5[T] = 0.0
# # factor_all_num.append(weight_plan5)
# # variable_name_num.append('weight_plan_self')
# # weight_plan_care_provider=(data_read_and_fill_fornum(56, 'V1AD18', new_ID))
# # weight_plan_care_provider1=(data_read_and_fill_fornum(56, 'V1AD18a1', new_ID))
# # weight_plan_care_provider2=(data_read_and_fill_fornum(56, 'V1AD18a1_1', new_ID))
# # weight_plan_care_provider3=(data_read_and_fill_fornum(56, 'V1AD18a2', new_ID))
# # weight_plan_care_provider4=(data_read_and_fill_fornum(56, 'V1AD18a3', new_ID))
# # weight_plan_care_provider5=(data_read_and_fill_fornum(56, 'V1AD18a3_1', new_ID))
# # for T in range(len(weight_plan_care_provider5)):
# #     if pd.isnull(weight_plan_care_provider5[T])==True:
# #         if pd.isnull(weight_plan_care_provider2[T])==False:
# #             weight_plan_care_provider5[T]=weight_plan_care_provider2[T]
# #         else:
# #             weight_plan_care_provider5[T] = 0.0
# # factor_all_num.append(weight_plan_care_provider5)
# # variable_name_num.append('weight_plan_care_provider')
# #
# #
# # weight_before_you_got_pregnant_kg=data_read_and_fill_fornum(56, 'V1AD01a', new_ID)
# # weight_before_you_got_pregnant_lb=data_read_and_fill_fornum(56, 'V1AD01b', new_ID)
# #
#
# # LGA_90=np.nanpercentile(LGA,90)
# # LGA_10=np.nanpercentile(LGA,10)
#
#
# DP=(data_read_and_fill_fornum(26, 'CMAE04a1c', new_ID))
# DP_before_preg=(data_read_and_fill_fornum(26, 'CMAE04a1a', new_ID))
# anxiety_before_preg=(data_read_and_fill_fornum(26, 'CMAE04a2a', new_ID))
# Bipolar_disorder_before_preg=(data_read_and_fill_fornum(26, 'CMAE04a3a', new_ID))
# PTSD_before_preg=(data_read_and_fill_fornum(26, 'CMAE04a4a', new_ID))
# Schizophrenia_Schizoaffective_disorder_before_preg=(data_read_and_fill_fornum(26, 'CMAE04a5a', new_ID))
# factor_all_num.append(DP_before_preg)
# variable_name_num.append('DP_before_preg')
# factor_all_num.append(anxiety_before_preg)
# variable_name_num.append('anxiety_before_preg')
# factor_all_num.append(Bipolar_disorder_before_preg)
# variable_name_num.append('Bipolar_disorder_before_preg')
# factor_all_num.append(PTSD_before_preg)
# variable_name_num.append('PTSD_before_preg')
# factor_all_num.append(Schizophrenia_Schizoaffective_disorder_before_preg)
# variable_name_num.append('Schizophrenia_Schizoaffective_disorder_before_preg')
# (factor_all_num,variable_name_num)=feature_select_food_ahei(full_variable_name, full_variable_type, 37, full_variable_code_list, factor_all_num,new_ID,GDM_relabel,variable_name_num)
#
# #SBTiming
# factor_all_num=factor_all_num+list(HRV_data.T)
# variable_name_num=variable_name_num+list(HRV_name)
# variable_name_num = [str(item) for item in variable_name_num]
# # factor_all_num.append(stillbirth_infection_mobility)
# # variable_name_num.append('stillbirth_infection_mobility')

factor_all_num=np.load(os.path.join(DATA_ROOT, "factor_all_num_9289_Ultrasound_U3_macro.npy"))
factor_all_cat=np.load(os.path.join(DATA_ROOT, "factor_all_cat_9289_Ultrasound_U3_macro.npy"))
variable_name_num=np.load(os.path.join(DATA_ROOT, "variable_name_num_9289_Ultrasound_U3_macro.npy"))
variable_name_cat=np.load(os.path.join(DATA_ROOT, "variable_name_cat_9289_Ultrasound_U3_macro.npy"))
basic_explaination=["Maternal weight at visit 2","SBP at visit 2","DBP at visit 2","Maternal weight at visit 3",
                    "SBP at visit 3","DBP at visit 3","GA on U/S at visit 2","GA (days) on U/S date at visit 2",
                    "BPD at visit 2","HC (U/S) at visit 2","AC at visit 2","FDL at visit 2",
                    "EFW at visit 2","EFW percentile at visit 2"]

basic_explaination_U2=["Maternal weight at visit 2","SBP at visit 2","DBP at visit 2","DBP at visit 3",
                       "GA on U/S at visit 2","GA (days) on U/S date at visit 2",
                    "BPD at visit 2","HC (U/S) at visit 2","AC at visit 2","FDL at visit 2",
                    "EFW at visit 2","EFW percentile at visit 2"]

basic_explaination_u2b=["GA on U/S at visit 2 (B)","GA on U/S at visit 2 (B_days)","CL at visit 2","FL st visit 2"]
basic_explaination_u2c=["Left UA - HR at visit 2","Left UA - Angle of insonation at visit 2","Left UA - Peak Systolic Velocity at visit 2","Left UA - S/D ratio at visit 2" ,
                         "Left UARI at visit 2","Left UAPI at visit 2","Left UA - Systolic acceleration time at visit 2","Left UA - Depth of diastolic notch at visit 2",
                         "Right UA - HR at visit 2","Right UA - Angle of insonation at visit 2","Right UA - Peak Systolic Velocity at visit 2","Right UA - S/D ratio at visit 2",
                         "Right UARI at visit 2","Right UAPI at visit 2","Right UA - Systolic acceleration time at visit 2","Right UA - Depth of diastolic notch at visit 2"]

basic_explaination_u3a=["GA on U/S at visit 3","GA (days) on U/S date at visit 3",
                    "BPD at visit 3","HC at visit 3","AC at viti 3","FDL at visit 3",
                    "AFI Quadrant 1 at visit 3","AFI Quadrant 2 at visit 3","AFI Quadrant 3 at visit 3","AFI Quadrant 4 at visit 3",
                    "EFW at visit 3","EFW percentile at visit 3"]

basic_explaination_u3b=["GA on U/S at visit 3 (B)","GA on U/S at visit 3 (B_days)","CL at visit 3","FL st visit 3"]
basic_explaination_u3c=["Left UA - HR at visit 3","Left UA - Angle of insonation at visit 3","Left UA - Peak Systolic Velocity at visit 3","Left UA - S/D ratio",
                         "Left UARI at visit 3","Left UAPI at visit 3","Left UA - Systolic acceleration time at visit 3","Left UA - Depth of diastolic notch at visit 3",
                         "Right UA - HR at visit 3","Right UA - Angle of insonation at visit 3","Right UA - Peak Systolic Velocity at visit 3","Right UA - S/D ratio",
                         "Right UARI at visit 3","Right UAPI at visit 3","Right UA - Systolic acceleration time at visit 3","Right UA - Depth of diastolic notch at visit 3"]

basic_explaination_rest=['Age', 'BMI at visit 1' ,'Fetal gender', 'pre-existed DM', 'GDM', 'EPDS score at visit 1','EPDS score at visit 3',
 'RACE-original' ,'Non-hispanic white', 'Non-hispanic black', 'Hispanic',
 'American indian, asian and hawaii' ,'Other Etinicity', 'Multiracial',
 'Smoke in 3 months before visit 1', 'Smoke in 3 months before visit 2' ,'Smoke in 3 months before visit 3', 'PCOS' ,'Family history of diabetes', 'Income level',
 'Education level' ,'SBP at visit 1', 'DBP at visit 1' ,'Snore at or before visit 1','Waist Circ. at visit 1',
 'Waist over iliac crest Circ. at visit 1', 'Hip Circ. at visit 1', 'Neck Circ. at visit 1' ,'bleed at visit 1',
 'bleed at visit 2', 'bleed at visit 3','Depression before pregnancy', 'Anxiety before pregnancy',
 'Bipolar disorder before pregnancy','PTSD before pregnancy','SCZ/SAD before pregnancy']

basic_explaination_rest_GDM_1=['Age', 'BMI at visit 1' ,'Fetal gender', 'EPDS score at visit 1',
 'RACE-original' ,'Non-hispanic white', 'Non-hispanic black', 'Hispanic',
 'American indian, asian and hawaii' ,'Other Etinicity', 'Multiracial',
 'Smoke in 3 months before visit 1', 'PCOS' ,'Family history of diabetes', 'Income level',
 'Education level' ,'SBP at visit 1', 'DBP at visit 1' ,'Snore at or before visit 1','Waist Circ. at visit 1',
 'Waist over iliac crest Circ. at visit 1', 'Hip Circ. at visit 1', 'Neck Circ. at visit 1' ,'bleed at visit 1',
 'Depression before pregnancy', 'Anxiety before pregnancy',
 'Bipolar disorder before pregnancy','PTSD before pregnancy','SCZ/SAD before pregnancy']

basic_explaination_rest_GDM_2=[ 'Smoke in 3 months before visit 2' ,
 'bleed at visit 2', "Maternal weight at visit 2","SBP at visit 2","DBP at visit 2",
                    "GA on U/S at visit 2","GA (days) on U/S date at visit 2",
                    "BPD at visit 2","HC (U/S) at visit 2","AC at visit 2","FDL at visit 2",
                    "EFW at visit 2","EFW percentile at visit 2",'GDM']

basic_explaination_rest_GDM_3=['EPDS score at visit 3',
'Smoke in 3 months before visit 3', 'bleed at visit 3',"Maternal weight at visit 3",
                    "SBP at visit 3","DBP at visit 3",
                    ]
basic_explaination_rest_U2=['Age', 'BMI at visit 1' ,'Fetal gender',  'EPDS score at visit 1',
 'RACE-original' ,'Non-hispanic white', 'Non-hispanic black', 'Hispanic',
 'American indian, asian and hawaii' ,'Other Etinicity', 'Multiracial',
 'Smoke in 3 months before visit 1', 'Smoke in 3 months before visit 2' ,'PCOS' ,'Family history of diabetes', 'Income level',
 'Education level' ,'SBP at visit 1', 'DBP at visit 1' ,'Snore at or before visit 1','Waist Circ. at visit 1',
 'Waist over iliac crest Circ. at visit 1', 'Hip Circ. at visit 1', 'Neck Circ. at visit 1' ,'bleed at visit 1',
 'bleed at visit 2', 'Depression before pregnancy', 'Anxiety before pregnancy',
 'Bipolar disorder before pregnancy','PTSD before pregnancy','SCZ/SAD before pregnancy']

name_food=["HEI (total vegetables)","HEI (dark green vegetables & legumes)","HEI (total fruit)","HEI (whole fruit)",
           "HEI (whole grains)","HEI (milk)","HEI (meat and beans)","HEI (seafood & plant protein)",
           "HEI (fatty acid ratio)","HEI (sodium)","HEI (refined grains)","HEI (SoFAAS Cals)","HEI (total score)",
           "AHEI (vegetable score)","AHEI (fruit score)",
           "AHEI (whole grain score)","AHEI (sugary beverages score)",
           "AHEI (nuts and legumes score)","AHEI (red meats score)","AHEI (trans-fat percent score)",
           "AHEI (DHA & EPA intake score)","AHEI (polyunsaturated fat percent score)","AHEI (sodium intake score)",
           "AHEI (alcoholic drinks score)","AHEI (total score)"]
name_foodx=[
           "AHEI (alcoholic drinks score)","AHEI (total score)"]
basic_explaination_rest_U1=['Age', 'BMI at visit 1' ,'Fetal gender', 'EPDS score at visit 1',
 'Non-hispanic white', 'Non-hispanic black', 'Hispanic',
 'American indian, asian and hawaii' ,'Other Etinicity', 'Multiracial',
 'Smoke in 3 months before visit 1', 'PCOS' ,'Family history of diabetes', 'Income level',
 'Education level' ,'SBP at visit 1', 'DBP at visit 1' ,'Snore at or before visit 1','Waist Circ. at visit 1',
 'Waist over iliac crest Circ. at visit 1', 'Hip Circ. at visit 1', 'Neck Circ. at visit 1' ,'bleed at visit 1',
  'Depression before pregnancy', 'Anxiety before pregnancy',
 'Bipolar disorder before pregnancy','PTSD before pregnancy','SCZ/SAD before pregnancy']


# =============================================================================
# Three antenatal feature sets: U1, U2 and U3
# =============================================================================
# The original script selected only one feature set before running the model.
# Here the full feature matrix is preserved, and the same fixed 10-fold split is
# reused for U1, U2 and U3.

Name_explainx_U3 = (
    #basic_explaination
    basic_explaination_u2b
    + basic_explaination_u2c
    + basic_explaination_u3a
    + basic_explaination_u3b
    + basic_explaination_u3c
    + basic_explaination_rest_GDM_1
    + basic_explaination_rest_GDM_2
    + basic_explaination_rest_GDM_3
    + name_food
)

Name_explainx_U1 = (
    #basic_explaination
    basic_explaination_rest_GDM_1
    + name_food
)

Name_explainx_U2 = (
    #basic_explaination
    basic_explaination_u2b
    + basic_explaination_u2c
    + basic_explaination_rest_GDM_1
    + basic_explaination_rest_GDM_2
    + name_food
)

# Complete name list corresponding to the rows of factor_all_num loaded above.
# Name_explain_full = (
#     #basic_explaination
#     basic_explaination_u2b
#     + basic_explaination_u2c
#     + basic_explaination_u3a
#     + basic_explaination_u3b
#     + basic_explaination_u3c
#     + basic_explaination_rest_GDM_1
#     + basic_explaination_rest_GDM_2
#     + basic_explaination_rest_GDM_3
#     + name_food
# )
Name_explain_full=basic_explaination+basic_explaination_u2b+basic_explaination_u2c+basic_explaination_u3a+basic_explaination_u3b+basic_explaination_u3c+basic_explaination_rest+name_food

FEATURE_SETS = {
    "U1": Name_explainx_U1,
    "U2": Name_explainx_U2,
    "U3": Name_explainx_U3,
}

# -----------------------------------------------------------------------------
# Reproducibility, modelling and local-output settings
# -----------------------------------------------------------------------------
FIXED_GAMMA = 42
N_SPLITS = 10
CALIBRATION_SIZE = 0.20
MISSING_RATE_THRESHOLD = 0.20
CORRELATION_THRESHOLD = None  # No correlation-based feature selection in the manuscript pipeline
KNN_NEIGHBORS = 5
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_RANDOM_STATE = 2026

# Central model switch.
#   "EBM"  : run EBM only; EBM is also used for conformal prediction.
#   "LR"   : run elastic-net logistic regression only; LR is also used for CP.
#   "BOTH" : run both AUC models; PRIMARY_MODEL determines the CP/trajectory model.
MODEL_SELECTION = "LR"
PRIMARY_MODEL = "LR"  # Used only when MODEL_SELECTION == "BOTH".

VALID_MODEL_SELECTIONS = {"EBM", "LR", "BOTH"}
MODEL_DISPLAY_NAMES = {
    "LR": "Elastic-net LR",
    "EBM": "EBM",
}
MODEL_SELECTION = str(MODEL_SELECTION).upper()
PRIMARY_MODEL = str(PRIMARY_MODEL).upper()
if MODEL_SELECTION not in VALID_MODEL_SELECTIONS:
    raise ValueError(
        f"MODEL_SELECTION must be one of {sorted(VALID_MODEL_SELECTIONS)}, "
        f"but received {MODEL_SELECTION!r}."
    )
if PRIMARY_MODEL not in {"EBM", "LR"}:
    raise ValueError("PRIMARY_MODEL must be either 'EBM' or 'LR'.")

if MODEL_SELECTION == "BOTH":
    ACTIVE_MODEL_KEYS = ("LR", "EBM")
    PRIMARY_MODEL_KEY = PRIMARY_MODEL
else:
    ACTIVE_MODEL_KEYS = (MODEL_SELECTION,)
    PRIMARY_MODEL_KEY = MODEL_SELECTION

if PRIMARY_MODEL_KEY not in ACTIVE_MODEL_KEYS:
    raise ValueError(
        "PRIMARY_MODEL must be included among the models selected for execution."
    )
PRIMARY_MODEL_DISPLAY = MODEL_DISPLAY_NAMES[PRIMARY_MODEL_KEY]
ACTIVE_MODEL_DISPLAY_NAMES = tuple(
    MODEL_DISPLAY_NAMES[key] for key in ACTIVE_MODEL_KEYS
)

# Results are written under the shared macOS project results directory.
OUTPUT_DIR = os.path.join(
    RESULTS_ROOT,
    "FGR_LR_nestedCV_performance_metrics",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

alphas = np.linspace(0.0, 1.0, 200)

# 90% is the primary operating point; 80% is the sensitivity analysis.
TRAJECTORY_CONFIDENCE_LEVELS = (0.95, 0.90, 0.85, 0.80)
PRIMARY_TRAJECTORY_CONFIDENCE = 0.90
DELIVERED_STATE = "D"
TRAJECTORY_STATE_ORDER = ["L", "R", "H", DELIVERED_STATE]
RAW_CP_STATE_ORDER = ["L", "R", "H", "E", DELIVERED_STATE]
MIN_GROUP_SIZE_FOR_INTERPRETATION = 20

# Preserve the complete feature-by-subject matrix before visit-specific selection.
factor_all_num_full = np.asarray(factor_all_num)
label1 = np.asarray(fgr_relabel0, dtype=int)
master_index = np.asarray(fgr_index, dtype=int)
subject_ids_master = np.asarray(fgr_ID_relabel0, dtype=object)

if factor_all_num_full.ndim != 2:
    raise ValueError("factor_all_num must be a two-dimensional feature-by-subject array.")
if factor_all_num_full.shape[0] != len(Name_explain_full):
    raise ValueError(
        "The number of rows in factor_all_num does not match Name_explain_full: "
        f"{factor_all_num_full.shape[0]} versus {len(Name_explain_full)}."
    )
if len(label1) != len(master_index):
    raise ValueError(
        "The FGR label length does not match fgr_index: "
        f"{len(label1)} versus {len(master_index)}."
    )
if len(subject_ids_master) != len(label1):
    raise ValueError(
        "fgr_ID_relabel0 does not align with the FGR label vector: "
        f"{len(subject_ids_master)} versus {len(label1)}."
    )

print(
    "FGR outcome:",
    int(label1.sum()),
    "/",
    len(label1),
    "=",
    round(float(label1.mean()) * 100.0, 2),
    "%",
)

# =============================================================================
# U3 landmark safety mechanism
# =============================================================================
# U3 covers approximately 22-29 weeks. Pregnancies delivered before 30+0 weeks
# are excluded from every U3 training/calibration/test operation. Their positions
# are retained in the master arrays and their U3 state is marked D.
GA_WK_full = np.asarray(
    data_read_and_fill_fornum(40, 'GAwksA09', GDM_ID),
    dtype=float,
)
GA_DAYS_full = np.asarray(
    data_read_and_fill_fornum(40, 'GAdysA09', GDM_ID),
    dtype=float,
)
GA_WK_aligned = GA_WK_full[master_index]
GA_DAYS_aligned = GA_DAYS_full[master_index]
GA_total_weeks_aligned = GA_WK_aligned.astype(float).copy()
valid_ga_days_aligned = (
    np.isfinite(GA_DAYS_aligned)
    & (GA_DAYS_aligned >= 0)
    & (GA_DAYS_aligned <= 6)
    & (GA_DAYS_aligned == np.floor(GA_DAYS_aligned))
)
GA_total_weeks_aligned[valid_ga_days_aligned] = (
    GA_WK_aligned[valid_ga_days_aligned]
    + GA_DAYS_aligned[valid_ga_days_aligned] / 7.0
)

U3_LANDMARK_START_WEEKS = 30.0
U3_delivered_before_landmark = (
    np.isfinite(GA_total_weeks_aligned)
    & (GA_total_weeks_aligned < U3_LANDMARK_START_WEEKS)
)
# Missing delivery GA is retained in the risk set because absence of GA alone
# does not prove that delivery occurred before U3.
U3_landmark_eligible = ~U3_delivered_before_landmark

print(
    "\nU3 landmark exclusion (delivery before 30+0 weeks / completed GA <=29):",
    int(U3_delivered_before_landmark.sum()),
    "/",
    len(label1),
)
print(
    "FGR events among pregnancies delivered before U3:",
    int(label1[U3_delivered_before_landmark].sum()),
    "/",
    int(U3_delivered_before_landmark.sum()),
)


def build_raw_feature_dataframe(feature_names):
    """Select a visit-specific feature set and return subjects x features."""
    missing_names = [name for name in feature_names if name not in Name_explain_full]
    if missing_names:
        raise ValueError(
            "The following requested features are absent from Name_explain_full: "
            + ", ".join(missing_names)
        )

    positions = [Name_explain_full.index(name) for name in feature_names]
    if len(set(positions)) != len(positions):
        duplicated = [
            name for index, name in enumerate(feature_names)
            if positions.count(positions[index]) > 1
        ]
        raise ValueError(
            "Duplicated feature positions were found in the requested feature set: "
            + ", ".join(sorted(set(duplicated)))
        )

    selected = factor_all_num_full[positions, :]
    selected = selected[:, master_index].T
    return pd.DataFrame(selected, columns=feature_names, dtype=float)


def fit_preprocessor(X_train):
    """Fit manuscript-specified preprocessing using training data only.

    Existing FGR feature construction yields the analysis matrix used by the source
    pipeline. Continuous/matrix-valued predictors are KNN-imputed (k=5), then
    scaled using training-set 5th/95th percentiles and clipped to [0, 1]. Variables
    with >20% missingness are excluded. No correlation or other feature-selection
    step is applied here.
    """
    missing_rate = X_train.isna().mean(axis=0)
    retained_columns = missing_rate[missing_rate <= MISSING_RATE_THRESHOLD].index.tolist()
    if not retained_columns:
        raise ValueError("No feature remained after the >20% missingness filter.")

    X_retained = X_train.loc[:, retained_columns]
    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    X_imputed = imputer.fit_transform(X_retained)

    lower = np.nanpercentile(X_imputed, 5, axis=0)
    upper = np.nanpercentile(X_imputed, 95, axis=0)
    scale_range = upper - lower
    variable_mask = scale_range > 0

    X_scaled = np.zeros_like(X_imputed, dtype=float)
    X_scaled[:, variable_mask] = np.clip(
        (X_imputed[:, variable_mask] - lower[variable_mask]) / scale_range[variable_mask],
        0.0, 1.0,
    )
    # Constant columns map to 0.
    X_scaled_df = pd.DataFrame(X_scaled, columns=retained_columns, index=X_train.index)
    state = {
        "retained_columns": retained_columns,
        "imputer": imputer,
        "lower": lower,
        "scale_range": scale_range,
        "variable_mask": variable_mask,
        "final_columns": retained_columns,
    }
    return X_scaled_df, state

def transform_preprocessor(X, state):
    """Apply a fitted preprocessing state unchanged to validation/test data."""
    X_retained = X.loc[:, state["retained_columns"]]
    X_imputed = state["imputer"].transform(X_retained)
    variable_mask = state["variable_mask"]
    scale_range = state["scale_range"]
    lower = state["lower"]
    X_scaled = np.zeros_like(X_imputed, dtype=float)
    X_scaled[:, variable_mask] = np.clip(
        (X_imputed[:, variable_mask] - lower[variable_mask]) / scale_range[variable_mask],
        0.0, 1.0,
    )
    return pd.DataFrame(X_scaled, columns=state["retained_columns"], index=X.index)

def calculate_sqrt_positive_weight(y):
    """Return sqrt(n_negative / n_positive) for the current training data."""
    y = np.asarray(y, dtype=int)
    positive_n = int((y == 1).sum())
    negative_n = int((y == 0).sum())
    if positive_n == 0 or negative_n == 0:
        raise ValueError("Both outcome classes are required to calculate class weights.")
    return float(np.sqrt(negative_n / positive_n))


def make_binary_sample_weight(y, positive_weight):
    y = np.asarray(y, dtype=int)
    sample_weight = np.ones(len(y), dtype=float)
    sample_weight[y == 1] = float(positive_weight)
    return sample_weight


def make_logistic_model(positive_weight):
    """Elastic-net logistic-regression baseline."""
    return LogisticRegression(
        C=0.1,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.3,
        class_weight={0: 1.0, 1: float(positive_weight)},
        max_iter=50000,
        random_state=FIXED_GAMMA,
    )


def make_ebm_model():
    """Explainable Boosting Machine used for the primary AUC and CP analyses."""
    if ExplainableBoostingClassifier is None:
        raise ImportError(
            "InterpretML is required. Install it in the active environment with: "
            "pip install interpret"
        )
    return ExplainableBoostingClassifier(
        interactions=10,
        max_leaves=3,
        learning_rate=0.015,
        max_rounds=5000,
        early_stopping_rounds=100,
        outer_bags=8,
        min_samples_leaf=5,
        random_state=FIXED_GAMMA,
        n_jobs=-1,
    )


def make_selected_model(model_key, positive_weight):
    """Create the requested model using a shared model-key interface."""
    model_key = str(model_key).upper()
    if model_key == "LR":
        return make_logistic_model(positive_weight)
    if model_key == "EBM":
        return make_ebm_model()
    raise ValueError(f"Unsupported model key: {model_key!r}.")


def fit_selected_model(model, model_key, X, y, positive_weight):
    """Fit LR or EBM while preserving each estimator's weighting interface."""
    model_key = str(model_key).upper()
    if model_key == "LR":
        model.fit(X, y)
    elif model_key == "EBM":
        model.fit(
            X,
            y,
            sample_weight=make_binary_sample_weight(y, positive_weight),
        )
    else:
        raise ValueError(f"Unsupported model key: {model_key!r}.")
    return model


def bootstrap_auc_ci(
    y_true,
    y_score,
    n_bootstrap=BOOTSTRAP_ITERATIONS,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    """Percentile 95% CI from participant-level bootstrap of OOF scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        sample_index = rng.integers(0, n_samples, size=n_samples)
        y_boot = y_true[sample_index]
        if np.unique(y_boot).size < 2:
            continue
        bootstrap_aucs.append(
            metrics.roc_auc_score(y_boot, y_score[sample_index])
        )
    if not bootstrap_aucs:
        raise RuntimeError("No valid bootstrap sample contained both outcome classes.")
    lower_ci, upper_ci = np.percentile(bootstrap_aucs, [2.5, 97.5])
    return float(lower_ci), float(upper_ci), np.asarray(bootstrap_aucs)




def finite_sample_conformal_quantile(scores, alpha):
    """Return the canonical finite-sample split-conformal quantile.

    For n calibration scores and significance level alpha, the threshold is
    the k-th order statistic with k = ceil((n + 1) * (1 - alpha)).
    The boundary cases alpha=0 and alpha=1 are represented by +inf and -inf,
    respectively, so the helper is also well-defined for diagnostic curves.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return np.nan

    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    if k <= 0:
        return -np.inf
    if k > n:
        return np.inf

    return float(np.partition(scores, k - 1)[k - 1])

def calculate_mondrian_cp_rates(model, X_calib, y_calib, X_test, y_test):
    """Calculate CP efficiency/error curves using the selected uncalibrated model."""
    y_calib = np.asarray(y_calib, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    calib_proba = model.predict_proba(X_calib)
    calib_class0 = y_calib == 0
    calib_class1 = y_calib == 1
    if not np.any(calib_class0) or not np.any(calib_class1):
        raise ValueError("The conformal calibration set must contain both classes.")

    scores_calib0 = 1.0 - calib_proba[calib_class0, 0]
    scores_calib1 = 1.0 - calib_proba[calib_class1, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    single_correct_rates = []
    single_error_rates = []
    multi_rates = []
    empty_rates = []
    positive_single_correct_rates = []
    negative_single_as_positive_rates = []

    for alpha in alphas:
        q0 = finite_sample_conformal_quantile(scores_calib0, alpha)
        q1 = finite_sample_conformal_quantile(scores_calib1, alpha)
        include0 = proba_test <= q0
        include1 = (1.0 - proba_test) <= q1

        singleton0 = include0 & ~include1
        singleton1 = ~include0 & include1
        multiple = include0 & include1
        empty = ~include0 & ~include1
        single_correct = (singleton0 & (y_test == 0)) | (singleton1 & (y_test == 1))
        single_error = (singleton0 & (y_test == 1)) | (singleton1 & (y_test == 0))

        single_correct_rates.append(single_correct.mean())
        single_error_rates.append(single_error.mean())
        multi_rates.append(multiple.mean())
        empty_rates.append(empty.mean())

        positive_mask = y_test == 1
        negative_mask = y_test == 0
        positive_single_correct_rates.append(
            float(singleton1[positive_mask].mean()) if positive_mask.any() else np.nan
        )
        negative_single_as_positive_rates.append(
            float(singleton1[negative_mask].mean()) if negative_mask.any() else np.nan
        )

    return {
        "single_correct": np.asarray(single_correct_rates),
        "single_error": np.asarray(single_error_rates),
        "multiple": np.asarray(multi_rates),
        "empty": np.asarray(empty_rates),
        "positive_accuracy": np.asarray(positive_single_correct_rates),
        "negative_error": np.asarray(negative_single_as_positive_rates),
    }


def calculate_mondrian_cp_states_at_confidences(
    model,
    X_calib,
    y_calib,
    X_test,
    confidence_levels=TRAJECTORY_CONFIDENCE_LEVELS,
):
    """Return CP states plus a calibration-only distribution-matched score comparator.

    For each outer fold, visit, and confidence level, the comparator uses the same
    CP-model probability and chooses two probability thresholds only from the
    conformal-calibration subset. The target L/R/H proportions are the empirical
    CP analysis-state proportions in that calibration subset. Thresholds are then
    applied unchanged to the held-out outer-test subjects.
    """
    y_calib = np.asarray(y_calib, dtype=int)
    calib_proba = model.predict_proba(X_calib)
    calib_proba_1 = np.asarray(calib_proba[:, 1], dtype=float)
    calib_class0 = y_calib == 0
    calib_class1 = y_calib == 1
    if not np.any(calib_class0) or not np.any(calib_class1):
        raise ValueError("The conformal calibration set must contain both classes.")

    scores_calib0 = 1.0 - calib_proba[calib_class0, 0]
    scores_calib1 = 1.0 - calib_proba[calib_class1, 1]
    test_proba_1 = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)

    output = {
        "probability_1": test_proba_1,
        "calibration_probability_1": calib_proba_1,
        "by_confidence": {},
    }
    for confidence in confidence_levels:
        confidence = float(confidence)
        if not 0.0 < confidence < 1.0:
            raise ValueError("Trajectory confidence must lie between 0 and 1.")
        alpha = 1.0 - confidence
        q0 = finite_sample_conformal_quantile(scores_calib0, alpha)
        q1 = finite_sample_conformal_quantile(scores_calib1, alpha)

        include0 = test_proba_1 <= q0
        include1 = (1.0 - test_proba_1) <= q1
        raw_state = np.full(len(test_proba_1), "", dtype="<U1")
        raw_state[include0 & ~include1] = "L"
        raw_state[~include0 & include1] = "H"
        raw_state[include0 & include1] = "R"
        raw_state[~include0 & ~include1] = "E"
        state = raw_state.copy()
        state[state == "E"] = "R"

        # Recreate CP analysis states on calibration subjects using q0/q1.
        calib_include0 = calib_proba_1 <= q0
        calib_include1 = (1.0 - calib_proba_1) <= q1
        calib_raw_state = np.full(len(calib_proba_1), "", dtype="<U1")
        calib_raw_state[calib_include0 & ~calib_include1] = "L"
        calib_raw_state[~calib_include0 & calib_include1] = "H"
        calib_raw_state[calib_include0 & calib_include1] = "R"
        calib_raw_state[~calib_include0 & ~calib_include1] = "E"
        calib_state = calib_raw_state.copy()
        calib_state[calib_state == "E"] = "R"

        pi_l = float(np.mean(calib_state == "L"))
        pi_r = float(np.mean(calib_state == "R"))
        pi_h = float(np.mean(calib_state == "H"))
        if not np.isclose(pi_l + pi_r + pi_h, 1.0):
            raise RuntimeError("Calibration CP state proportions do not sum to one.")

        score_low_threshold = float(np.quantile(calib_proba_1, pi_l, method="higher"))
        score_high_threshold = float(np.quantile(calib_proba_1, 1.0 - pi_h, method="higher"))
        if score_low_threshold >= score_high_threshold:
            raise RuntimeError(
                "Matched-score thresholds overlap; cannot construct ordered L/R/H states."
            )

        score_state = np.full(len(test_proba_1), "R", dtype="<U1")
        score_state[test_proba_1 <= score_low_threshold] = "L"
        score_state[test_proba_1 >= score_high_threshold] = "H"

        calib_score_state = np.full(len(calib_proba_1), "R", dtype="<U1")
        calib_score_state[calib_proba_1 <= score_low_threshold] = "L"
        calib_score_state[calib_proba_1 >= score_high_threshold] = "H"

        output["by_confidence"][confidence] = {
            "alpha": alpha,
            "state": state,
            "raw_state": raw_state,
            "score_state": score_state,
            "quantile_class0": float(q0),
            "quantile_class1": float(q1),
            "score_low_threshold": score_low_threshold,
            "score_high_threshold": score_high_threshold,
            "calibration_cp_L_prop": pi_l,
            "calibration_cp_R_prop": pi_r,
            "calibration_cp_H_prop": pi_h,
            "calibration_score_L_prop": float(np.mean(calib_score_state == "L")),
            "calibration_score_R_prop": float(np.mean(calib_score_state == "R")),
            "calibration_score_H_prop": float(np.mean(calib_score_state == "H")),
        }
    return output


def _make_stratified_master_splits(analysis_mask):
    """Create visit-specific folds and map them back to the master cohort."""
    analysis_mask = np.asarray(analysis_mask, dtype=bool)
    eligible_indices = np.flatnonzero(analysis_mask)
    y_eligible = label1[eligible_indices]
    class_counts = np.bincount(y_eligible, minlength=2)
    if np.any(class_counts < N_SPLITS):
        raise ValueError(
            "Each class must contain at least N_SPLITS subjects. "
            f"Counts={class_counts.tolist()}, N_SPLITS={N_SPLITS}."
        )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=FIXED_GAMMA,
    )
    mapped_splits = []
    for local_train, local_test in splitter.split(
        np.zeros(len(eligible_indices)),
        y_eligible,
    ):
        mapped_splits.append(
            (eligible_indices[local_train], eligible_indices[local_test])
        )
    return mapped_splits


def run_visit_model(visit_name, feature_names, cv_splits, analysis_mask):
    """Run selected AUC model(s) and selected-model Mondrian CP."""
    X_raw = build_raw_feature_dataframe(feature_names)
    analysis_mask = np.asarray(analysis_mask, dtype=bool)
    excluded_mask = ~analysis_mask
    if len(X_raw) != len(label1):
        raise ValueError(f"{visit_name}: feature rows and labels are misaligned.")
    if analysis_mask.shape != (len(label1),):
        raise ValueError(f"{visit_name}: invalid analysis-mask shape.")

    model_names = list(ACTIVE_MODEL_DISPLAY_NAMES)
    oof_predictions_by_model = {
        name: np.full(len(label1), np.nan, dtype=float) for name in model_names
    }
    oof_cp_probability = np.full(len(label1), np.nan, dtype=float)
    oof_fold_id = np.full(len(label1), np.nan, dtype=float)
    oof_score_state = {
        float(c): np.full(len(label1), "", dtype="<U1")
        for c in TRAJECTORY_CONFIDENCE_LEVELS
    }
    oof_cp_state = {
        float(c): np.full(len(label1), "", dtype="<U1")
        for c in TRAJECTORY_CONFIDENCE_LEVELS
    }
    oof_cp_raw_state = {
        float(c): np.full(len(label1), "", dtype="<U1")
        for c in TRAJECTORY_CONFIDENCE_LEVELS
    }
    if np.any(excluded_mask):
        for confidence in TRAJECTORY_CONFIDENCE_LEVELS:
            confidence = float(confidence)
            oof_cp_state[confidence][excluded_mask] = DELIVERED_STATE
            oof_cp_raw_state[confidence][excluded_mask] = DELIVERED_STATE
            oof_score_state[confidence][excluded_mask] = DELIVERED_STATE

    cp_single_correct = np.zeros(len(alphas))
    cp_single_error = np.zeros(len(alphas))
    cp_multiple = np.zeros(len(alphas))
    cp_empty = np.zeros(len(alphas))
    cp_positive_accuracy = np.zeros(len(alphas))
    cp_negative_error = np.zeros(len(alphas))
    features_after_preprocessing = []
    fold_positive_weights = []
    matched_score_threshold_rows = []

    for fold_number, (train_index, test_index) in enumerate(cv_splits, start=1):
        train_index = np.asarray(train_index, dtype=int)
        test_index = np.asarray(test_index, dtype=int)
        oof_fold_id[test_index] = float(fold_number)
        if not np.all(analysis_mask[train_index]) or not np.all(analysis_mask[test_index]):
            raise RuntimeError(
                f"{visit_name}: a CV fold contains a subject outside the risk set."
            )
        print(
            f"{visit_name}: Processing Fold {fold_number}/{N_SPLITS} "
            f"(train={len(train_index)}, test={len(test_index)}; "
            f"models={','.join(ACTIVE_MODEL_KEYS)}; CP={PRIMARY_MODEL_KEY})"
        )

        X_train_raw = X_raw.iloc[train_index]
        X_test_raw = X_raw.iloc[test_index]
        y_train = label1[train_index]
        y_test = label1[test_index]

        # AUC branch: train the selected model(s) on the complete outer-training fold.
        X_train_auc, auc_preprocessor = fit_preprocessor(X_train_raw)
        X_test_auc = transform_preprocessor(X_test_raw, auc_preprocessor)
        features_after_preprocessing.append(X_train_auc.shape[1])
        positive_weight_auc = calculate_sqrt_positive_weight(y_train)
        fold_positive_weights.append(positive_weight_auc)

        for model_key in ACTIVE_MODEL_KEYS:
            model = make_selected_model(model_key, positive_weight_auc)
            model = fit_selected_model(
                model=model,
                model_key=model_key,
                X=X_train_auc,
                y=y_train,
                positive_weight=positive_weight_auc,
            )
            model_name = MODEL_DISPLAY_NAMES[model_key]
            oof_predictions_by_model[model_name][test_index] = (
                model.predict_proba(X_test_auc)[:, 1]
            )

        # CP branch: proper training + independent conformal calibration.
        proper_local_index, calibration_local_index = train_test_split(
            np.arange(len(train_index)),
            test_size=CALIBRATION_SIZE,
            stratify=y_train,
            random_state=FIXED_GAMMA,
        )
        X_proper_raw = X_train_raw.iloc[proper_local_index]
        X_calib_raw = X_train_raw.iloc[calibration_local_index]
        y_proper = y_train[proper_local_index]
        y_calib = y_train[calibration_local_index]

        X_proper, cp_preprocessor = fit_preprocessor(X_proper_raw)
        X_calib = transform_preprocessor(X_calib_raw, cp_preprocessor)
        X_test_cp = transform_preprocessor(X_test_raw, cp_preprocessor)

        positive_weight_cp = calculate_sqrt_positive_weight(y_proper)
        cp_model = make_selected_model(PRIMARY_MODEL_KEY, positive_weight_cp)
        cp_model = fit_selected_model(
            model=cp_model,
            model_key=PRIMARY_MODEL_KEY,
            X=X_proper,
            y=y_proper,
            positive_weight=positive_weight_cp,
        )
        # No isotonic fitting. y_calib is used only for conformal calibration.
        fold_cp = calculate_mondrian_cp_rates(
            model=cp_model,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test_cp,
            y_test=y_test,
        )
        fold_trajectory = calculate_mondrian_cp_states_at_confidences(
            model=cp_model,
            X_calib=X_calib,
            y_calib=y_calib,
            X_test=X_test_cp,
        )

        oof_cp_probability[test_index] = fold_trajectory["probability_1"]
        for confidence in TRAJECTORY_CONFIDENCE_LEVELS:
            confidence = float(confidence)
            oof_cp_state[confidence][test_index] = (
                fold_trajectory["by_confidence"][confidence]["state"]
            )
            oof_cp_raw_state[confidence][test_index] = (
                fold_trajectory["by_confidence"][confidence]["raw_state"]
            )
            oof_score_state[confidence][test_index] = (
                fold_trajectory["by_confidence"][confidence]["score_state"]
            )
            audit = fold_trajectory["by_confidence"][confidence]
            matched_score_threshold_rows.append(
                {
                    "Visit": visit_name,
                    "Fold": int(fold_number),
                    "Confidence": int(round(confidence * 100)),
                    "Calibration N": int(len(y_calib)),
                    "CP L proportion in calibration": audit["calibration_cp_L_prop"],
                    "CP R proportion in calibration": audit["calibration_cp_R_prop"],
                    "CP H proportion in calibration": audit["calibration_cp_H_prop"],
                    "Score L proportion in calibration": audit["calibration_score_L_prop"],
                    "Score R proportion in calibration": audit["calibration_score_R_prop"],
                    "Score H proportion in calibration": audit["calibration_score_H_prop"],
                    "Score low threshold": audit["score_low_threshold"],
                    "Score high threshold": audit["score_high_threshold"],
                    "CP class-0 quantile": audit["quantile_class0"],
                    "CP class-1 quantile": audit["quantile_class1"],
                }
            )

        cp_single_correct += fold_cp["single_correct"]
        cp_single_error += fold_cp["single_error"]
        cp_multiple += fold_cp["multiple"]
        cp_empty += fold_cp["empty"]
        cp_positive_accuracy += fold_cp["positive_accuracy"]
        cp_negative_error += fold_cp["negative_error"]

    for model_name, predictions in oof_predictions_by_model.items():
        if np.isnan(predictions[analysis_mask]).any():
            raise RuntimeError(f"{visit_name}: missing {model_name} OOF predictions.")
        if np.any(~np.isnan(predictions[excluded_mask])):
            raise RuntimeError(f"{visit_name}: excluded subjects received predictions.")
    if np.isnan(oof_cp_probability[analysis_mask]).any():
        raise RuntimeError(f"{visit_name}: missing CP probabilities in the risk set.")
    if np.any(~np.isnan(oof_cp_probability[excluded_mask])):
        raise RuntimeError(f"{visit_name}: excluded subjects received CP probabilities.")
    for confidence in TRAJECTORY_CONFIDENCE_LEVELS:
        confidence = float(confidence)
        if np.any(oof_cp_state[confidence][analysis_mask] == ""):
            raise RuntimeError(f"{visit_name}: missing CP states in the risk set.")
        if np.any(excluded_mask) and np.any(
            oof_cp_state[confidence][excluded_mask] != DELIVERED_STATE
        ):
            raise RuntimeError(f"{visit_name}: excluded subjects were not marked D.")

    if np.isnan(oof_fold_id[analysis_mask]).any():
        raise RuntimeError(f"{visit_name}: missing outer-fold IDs in the risk set.")
    if np.any(~np.isnan(oof_fold_id[excluded_mask])):
        raise RuntimeError(f"{visit_name}: excluded subjects received outer-fold IDs.")
    for confidence in TRAJECTORY_CONFIDENCE_LEVELS:
        confidence = float(confidence)
        if np.any(oof_score_state[confidence][analysis_mask] == ""):
            raise RuntimeError(f"{visit_name}: missing matched-score states in the risk set.")
        if np.any(excluded_mask) and np.any(
            oof_score_state[confidence][excluded_mask] != DELIVERED_STATE
        ):
            raise RuntimeError(
                f"{visit_name}: excluded subjects were not marked as D in score states."
            )

    y_eval = label1[analysis_mask]
    model_auc_results = {}
    for model_name, full_predictions in oof_predictions_by_model.items():
        predictions = full_predictions[analysis_mask]
        fpr, tpr, threshold = metrics.roc_curve(y_eval, predictions)
        auc_value = metrics.auc(fpr, tpr)
        ci_lower, ci_upper, bootstrap_aucs = bootstrap_auc_ci(y_eval, predictions)
        model_auc_results[model_name] = {
            "roc_auc": float(auc_value),
            "auc_ci_lower": ci_lower,
            "auc_ci_upper": ci_upper,
            "oof_prediction": full_predictions,
            "evaluation_prediction": predictions,
            "evaluation_label": y_eval,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": threshold,
            "bootstrap_aucs": bootstrap_aucs,
        }
        print(
            f"{visit_name} {model_name} FGR AUC = {auc_value:.4f} "
            f"(bootstrap 95% CI {ci_lower:.4f}-{ci_upper:.4f}; "
            f"risk-set N={analysis_mask.sum()})"
        )

    primary_auc_result = model_auc_results[PRIMARY_MODEL_DISPLAY]
    fold_count = len(cv_splits)
    return {
        "visit": visit_name,
        "analysis_mask": analysis_mask.copy(),
        "excluded_before_visit_mask": excluded_mask.copy(),
        "n_subjects": int(analysis_mask.sum()),
        "n_excluded_before_visit": int(excluded_mask.sum()),
        "n_positive": int(label1[analysis_mask].sum()),
        "n_requested_features": len(feature_names),
        "n_features_after_preprocessing_mean": float(
            np.mean(features_after_preprocessing)
        ),
        "mean_sqrt_positive_weight": float(np.mean(fold_positive_weights)),
        "model_auc_results": model_auc_results,
        "oof_predictions_by_model": oof_predictions_by_model,
        "primary_model": PRIMARY_MODEL_DISPLAY,
        "conformal_model": f"{PRIMARY_MODEL_DISPLAY} (no isotonic)",
        "roc_auc_local": primary_auc_result["roc_auc"],
        "auc_ci_lower": primary_auc_result["auc_ci_lower"],
        "auc_ci_upper": primary_auc_result["auc_ci_upper"],
        "oof_prediction": primary_auc_result["oof_prediction"],
        "oof_cp_probability": oof_cp_probability,
        "oof_fold_id": oof_fold_id,
        "oof_score_state": oof_score_state,
        "matched_score_thresholds": matched_score_threshold_rows,
        "oof_cp_state": oof_cp_state,
        "oof_cp_raw_state": oof_cp_raw_state,
        "fpr_local": primary_auc_result["fpr"],
        "tpr_local": primary_auc_result["tpr"],
        "threshold_local": primary_auc_result["threshold"],
        "bootstrap_aucs": primary_auc_result["bootstrap_aucs"],
        "cp_single_correct": cp_single_correct / fold_count,
        "cp_single_error": cp_single_error / fold_count,
        "cp_multiple": cp_multiple / fold_count,
        "cp_empty": cp_empty / fold_count,
        "cp_positive_accuracy": cp_positive_accuracy / fold_count,
        "cp_negative_error": cp_negative_error / fold_count,
    }

def _safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def _wilson_interval(event_n, total_n, z=1.959963984540054):
    if total_n <= 0:
        return np.nan, np.nan
    p = event_n / total_n
    denominator = 1.0 + (z ** 2) / total_n
    centre = (p + (z ** 2) / (2.0 * total_n)) / denominator
    half_width = (
        z
        * np.sqrt(
            p * (1.0 - p) / total_n
            + (z ** 2) / (4.0 * total_n ** 2)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def classify_trajectory_group(states):
    s1, s2, s3 = states
    if DELIVERED_STATE in states:
        return "Delivered before U3"
    trajectory = f"{s1}-{s2}-{s3}"
    if trajectory == "L-L-L":
        return "Stable low"
    if trajectory == "H-H-H":
        return "Persistent high"
    if trajectory == "R-R-R":
        return "Persistent review"
    if s3 == "H" and s1 != "H":
        return "Emerging high"
    if s3 == "L" and "R" in (s1, s2) and "H" not in (s1, s2):
        return "Resolved to low"
    if s1 == "H" and s3 != "H":
        return "High de-escalated"
    if s3 == "R":
        return "Review at Visit 3"
    return "Other/reversal"


def build_trajectory_dataframe(visit_results, confidence):
    """Link U1/U2/U3 decisions on the unchanged master index."""
    confidence = float(confidence)
    trajectory_df = pd.DataFrame(
        {
            "subject_id": subject_ids_master,
            "FGR": label1.astype(int),
            "GA_week": GA_WK_aligned,
            "GA_day": GA_DAYS_aligned,
            "GA_total_weeks": GA_total_weeks_aligned,
            "U3_landmark_eligible": U3_landmark_eligible,
            "delivered_before_U3": U3_delivered_before_landmark,
        }
    )
    for visit_name in ["U1", "U2", "U3"]:
        result = visit_results[visit_name]
        trajectory_df[f"{visit_name}_state"] = result["oof_cp_state"][confidence]
        trajectory_df[f"{visit_name}_raw_state"] = result[
            "oof_cp_raw_state"
        ][confidence]
        trajectory_df[f"{visit_name}_probability"] = result["oof_cp_probability"]

    expected_d = trajectory_df["delivered_before_U3"].to_numpy()
    observed_d = trajectory_df["U3_state"].to_numpy() == DELIVERED_STATE
    if not np.array_equal(expected_d, observed_d):
        raise RuntimeError("U3 D markers do not exactly match the exclusion mask.")

    trajectory_df["trajectory"] = (
        trajectory_df["U1_state"]
        + "-"
        + trajectory_df["U2_state"]
        + "-"
        + trajectory_df["U3_state"]
    )
    trajectory_df["U1_U2_trajectory"] = (
        trajectory_df["U1_state"] + "-" + trajectory_df["U2_state"]
    )

    eligible = trajectory_df["U3_landmark_eligible"].to_numpy()
    changed_any = np.full(len(trajectory_df), np.nan, dtype=float)
    number_of_changes = np.full(len(trajectory_df), np.nan, dtype=float)
    changed_any[eligible] = (
        ~(
            (trajectory_df.loc[eligible, "U1_state"].to_numpy()
             == trajectory_df.loc[eligible, "U2_state"].to_numpy())
            &
            (trajectory_df.loc[eligible, "U2_state"].to_numpy()
             == trajectory_df.loc[eligible, "U3_state"].to_numpy())
        )
    ).astype(float)
    number_of_changes[eligible] = (
        (trajectory_df.loc[eligible, "U1_state"].to_numpy()
         != trajectory_df.loc[eligible, "U2_state"].to_numpy()).astype(int)
        +
        (trajectory_df.loc[eligible, "U2_state"].to_numpy()
         != trajectory_df.loc[eligible, "U3_state"].to_numpy()).astype(int)
    )
    trajectory_df["changed_any"] = changed_any
    trajectory_df["number_of_changes"] = number_of_changes
    trajectory_df["trajectory_group"] = trajectory_df.apply(
        lambda row: classify_trajectory_group(
            (row["U1_state"], row["U2_state"], row["U3_state"])
        ),
        axis=1,
    )
    return trajectory_df


def summarize_state_prevalence(trajectory_df):
    """Report D explicitly and use an at-risk denominator for L/R/H."""
    rows = []
    subsets = {
        "All samples": pd.Series(True, index=trajectory_df.index),
        "FGR-positive only": trajectory_df["FGR"] == 1,
        "FGR-negative only": trajectory_df["FGR"] == 0,
    }
    for subset_name, subset_mask in subsets.items():
        subset_n = int(subset_mask.sum())
        for visit_name in ["U1", "U2", "U3"]:
            states = trajectory_df.loc[subset_mask, f"{visit_name}_state"]
            raw_states = trajectory_df.loc[subset_mask, f"{visit_name}_raw_state"]
            at_risk_n = int((states != DELIVERED_STATE).sum())
            for state in TRAJECTORY_STATE_ORDER:
                n_state = int((states == state).sum())
                rows.append(
                    {
                        "Subset": subset_name,
                        "Visit": visit_name,
                        "State": state,
                        "N": n_state,
                        "Subset N": subset_n,
                        "At-risk N": at_risk_n,
                        "Percent of subset": 100.0 * _safe_rate(n_state, subset_n),
                        "Percent among at-risk": (
                            100.0 * _safe_rate(n_state, at_risk_n)
                            if state != DELIVERED_STATE else np.nan
                        ),
                        "Empty raw sets": (
                            int((raw_states == "E").sum()) if state == "R" else 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_empirical_coverage(trajectory_df, confidence):
    """Calculate pooled OOF marginal and class-conditional empirical coverage."""
    rows = []
    nominal = float(confidence)
    for visit_name in ["U1", "U2", "U3"]:
        at_risk = trajectory_df[f"{visit_name}_state"] != DELIVERED_STATE
        raw_state = trajectory_df.loc[at_risk, f"{visit_name}_raw_state"]
        y = trajectory_df.loc[at_risk, "FGR"].astype(int)
        covered = (
            ((y == 0) & raw_state.isin(["L", "R"]))
            | ((y == 1) & raw_state.isin(["H", "R"]))
        )
        for subset_name, subset_mask in {
            "Overall": pd.Series(True, index=y.index),
            "Class 0": y == 0,
            "Class 1": y == 1,
        }.items():
            n = int(subset_mask.sum())
            covered_n = int((covered & subset_mask).sum())
            rate = _safe_rate(covered_n, n)
            ci_lower, ci_upper = _wilson_interval(covered_n, n)
            rows.append(
                {
                    "Confidence": nominal,
                    "Visit": visit_name,
                    "Subset": subset_name,
                    "N": n,
                    "Covered": covered_n,
                    "Empirical coverage": rate,
                    "95% CI lower": ci_lower,
                    "95% CI upper": ci_upper,
                    "Difference from nominal": rate - nominal,
                }
            )
    return pd.DataFrame(rows)


def summarize_early_delivery(trajectory_df):
    early = trajectory_df.loc[trajectory_df["delivered_before_U3"]].copy()
    if early.empty:
        return pd.DataFrame(), pd.DataFrame()
    outcome_summary = pd.DataFrame(
        {
            "Group": ["Delivered before U3"],
            "N": [len(early)],
            "FGR events": [int(early["FGR"].sum())],
            "FGR event rate": [float(early["FGR"].mean())],
        }
    )
    path_summary = (
        early.groupby(["FGR", "U1_U2_trajectory"], dropna=False)
        .size()
        .reset_index(name="N")
        .sort_values(["FGR", "N"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return outcome_summary, path_summary


def summarize_exact_trajectories(trajectory_df, positive_only=False):
    analysis_df = trajectory_df.loc[trajectory_df["U3_landmark_eligible"]]
    if positive_only:
        analysis_df = analysis_df.loc[analysis_df["FGR"] == 1]
    reference_df = trajectory_df.loc[trajectory_df["U3_landmark_eligible"]]
    reference_rate = float(reference_df["FGR"].mean())
    total_n = len(analysis_df)
    rows = []
    for trajectory, group in analysis_df.groupby("trajectory", sort=False):
        n_group = len(group)
        event_n = int(group["FGR"].sum())
        event_rate = _safe_rate(event_n, n_group)
        ci_lower, ci_upper = _wilson_interval(event_n, n_group)
        rows.append(
            {
                "Trajectory": trajectory,
                "N": n_group,
                "Percent": 100.0 * _safe_rate(n_group, total_n),
                "FGR events": event_n,
                "FGR event rate": event_rate,
                "Event-rate CI lower": ci_lower,
                "Event-rate CI upper": ci_upper,
                "Enrichment vs U3 risk set": (
                    event_rate / reference_rate if reference_rate > 0 else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["N", "FGR event rate"], ascending=[False, False]
    ).reset_index(drop=True)


def summarize_trajectory_groups(trajectory_df):
    eligible_reference = trajectory_df.loc[trajectory_df["U3_landmark_eligible"]]
    reference_rate = float(eligible_reference["FGR"].mean())
    rows = []
    for group_name, group in trajectory_df.groupby("trajectory_group", sort=False):
        n_group = len(group)
        event_n = int(group["FGR"].sum())
        event_rate = _safe_rate(event_n, n_group)
        ci_lower, ci_upper = _wilson_interval(event_n, n_group)
        denominator = (
            int(U3_delivered_before_landmark.sum())
            if group_name == "Delivered before U3"
            else len(eligible_reference)
        )
        rows.append(
            {
                "Trajectory group": group_name,
                "N": n_group,
                "Percent of relevant population": 100.0 * _safe_rate(n_group, denominator),
                "FGR events": event_n,
                "FGR event rate": event_rate,
                "Event-rate CI lower": ci_lower,
                "Event-rate CI upper": ci_upper,
                "Enrichment vs U3 risk set": (
                    event_rate / reference_rate if reference_rate > 0 else np.nan
                ),
                "Interpret cautiously": n_group < MIN_GROUP_SIZE_FOR_INTERPRETATION,
            }
        )
    return pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)


def summarize_transition_matrix(
    trajectory_df,
    from_visit,
    to_visit,
    positive_only=False,
):
    analysis_df = trajectory_df
    if positive_only:
        analysis_df = analysis_df.loc[analysis_df["FGR"] == 1]
    if to_visit == "U3":
        analysis_df = analysis_df.loc[analysis_df["U3_landmark_eligible"]]
    state_order = ["L", "R", "H"]
    count_table = pd.crosstab(
        analysis_df[f"{from_visit}_state"],
        analysis_df[f"{to_visit}_state"],
    ).reindex(index=state_order, columns=state_order, fill_value=0)
    row_percent = count_table.div(
        count_table.sum(axis=1).replace(0, np.nan), axis=0
    ) * 100.0
    return count_table, row_percent


def summarize_longitudinal_metrics(trajectory_df):
    rows = []
    subsets = {
        "All samples": pd.Series(True, index=trajectory_df.index),
        "FGR-positive only": trajectory_df["FGR"] == 1,
        "FGR-negative only": trajectory_df["FGR"] == 0,
    }
    for subset_name, subset_mask in subsets.items():
        subset = trajectory_df.loc[subset_mask]
        full_path = subset.loc[subset["U3_landmark_eligible"]]
        if len(full_path):
            for metric_name, numerator in {
                "Any state change": int((full_path["changed_any"] == 1).sum()),
                "Two state changes": int((full_path["number_of_changes"] == 2).sum()),
            }.items():
                rows.append(
                    {
                        "Subset": subset_name,
                        "Transition": "Across U1-U3",
                        "Metric": metric_name,
                        "Numerator": numerator,
                        "Denominator": len(full_path),
                        "Rate": _safe_rate(numerator, len(full_path)),
                    }
                )

        for from_visit, to_visit in [("U1", "U2"), ("U2", "U3")]:
            transition_df = subset
            if to_visit == "U3":
                transition_df = transition_df.loc[
                    transition_df["U3_landmark_eligible"]
                ]
            from_state = transition_df[f"{from_visit}_state"]
            to_state = transition_df[f"{to_visit}_state"]
            y = transition_df["FGR"]
            review_mask = from_state == "R"
            resolution_mask = review_mask & to_state.isin(["L", "H"])
            correct_resolution = resolution_mask & (
                ((to_state == "H") & (y == 1))
                | ((to_state == "L") & (y == 0))
            )
            high_emergence_den = from_state != "H"
            metrics_to_report = [
                ("Uncertainty resolution (R to L/H)", resolution_mask, review_mask),
                ("Correct uncertainty resolution", correct_resolution, review_mask),
                ("R to L", review_mask & (to_state == "L"), review_mask),
                ("R to H", review_mask & (to_state == "H"), review_mask),
                ("High-risk emergence", high_emergence_den & (to_state == "H"), high_emergence_den),
                ("State stability", from_state == to_state, pd.Series(True, index=transition_df.index)),
            ]
            for metric_name, numerator_mask, denominator_mask in metrics_to_report:
                numerator = int(numerator_mask.sum())
                denominator = int(denominator_mask.sum())
                rows.append(
                    {
                        "Subset": subset_name,
                        "Transition": f"{from_visit}->{to_visit}",
                        "Metric": metric_name,
                        "Numerator": numerator,
                        "Denominator": denominator,
                        "Rate": _safe_rate(numerator, denominator),
                    }
                )
    return pd.DataFrame(rows)


def build_aligned_co_outcomes():
    """Build descriptive non-FGR outcomes aligned to the FGR master cohort."""
    full_index = master_index
    birthweight_subset = np.asarray(bw, dtype=float)[full_index]

    co_outcomes = {
        "GDM": np.asarray(GDM_glm, dtype=float)[full_index],
        "PE": np.asarray(PE_label, dtype=float)[full_index],
        "PTB": np.asarray(PTB, dtype=float)[full_index],
        "Macrosomia_4000g": np.where(
            np.isnan(birthweight_subset),
            np.nan,
            (birthweight_subset >= 4000.0).astype(float),
        ),
    }
    for outcome_name, values in co_outcomes.items():
        if len(values) != len(label1):
            raise ValueError(f"Co-outcome {outcome_name} is not aligned.")
    return co_outcomes

def summarize_co_outcome_enrichment(trajectory_df, co_outcomes):
    eligible = trajectory_df["U3_landmark_eligible"]
    group_masks = {
        "Whole FGR modelling cohort": pd.Series(True, index=trajectory_df.index),
        "U3 landmark risk set": eligible,
        "Delivered before U3": trajectory_df["delivered_before_U3"],
        "Any state change among U3-eligible": eligible & (trajectory_df["changed_any"] == 1),
        "Stable low (L-L-L)": eligible & (trajectory_df["trajectory"] == "L-L-L"),
        "End high at U3": eligible & (trajectory_df["U3_state"] == "H"),
        "Emerging high by U3": eligible & (trajectory_df["U1_state"] != "H") & (trajectory_df["U3_state"] == "H"),
        "Persistent high (H-H-H)": eligible & (trajectory_df["trajectory"] == "H-H-H"),
        "Persistent review (R-R-R)": eligible & (trajectory_df["trajectory"] == "R-R-R"),
        "Review resolved by U3": eligible & (trajectory_df["U1_state"] == "R") & trajectory_df["U3_state"].isin(["L", "H"]),
        "R-to-H at either transition": (
            ((trajectory_df["U1_state"] == "R") & (trajectory_df["U2_state"] == "H"))
            | (eligible & (trajectory_df["U2_state"] == "R") & (trajectory_df["U3_state"] == "H"))
        ),
    }
    rows = []
    for outcome_name, values in co_outcomes.items():
        outcome_series = pd.Series(values, index=trajectory_df.index, dtype=float)
        valid_all = outcome_series.notna()
        baseline_rate = float(outcome_series.loc[valid_all].mean())
        for group_name, group_mask in group_masks.items():
            valid_group = group_mask & valid_all
            group_n = int(valid_group.sum())
            event_n = int(outcome_series.loc[valid_group].sum()) if group_n else 0
            prevalence = _safe_rate(event_n, group_n)
            rows.append(
                {
                    "Other outcome": outcome_name,
                    "Trajectory stratum": group_name,
                    "N with outcome data": group_n,
                    "Events": event_n,
                    "Prevalence": prevalence,
                    "Cohort prevalence": baseline_rate,
                    "Prevalence ratio": (
                        prevalence / baseline_rate
                        if baseline_rate > 0 and not np.isnan(prevalence)
                        else np.nan
                    ),
                    "Interpret cautiously": group_n < MIN_GROUP_SIZE_FOR_INTERPRETATION,
                }
            )
    return pd.DataFrame(rows)


def print_trajectory_analysis(trajectory_df, confidence, co_outcomes=None):
    print("\n" + "=" * 100)
    print(
        f"SUBJECT-LEVEL FGR TRAJECTORY ANALYSIS: confidence={confidence:.0%}, "
        f"alpha={1.0-confidence:.2f}"
    )
    print("=" * 100)
    eligible = trajectory_df.loc[trajectory_df["U3_landmark_eligible"]]
    print(
        f"Master cohort N={len(trajectory_df)}, FGR events={int(trajectory_df['FGR'].sum())} "
        f"({trajectory_df['FGR'].mean():.2%}); U3 risk set N={len(eligible)}, "
        f"events={int(eligible['FGR'].sum())} ({eligible['FGR'].mean():.2%}); "
        f"delivered before U3 N={int(trajectory_df['delivered_before_U3'].sum())}"
    )

    coverage_summary = summarize_empirical_coverage(trajectory_df, confidence)
    print("\n0) Empirical marginal and class-conditional coverage:")
    print(
        coverage_summary.to_string(
            index=False,
            formatters={
                "Empirical coverage": lambda x: f"{x:.4f}",
                "95% CI lower": lambda x: f"{x:.4f}",
                "95% CI upper": lambda x: f"{x:.4f}",
                "Difference from nominal": lambda x: f"{x:+.4f}",
            },
        )
    )

    state_summary = summarize_state_prevalence(trajectory_df)
    print("\n1) L/R/H states and structural D markers by visit:")
    print(
        state_summary.to_string(
            index=False,
            formatters={
                "Percent of subset": lambda x: f"{x:.2f}",
                "Percent among at-risk": lambda x: f"{x:.2f}" if pd.notna(x) else "NaN",
            },
        )
    )

    early_summary, early_path = summarize_early_delivery(trajectory_df)
    print("\n2) Pregnancies delivered before U3 (D, not compared with U3 L/R/H):")
    if early_summary.empty:
        print("None")
    else:
        print(early_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("U1-U2 paths before delivery:")
        print(early_path.to_string(index=False))

    exact_summary = summarize_exact_trajectories(trajectory_df, positive_only=False)
    print("\n3) Exact U1-U2-U3 trajectories in the U3 risk set only:")
    print(exact_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    positive_summary = summarize_exact_trajectories(trajectory_df, positive_only=True)
    print("\n4) Exact trajectory distribution among FGR-positive pregnancies in the U3 risk set:")
    if positive_summary.empty:
        print("None")
    else:
        print(
            positive_summary[["Trajectory", "N", "Percent"]].to_string(
                index=False, float_format=lambda x: f"{x:.2f}"
            )
        )

    grouped_summary = summarize_trajectory_groups(trajectory_df)
    print("\n5) Clinically grouped trajectories and observed FGR incidence:")
    print(grouped_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    transition_tables = {}
    for positive_only in [False, True]:
        subset_label = "FGR-positive only" if positive_only else "all samples"
        for from_visit, to_visit in [("U1", "U2"), ("U2", "U3")]:
            counts, row_percent = summarize_transition_matrix(
                trajectory_df,
                from_visit,
                to_visit,
                positive_only=positive_only,
            )
            key = f"{'positive' if positive_only else 'all'}_{from_visit}_{to_visit}"
            transition_tables[key + "_counts"] = counts
            transition_tables[key + "_row_percent"] = row_percent
            suffix = " (D excluded)" if to_visit == "U3" else ""
            print(f"\n6) {from_visit}->{to_visit} counts ({subset_label}){suffix}:")
            print(counts.to_string())
            print("Row percentages:")
            print(row_percent.to_string(float_format=lambda x: f"{x:.2f}"))

    longitudinal_metrics = summarize_longitudinal_metrics(trajectory_df)
    print("\n7) Longitudinal change and uncertainty-resolution metrics:")
    print(longitudinal_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    co_outcome_summary = None
    if co_outcomes is not None:
        co_outcome_summary = summarize_co_outcome_enrichment(trajectory_df, co_outcomes)
        print("\n8) Descriptive co-outcome prevalence by landmark-aware trajectory stratum:")
        print(co_outcome_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return {
        "subject_level": trajectory_df,
        "coverage_summary": coverage_summary,
        "state_summary": state_summary,
        "early_delivery_summary": early_summary,
        "early_delivery_U1_U2_path_summary": early_path,
        "exact_trajectory_summary": exact_summary,
        "positive_only_trajectory_summary": positive_summary,
        "grouped_trajectory_summary": grouped_summary,
        "transition_tables": transition_tables,
        "longitudinal_metrics": longitudinal_metrics,
        "co_outcome_summary": co_outcome_summary,
    }


def plot_and_save_conformal_result(result):
    colors = ["#69a963", "#5e548e", "#822e2e", "#d8d8d8"]
    labels = ["Single correct", "Single error", "Multiple", "Empty"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.stackplot(
        alphas,
        result["cp_single_correct"],
        result["cp_single_error"],
        result["cp_multiple"],
        result["cp_empty"],
        colors=colors,
        labels=labels,
        alpha=0.95,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(f"{PRIMARY_MODEL_DISPLAY} Mondrian CP ({result['visit']}, no isotonic)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Significance (1 - Confidence)")
    ax.set_ylabel("Fraction")
    fig.tight_layout()
    output_path = os.path.join(
        OUTPUT_DIR, f"conformal_curve_{result['visit']}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# =============================================================================
# Full 9,289-participant cohort alignment for cross-outcome workload analysis
# =============================================================================
# State semantics in the standardized export:
#   L/R/H = outcome model generated a conformal state;
#   D     = structurally unavailable at a later visit because delivery/event
#           occurred before the landmark;
#   X     = excluded from this outcome model from the outset because the target
#           was not applicable or could not be defined.
#
# X is deliberately distinct from raw conformal state E, which means an empty
# prediction set and is mapped to R in the analysis state.
FULL_COHORT_EXCLUDED_STATE = "X"
FULL_COHORT_SUBJECT_IDS = np.asarray(GDM_ID, dtype=object)
FULL_COHORT_N = len(FULL_COHORT_SUBJECT_IDS)
OUTCOME_NAME_FOR_ALIGNMENT = "FGR"


def _full_cohort_exclusion_reason():
    fgr_code = np.asarray(fgr, dtype=float)
    reason = np.full(FULL_COHORT_N, "", dtype=object)
    reason[~np.isfinite(fgr_code)] = "missing_FGR_outcome"
    unsupported = np.isfinite(fgr_code) & ~np.isin(fgr_code, [1.0, 2.0])
    reason[unsupported] = "unsupported_FGR_code"
    return reason


def build_full_cohort_oof_export(oof_df):
    """Expand the outcome-specific OOF table to the unchanged 9,289-person cohort."""
    if oof_df["subject_id"].duplicated().any():
        duplicated = oof_df.loc[oof_df["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(
            "Outcome-specific OOF table contains duplicate subject IDs: "
            + ", ".join(map(str, duplicated[:10]))
        )
    if pd.Series(FULL_COHORT_SUBJECT_IDS).duplicated().any():
        raise ValueError("GDM_ID contains duplicate subject IDs; full-cohort alignment is unsafe.")

    eligible_index = np.asarray(master_index, dtype=int)
    if eligible_index.ndim != 1:
        raise ValueError("Outcome modelling index must be one-dimensional.")
    if len(eligible_index) != len(oof_df):
        raise ValueError(
            "Outcome modelling index does not match the OOF table length: "
            f"{len(eligible_index)} versus {len(oof_df)}."
        )
    if np.any(eligible_index < 0) or np.any(eligible_index >= FULL_COHORT_N):
        raise IndexError("Outcome modelling index contains positions outside GDM_ID.")
    if len(np.unique(eligible_index)) != len(eligible_index):
        raise ValueError("Outcome modelling index contains duplicate full-cohort positions.")

    eligible_mask = np.zeros(FULL_COHORT_N, dtype=bool)
    eligible_mask[eligible_index] = True

    full_label = np.full(FULL_COHORT_N, np.nan, dtype=float)
    full_label[eligible_index] = np.asarray(label1, dtype=float)

    reason = np.asarray(_full_cohort_exclusion_reason(), dtype=object)
    if len(reason) != FULL_COHORT_N:
        raise ValueError("Full-cohort exclusion-reason vector has the wrong length.")
    reason[eligible_mask] = ""

    full_df = pd.DataFrame({"subject_id": FULL_COHORT_SUBJECT_IDS})
    full_df = full_df.merge(
        oof_df,
        on="subject_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    full_df.insert(1, "outcome", OUTCOME_NAME_FOR_ALIGNMENT)
    full_df.insert(2, "outcome_model_eligible", eligible_mask)
    full_df.insert(3, "outcome_exclusion_reason", reason)
    full_df.insert(4, "outcome_label_defined", eligible_mask)
    full_df[OUTCOME_NAME_FOR_ALIGNMENT] = full_label

    state_columns = [
        column
        for column in full_df.columns
        if column.startswith(("U1_", "U2_", "U3_")) and "_state_" in column
    ]
    for column in state_columns:
        full_df.loc[~eligible_mask, column] = FULL_COHORT_EXCLUDED_STATE
        if full_df.loc[eligible_mask, column].isna().any():
            raise ValueError(
                f"Eligible participants have missing states after full-cohort alignment: {column}."
            )

    if len(full_df) != FULL_COHORT_N:
        raise RuntimeError(
            f"Full-cohort export has {len(full_df)} rows; expected {FULL_COHORT_N}."
        )
    if not np.array_equal(
        full_df["subject_id"].to_numpy(dtype=object),
        FULL_COHORT_SUBJECT_IDS,
    ):
        raise RuntimeError("Full-cohort export no longer follows the original GDM_ID order.")

    return full_df


def save_analysis_results(
    visit_results,
    auc_summary,
    auc_comparison_wide,
    trajectory_outputs,
):
    """Save all principal and participant-level results to the local output folder."""
    auc_summary.to_csv(os.path.join(OUTPUT_DIR, "auc_summary.csv"), index=False)
    auc_comparison_wide.to_csv(
        os.path.join(OUTPUT_DIR, "auc_comparison_wide.csv")
    )

    oof_df = pd.DataFrame(
        {
            "subject_id": subject_ids_master,
            "FGR": label1,
            "GA_week": GA_WK_aligned,
            "GA_day": GA_DAYS_aligned,
            "GA_total_weeks": GA_total_weeks_aligned,
            "U3_landmark_eligible": U3_landmark_eligible,
            "delivered_before_U3": U3_delivered_before_landmark,
        }
    )
    cp_curve_rows = []
    matched_score_threshold_rows = []
    for visit_name, result in visit_results.items():
        for model_name, predictions in result["oof_predictions_by_model"].items():
            safe_model_name = model_name.replace("-", "_").replace(" ", "_")
            oof_df[f"{visit_name}_{safe_model_name}_OOF_probability"] = predictions
        safe_cp_model_name = PRIMARY_MODEL_DISPLAY.replace("-", "_").replace(" ", "_")
        oof_df[f"{visit_name}_{safe_cp_model_name}_CP_probability"] = result["oof_cp_probability"]
        oof_df[f"{visit_name}_fold_id"] = result["oof_fold_id"]
        matched_score_threshold_rows.extend(result["matched_score_thresholds"])
        for confidence in TRAJECTORY_CONFIDENCE_LEVELS:
            confidence = float(confidence)
            suffix = str(int(round(confidence * 100)))
            oof_df[f"{visit_name}_state_{suffix}"] = result["oof_cp_state"][confidence]
            oof_df[f"{visit_name}_raw_state_{suffix}"] = result[
                "oof_cp_raw_state"
            ][confidence]
            oof_df[f"{visit_name}_score_state_{suffix}"] = result[
                "oof_score_state"
            ][confidence]
        for i, alpha in enumerate(alphas):
            cp_curve_rows.append(
                {
                    "Visit": visit_name,
                    "Alpha": alpha,
                    "Single correct": result["cp_single_correct"][i],
                    "Single error": result["cp_single_error"][i],
                    "Multiple": result["cp_multiple"][i],
                    "Empty": result["cp_empty"][i],
                    "Positive singleton correct": result["cp_positive_accuracy"][i],
                    "Negative singleton as positive": result["cp_negative_error"][i],
                }
            )
    oof_df.to_csv(
        os.path.join(OUTPUT_DIR, "subject_level_oof_predictions_and_states.csv"),
        index=False,
    )
    full_oof_df = build_full_cohort_oof_export(oof_df)
    full_oof_path = os.path.join(
        OUTPUT_DIR, "subject_level_oof_predictions_and_states_full_cohort.csv"
    )
    full_oof_df.to_csv(full_oof_path, index=False)
    print(
        "Full-cohort aligned participant export:",
        len(full_oof_df),
        "rows; outcome-excluded X =",
        int((~full_oof_df["outcome_model_eligible"]).sum()),
    )
    cp_curve_df = pd.DataFrame(cp_curve_rows)
    cp_curve_df.to_csv(os.path.join(OUTPUT_DIR, "conformal_curves.csv"), index=False)
    matched_score_threshold_df = pd.DataFrame(matched_score_threshold_rows)
    matched_score_threshold_df.to_csv(
        os.path.join(OUTPUT_DIR, "matched_score_thresholds_by_fold.csv"),
        index=False,
    )

    for confidence, output in trajectory_outputs.items():
        suffix = str(int(round(float(confidence) * 100)))
        for key, table in output.items():
            if key == "transition_tables":
                for transition_name, transition_table in table.items():
                    transition_table.to_csv(
                        os.path.join(
                            OUTPUT_DIR,
                            f"trajectory_{suffix}_{transition_name}.csv",
                        )
                    )
                continue
            if isinstance(table, pd.DataFrame):
                table.to_csv(
                    os.path.join(OUTPUT_DIR, f"trajectory_{suffix}_{key}.csv"),
                    index=False,
                )

    excel_path = os.path.join(OUTPUT_DIR, f"FGR_{MODEL_SELECTION}_CP_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        auc_summary.to_excel(writer, sheet_name="AUC_summary", index=False)
        auc_comparison_wide.to_excel(writer, sheet_name="AUC_wide")
        cp_curve_df.to_excel(writer, sheet_name="CP_curves", index=False)
        matched_score_threshold_df.to_excel(writer, sheet_name="score_thresholds", index=False)
        oof_df.to_excel(writer, sheet_name="OOF_subjects", index=False)
        full_oof_df.to_excel(
            writer, sheet_name="OOF_full_cohort", index=False
        )
        for confidence, output in trajectory_outputs.items():
            suffix = str(int(round(float(confidence) * 100)))
            sheet_map = {
                "coverage_summary": f"coverage_{suffix}",
                "state_summary": f"states_{suffix}",
                "early_delivery_summary": f"earlyD_{suffix}",
                "early_delivery_U1_U2_path_summary": f"earlyD_path_{suffix}",
                "exact_trajectory_summary": f"exact_{suffix}",
                "positive_only_trajectory_summary": f"positive_{suffix}",
                "grouped_trajectory_summary": f"groups_{suffix}",
                "longitudinal_metrics": f"metrics_{suffix}",
                "co_outcome_summary": f"cooutcomes_{suffix}",
            }
            for key, sheet_name in sheet_map.items():
                table = output.get(key)
                if isinstance(table, pd.DataFrame):
                    table.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    config_path = os.path.join(OUTPUT_DIR, "run_configuration.txt")
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write("Outcome: FGR based on CMAE08 (1=FGR, 2=no FGR)\n")
        handle.write(f"Model selection switch: {MODEL_SELECTION}\n")
        handle.write(f"Primary/CP model key: {PRIMARY_MODEL_KEY}\n")
        handle.write("AUC models: " + ", ".join(ACTIVE_MODEL_DISPLAY_NAMES) + "\n")
        handle.write(f"Primary AUC model: {PRIMARY_MODEL_DISPLAY}\n")
        handle.write(f"Conformal model: {PRIMARY_MODEL_DISPLAY} without isotonic calibration\n")
        handle.write(f"Cross-validation folds: {N_SPLITS}\n")
        handle.write(f"Fixed random state: {FIXED_GAMMA}\n")
        handle.write(f"Conformal calibration fraction: {CALIBRATION_SIZE}\n")
        handle.write(f"U3 landmark: {U3_LANDMARK_START_WEEKS:.1f} weeks\n")
        handle.write(
            f"Excluded/marked D before U3: {int(U3_delivered_before_landmark.sum())}\n"
        )
        handle.write(f"Full alignment cohort: {FULL_COHORT_N}\n")
        handle.write(
            f"Outcome-level excluded participants marked X: "
            f"{int((~full_oof_df['outcome_model_eligible']).sum())}\n"
        )
        handle.write(
            "State X is outcome-level ineligibility; D remains a structural "
            "pre-landmark event/delivery state; raw E remains an empty CP set.\n"
        )
        handle.write(
            "Confidence levels: "
            + ", ".join(f"{value:.0%}" for value in TRAJECTORY_CONFIDENCE_LEVELS)
            + "\n"
        )

    print("\nAll analysis results were saved locally to:")
    print(OUTPUT_DIR)
    return {
        "output_dir": OUTPUT_DIR,
        "full_cohort_oof_file": full_oof_path,
        "excel_file": excel_path,
        "oof_file": os.path.join(
            OUTPUT_DIR, "subject_level_oof_predictions_and_states.csv"
        ),
    }


# =============================================================================
# Visit-specific fixed CV, landmark-aware model execution and local saving
# =============================================================================

# =============================================================================
# Performance-only nested-CV analysis aligned with the current manuscript
# =============================================================================
L1_RATIO_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
C_GRID = (0.001, 0.01, 0.1, 0.5, 1.0)
INNER_SPLITS = 5
EPS = 1e-6


def make_tuned_logistic_model(positive_weight, C, l1_ratio):
    return LogisticRegression(
        C=float(C), penalty="elasticnet", solver="saga", l1_ratio=float(l1_ratio),
        class_weight={0: 1.0, 1: float(positive_weight)}, max_iter=50000,
        random_state=FIXED_GAMMA,
    )


def select_hyperparameters_nested(X_train_raw, y_train, outer_fold, visit_name):
    """Fivefold inner CV; preprocessing is refitted inside every inner-training fold."""
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=FIXED_GAMMA + outer_fold)
    rows = []
    for l1_ratio in L1_RATIO_GRID:
        for C in C_GRID:
            aucs = []
            for inner_train, inner_val in inner.split(np.zeros(len(y_train)), y_train):
                Xi_train_raw = X_train_raw.iloc[inner_train]
                Xi_val_raw = X_train_raw.iloc[inner_val]
                yi_train = y_train[inner_train]
                yi_val = y_train[inner_val]
                Xi_train, prep = fit_preprocessor(Xi_train_raw)
                Xi_val = transform_preprocessor(Xi_val_raw, prep)
                pw = calculate_sqrt_positive_weight(yi_train)
                model = make_tuned_logistic_model(pw, C=C, l1_ratio=l1_ratio)
                model.fit(Xi_train, yi_train)
                p = model.predict_proba(Xi_val)[:, 1]
                aucs.append(roc_auc_score(yi_val, p))
            rows.append({
                "visit": visit_name, "outer_fold": outer_fold, "C": C,
                "l1_ratio": l1_ratio, "mean_inner_AUROC": float(np.mean(aucs)),
                "sd_inner_AUROC": float(np.std(aucs, ddof=1)),
            })
    tuning = pd.DataFrame(rows).sort_values(
        ["mean_inner_AUROC", "C", "l1_ratio"], ascending=[False, True, True]
    ).reset_index(drop=True)
    best = tuning.iloc[0]
    return float(best["C"]), float(best["l1_ratio"]), tuning


def bootstrap_metric_ci(y, p, metric_fn, n_boot=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_RANDOM_STATE):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if np.unique(yb).size < 2:
            continue
        vals.append(float(metric_fn(yb, pb)))
    if not vals:
        return np.nan, np.nan
    return tuple(np.percentile(vals, [2.5, 97.5]))


def calibration_intercept_slope(y, p):
    """Logistic recalibration on logit(p): intercept with slope fixed at 1, and slope with intercept free."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=int)
    lp = np.log(p / (1.0 - p))
    # Calibration intercept: logit(y) = intercept + offset(logit(p)).
    intercept_fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    calibration_intercept = float(intercept_fit.params[0])
    # Calibration slope: logit(y) = intercept + slope * logit(p).
    slope_fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    calibration_slope = float(slope_fit.params[1])
    return calibration_intercept, calibration_slope


def make_filtered_splits(base_splits, eligible_mask):
    """Preserve master fold assignment while excluding visit-ineligible participants."""
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    out = []
    for train_idx, test_idx in base_splits:
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        out.append((train_idx[eligible_mask[train_idx]], test_idx[eligible_mask[test_idx]]))
    return out


def run_performance_visit(visit_name, feature_names, cv_splits, analysis_mask):
    X_raw = build_raw_feature_dataframe(feature_names)
    analysis_mask = np.asarray(analysis_mask, dtype=bool)
    oof = np.full(len(label1), np.nan, dtype=float)
    fold_id = np.full(len(label1), np.nan, dtype=float)
    tuning_rows, selected_rows = [], []

    for fold_number, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(f"{visit_name} fold {fold_number}: empty train/test set after eligibility filtering.")
        y_train = label1[train_idx]
        X_train_raw = X_raw.iloc[train_idx]
        X_test_raw = X_raw.iloc[test_idx]

        best_C, best_l1, tuning = select_hyperparameters_nested(
            X_train_raw, y_train, outer_fold=fold_number, visit_name=visit_name
        )
        tuning_rows.append(tuning)

        X_train, prep = fit_preprocessor(X_train_raw)
        X_test = transform_preprocessor(X_test_raw, prep)
        pw = calculate_sqrt_positive_weight(y_train)
        model = make_tuned_logistic_model(pw, C=best_C, l1_ratio=best_l1)
        model.fit(X_train, y_train)
        oof[test_idx] = model.predict_proba(X_test)[:, 1]
        fold_id[test_idx] = fold_number
        selected_rows.append({
            "visit": visit_name, "outer_fold": fold_number, "train_n": len(train_idx),
            "test_n": len(test_idx), "positive_weight": pw, "selected_C": best_C,
            "selected_l1_ratio": best_l1, "retained_features": X_train.shape[1],
        })
        print(f"{visit_name} fold {fold_number}/10: C={best_C:g}, l1_ratio={best_l1:g}, features={X_train.shape[1]}")

    if np.isnan(oof[analysis_mask]).any():
        raise RuntimeError(f"{visit_name}: missing OOF predictions among eligible participants.")
    if np.any(~np.isnan(oof[~analysis_mask])):
        raise RuntimeError(f"{visit_name}: ineligible participants received predictions.")

    y = label1[analysis_mask]
    p = oof[analysis_mask]
    auroc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    cal_i, cal_s = calibration_intercept_slope(y, p)
    auc_lo, auc_hi = bootstrap_metric_ci(y, p, roc_auc_score, seed=BOOTSTRAP_RANDOM_STATE + 10)
    pr_lo, pr_hi = bootstrap_metric_ci(y, p, average_precision_score, seed=BOOTSTRAP_RANDOM_STATE + 20)
    brier_lo, brier_hi = bootstrap_metric_ci(y, p, brier_score_loss, seed=BOOTSTRAP_RANDOM_STATE + 30)

    summary = {
        "outcome": "FGR", "visit": visit_name, "n": int(len(y)), "events": int(y.sum()),
        "prevalence": float(y.mean()), "AUROC": float(auroc), "AUROC_CI95_low": float(auc_lo),
        "AUROC_CI95_high": float(auc_hi), "AUPRC": float(auprc), "AUPRC_CI95_low": float(pr_lo),
        "AUPRC_CI95_high": float(pr_hi), "Brier_score": float(brier),
        "Brier_CI95_low": float(brier_lo), "Brier_CI95_high": float(brier_hi),
        "calibration_intercept": cal_i, "calibration_slope": cal_s,
        "mean_predicted_probability": float(np.mean(p)),
    }
    subject = pd.DataFrame({
        "subject_id": subject_ids_master, "outcome": label1, "visit": visit_name,
        "eligible": analysis_mask, "outer_fold": fold_id, "oof_probability": oof,
    })
    return summary, subject, pd.concat(tuning_rows, ignore_index=True), pd.DataFrame(selected_rows)


# Delivery-based visit eligibility: V2 >=22+0 weeks; V3 >=30+0 weeks.
V2_LANDMARK_START_WEEKS = 22.0
V2_landmark_eligible = ~(
    np.isfinite(GA_total_weeks_aligned) & (GA_total_weeks_aligned < V2_LANDMARK_START_WEEKS)
)
FULL_COHORT_MASK = np.ones(len(label1), dtype=bool)
VISIT_ANALYSIS_MASKS = {
    "U1": FULL_COHORT_MASK.copy(),
    "U2": np.asarray(V2_landmark_eligible, dtype=bool),
    "U3": np.asarray(U3_landmark_eligible, dtype=bool),
}
# Generate the master 10-fold assignment once; V2 retains the same assignment after
# eligibility filtering. U3 is repartitioned within its eligible risk set.
full_cv_splits = _make_stratified_master_splits(FULL_COHORT_MASK)
VISIT_CV_SPLITS = {
    "U1": full_cv_splits,
    "U2": make_filtered_splits(full_cv_splits, VISIT_ANALYSIS_MASKS["U2"]),
    "U3": _make_stratified_master_splits(VISIT_ANALYSIS_MASKS["U3"]),
}

summaries, subject_tables, tuning_tables, selected_tables = [], [], [], []
for visit_name in ("U1", "U2", "U3"):
    print(f"\n=== {visit_name} nested-CV performance analysis ===")
    summary, subject, tuning, selected = run_performance_visit(
        visit_name, FEATURE_SETS[visit_name], VISIT_CV_SPLITS[visit_name], VISIT_ANALYSIS_MASKS[visit_name]
    )
    summaries.append(summary); subject_tables.append(subject); tuning_tables.append(tuning); selected_tables.append(selected)

summary_df = pd.DataFrame(summaries)
subject_df = pd.concat(subject_tables, ignore_index=True)
tuning_df = pd.concat(tuning_tables, ignore_index=True)
selected_df = pd.concat(selected_tables, ignore_index=True)

summary_path = os.path.join(OUTPUT_DIR, "FGR_nestedCV_performance_summary.csv")
subject_path = os.path.join(OUTPUT_DIR, "FGR_nestedCV_subject_level_oof_predictions.csv")
tuning_path = os.path.join(OUTPUT_DIR, "FGR_nestedCV_inner_tuning_all_candidates.csv")
selected_path = os.path.join(OUTPUT_DIR, "FGR_nestedCV_selected_hyperparameters_by_fold.csv")
summary_df.to_csv(summary_path, index=False)
subject_df.to_csv(subject_path, index=False)
tuning_df.to_csv(tuning_path, index=False)
selected_df.to_csv(selected_path, index=False)

print("\nFinal performance summary:")
print(summary_df.to_string(index=False))
print("\nSaved to:", OUTPUT_DIR)
print(" -", summary_path)
print(" -", subject_path)
print(" -", tuning_path)
print(" -", selected_path)
