from sklearn.model_selection import train_test_split
import os
import numpy as np
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

# Optional legacy table-model dependency.
# CatBoost code is retained for compatibility but is not part of the central
# LR/EBM model switch used by this analysis.
RUN_CATBOOST = False

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

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
PTB_A09_raw=(data_read_and_fill_fornum(4, 'A09A03a', GDM_ID))
PTB_A09_raw_sub=(data_read_and_fill_fornum(4, 'A09A03a1', GDM_ID))
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
# bw=np.array((data_read_and_fill_fornum(40, 'bw', GDM_ID)))
# GA_WK=np.array((data_read_and_fill_fornum(40, 'GAwksA09', GDM_ID)))
# GA_DAYS=np.array((data_read_and_fill_fornum(40, 'GAdysA09', GDM_ID)))
import numpy as np
from scipy.stats import norm


# ============================================================
# 1. 读取出生信息
# ============================================================

bw = np.asarray(
    data_read_and_fill_fornum(40, 'bw', GDM_ID),
    dtype=float
)

GA_WK = np.asarray(
    data_read_and_fill_fornum(40, 'GAwksA09', GDM_ID),
    dtype=float
)

GA_DAYS = np.asarray(
    data_read_and_fill_fornum(40, 'GAdysA09', GDM_ID),
    dtype=float
)

# 胎儿性别变量
SEX = np.asarray(
    data_read_and_fill_fornum(21, 'CBAA02', GDM_ID),
    dtype=float
)


# ============================================================
# 2. 确认性别编码
# ============================================================
# 下面默认：
# 1 = Male
# 2 = Female
#
# 请先根据你的 nuMoM2b codebook 确认。
# 如果实际编码不同，只修改下面两个值即可。

MALE_CODE = 1
FEMALE_CODE = 2

print(
    "Observed fetal sex codes:",
    np.unique(SEX[np.isfinite(SEX)])
)


# ============================================================
# 3. Aris 2017 first-born LMS 参数
# 格式：
# gestational_week: (L, M, S)
#
# L = skewness / Box-Cox parameter
# M = median birth weight, grams
# S = coefficient of variation / dispersion
# ============================================================

ARIS_2017_FIRSTBORN_MALE = {
    22: (1.514,  503.890, 0.139),
    23: (1.520,  592.519, 0.156),
    24: (1.510,  683.038, 0.175),
    25: (1.466,  773.691, 0.194),
    26: (1.377,  873.465, 0.209),
    27: (1.259,  987.207, 0.218),
    28: (1.149, 1113.391, 0.220),
    29: (1.066, 1262.635, 0.215),
    30: (1.009, 1429.091, 0.207),
    31: (0.973, 1607.149, 0.199),
    32: (0.945, 1802.863, 0.191),
    33: (0.930, 2022.571, 0.179),
    34: (0.929, 2268.521, 0.166),
    35: (0.910, 2531.031, 0.156),
    36: (0.826, 2764.645, 0.146),
    37: (0.659, 2982.338, 0.136),
    38: (0.491, 3188.996, 0.124),
    39: (0.392, 3377.400, 0.117),
    40: (0.410, 3511.618, 0.112),
    41: (0.485, 3636.136, 0.110),
    42: (0.604, 3740.548, 0.110),
}

ARIS_2017_FIRSTBORN_FEMALE = {
    22: (0.603,  466.402, 0.158),
    23: (0.788,  545.404, 0.176),
    24: (0.930,  625.038, 0.195),
    25: (1.016,  712.144, 0.211),
    26: (1.047,  812.602, 0.221),
    27: (1.041,  924.822, 0.223),
    28: (1.017, 1041.628, 0.220),
    29: (0.983, 1170.746, 0.214),
    30: (0.905, 1325.737, 0.207),
    31: (0.771, 1505.975, 0.202),
    32: (0.638, 1703.148, 0.193),
    33: (0.571, 1922.379, 0.180),
    34: (0.620, 2174.540, 0.167),
    35: (0.715, 2432.968, 0.155),
    36: (0.698, 2671.268, 0.147),
    37: (0.544, 2872.597, 0.138),
    38: (0.371, 3071.299, 0.125),
    39: (0.258, 3250.409, 0.116),
    40: (0.276, 3383.711, 0.110),
    41: (0.312, 3497.869, 0.109),
    42: (0.395, 3592.890, 0.114),
}


# ============================================================
# 4. 计算 Aris 2017 z-score 和百分位
# ============================================================

def calculate_aris_2017_firstborn_percentile(
        birth_weight,
        ga_week,
        ga_day,
        fetal_sex,
        male_code=1,
        female_code=2
):
    """
    Calculate birthweight-for-gestational-age z-scores and percentiles
    using the Aris 2017 U.S. reference, first-born-specific LMS values.

    Parameters
    ----------
    birth_weight : array-like
        Birth weight in grams.
    ga_week : array-like
        Completed gestational weeks, expected range 22-42.
    ga_day : array-like
        Additional gestational days, normally 0-6.
        Aris parameters use completed weeks, so this variable is only
        checked and is not used for interpolation.
    fetal_sex : array-like
        Fetal sex code.
    male_code : numeric
        Code representing male.
    female_code : numeric
        Code representing female.

    Returns
    -------
    z_score : ndarray
        Birthweight-for-GA z-score.
    percentile : ndarray
        Percentile from 0 to 100.
    lga90 : ndarray
        1 = percentile >= 90
        0 = percentile < 90
        NaN = outcome cannot be determined.
    lga97 : ndarray
        1 = percentile >= 97
        0 = percentile < 97
        NaN = outcome cannot be determined.
    sga10 : ndarray
        1 = percentile <= 10
        0 = percentile > 10
        NaN = outcome cannot be determined.
    """

    birth_weight = np.asarray(birth_weight, dtype=float)
    ga_week = np.asarray(ga_week, dtype=float)
    ga_day = np.asarray(ga_day, dtype=float)
    fetal_sex = np.asarray(fetal_sex, dtype=float)

    if not (
        birth_weight.shape == ga_week.shape ==
        ga_day.shape == fetal_sex.shape
    ):
        raise ValueError(
            "bw, GA_WK, GA_DAYS and SEX must have the same shape."
        )

    z_score = np.full(birth_weight.shape, np.nan, dtype=float)
    percentile = np.full(birth_weight.shape, np.nan, dtype=float)

    # Aris reference applies to completed weeks 22-42.
    # Birth weight must be positive, and GA_WK must be an integer week.
    valid_common = (
        np.isfinite(birth_weight) &
        np.isfinite(ga_week) &
        np.isfinite(fetal_sex) &
        (birth_weight > 0) &
        (ga_week >= 22) &
        (ga_week <= 42) &
        (ga_week == np.floor(ga_week))
    )

    # GA_DAYS does not affect the Aris completed-week parameter selection,
    # but report clearly invalid day values.
    invalid_day = (
        np.isfinite(ga_day) &
        (
            (ga_day < 0) |
            (ga_day > 6) |
            (ga_day != np.floor(ga_day))
        )
    )

    if np.any(invalid_day):
        print(
            "Warning:",
            int(np.sum(invalid_day)),
            "records have GA_DAYS outside the valid 0-6 integer range."
        )

    parameter_sets = {
        male_code: ARIS_2017_FIRSTBORN_MALE,
        female_code: ARIS_2017_FIRSTBORN_FEMALE,
    }

    for sex_code, lms_table in parameter_sets.items():

        for week, (L, M, S) in lms_table.items():

            mask = (
                valid_common &
                (fetal_sex == sex_code) &
                (ga_week == week)
            )

            if not np.any(mask):
                continue

            y = birth_weight[mask]

            # General LMS formula
            if np.isclose(L, 0.0):
                z = np.log(y / M) / S
            else:
                z = (((y / M) ** L) - 1.0) / (L * S)

            z_score[mask] = z
            percentile[mask] = norm.cdf(z) * 100.0

    valid_result = np.isfinite(percentile)

    # Keep these arrays as float so missing outcomes remain NaN.
    lga90 = np.full(percentile.shape, np.nan, dtype=float)
    lga97 = np.full(percentile.shape, np.nan, dtype=float)
    sga10 = np.full(percentile.shape, np.nan, dtype=float)

    lga90[valid_result] = (
        percentile[valid_result] >= 90.0
    ).astype(float)

    lga97[valid_result] = (
        percentile[valid_result] >= 97.0
    ).astype(float)

    sga10[valid_result] = (
        percentile[valid_result] <= 10.0
    ).astype(float)

    return z_score, percentile, lga90, lga97, sga10


# ============================================================
# 5. 执行计算
# ============================================================

(
    BW_Z_ARIS,
    BW_PERCENTILE_ARIS,
    LGA90_ARIS,
    LGA97_ARIS,
    SGA10_ARIS
) = calculate_aris_2017_firstborn_percentile(
    birth_weight=bw,
    ga_week=GA_WK,
    ga_day=GA_DAYS,
    fetal_sex=SEX,
    male_code=MALE_CODE,
    female_code=FEMALE_CODE
)


# ============================================================
# 6. 基本结果检查
# ============================================================

valid_aris = np.isfinite(BW_PERCENTILE_ARIS)

print(
    "Valid Aris percentile results:",
    int(np.sum(valid_aris)),
    "/",
    len(BW_PERCENTILE_ARIS)
)

print(
    "Missing or invalid Aris results:",
    int(np.sum(~valid_aris))
)

if np.any(valid_aris):

    print(
        "Aris percentile range:",
        np.nanmin(BW_PERCENTILE_ARIS),
        "to",
        np.nanmax(BW_PERCENTILE_ARIS)
    )

    print(
        "LGA90:",
        int(np.nansum(LGA90_ARIS)),
        "/",
        int(np.sum(valid_aris)),
        "=",
        round(np.nanmean(LGA90_ARIS) * 100, 2),
        "%"
    )

    # print(
    #     "LGA97:",
    #     int(np.nansum(LGA97_ARIS)),
    #     "/",
    #     int(np.sum(valid_aris)),
    #     "=",
    #     round(np.nanmean(LGA97_ARIS) * 100, 2),
    #     "%"
    # )
    #
    # print(
    #     "SGA10:",
    #     int(np.nansum(SGA10_ARIS)),
    #     "/",
    #     int(np.sum(valid_aris)),
    #     "=",
    #     round(np.nanmean(SGA10_ARIS) * 100, 2),
    #     "%"
    # )
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
# =============================================================================
# PTB outcome used by all downstream modelling and trajectory analyses
# =============================================================================
# nuMoM2b all-cause PTB: delivery from 20+0 through 36+6 weeks.
# Records with missing GA or GA <20 weeks are excluded from the modelling cohort.
PTB_relabel0 = []
PTB_ID_relabel0 = []
PTB_index = []

for i in range(len(GA_WK)):
    if not np.isfinite(GA_WK[i]):
        continue

    if 20 <= GA_WK[i] < 37:
        PTB_relabel0.append(1.0)
        PTB_ID_relabel0.append(GDM_ID[i])
        PTB_index.append(i)
    elif GA_WK[i] >= 37:
        PTB_relabel0.append(0.0)
        PTB_ID_relabel0.append(GDM_ID[i])
        PTB_index.append(i)

print(
    "PTB outcome:",
    int(np.sum(PTB_relabel0)),
    "/",
    len(PTB_relabel0),
    "=",
    round(np.mean(PTB_relabel0) * 100, 2),
    "%"
)
pee=np.array((data_read_and_fill_fornum(29, 'CMDA08a1', GDM_ID)))
pee_all=np.zeros(len(pee))
pee_time=np.array((data_read_and_fill_fornum(29, 'CMDA08A3_INT', GDM_ID)))
PTB_A09 = np.asarray(PTB_A09_raw, dtype=float).copy()

for i in range(len(pee)):
    if pee[i]>=2:
        pee_all[i]=1
    if pee[i] > 1 and pee_time[i]<=-21:
        pee[i]=1
    else:
        pee[i]=0
#PTB=np.array((data_read_and_fill_fornum(4, 'A09A03a', HRV_id)))
for i in range(len(PTB_A09)):
    if PTB_A09[i] != 1:
        PTB_A09[i] = 0

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

basic_explaination_rest_GDM_1=['Age', 'BMI at visit 1' ,'Fetal gender', 'pre-existed DM','EPDS score at visit 1',
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
                    "EFW at visit 2","EFW percentile at visit 2"]

basic_explaination_rest_GDM_3=['EPDS score at visit 3',
'Smoke in 3 months before visit 3', 'bleed at visit 3',"Maternal weight at visit 3",
                    "SBP at visit 3","DBP at visit 3","GDM"
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

# Preserve the original feature-name lists before adding PTB-specific features.
Name_explainx_U1_base = list(Name_explainx_U1)
Name_explainx_U2_base = list(Name_explainx_U2)
Name_explainx_U3_base = list(Name_explainx_U3)
Name_explain_full_base = list(Name_explain_full)




BASE_FEATURE_SETS = {
    "U1": Name_explainx_U1_base,
    "U2": Name_explainx_U2_base,
    "U3": Name_explainx_U3_base,
}



# ============================================================================
# 0. 从 dataset 列表中动态查找 dataset index
# ============================================================================


def get_dataset_index(target_dataset_name):
    """
    在已有的 dataset_name 列表中查找指定 dataset 的位置。

    Parameters
    ----------
    target_dataset_name : str
        例如 'V1A'、'V2A'、'U2B'、'VXX'

    Returns
    -------
    int
        该 dataset 在 dataset_name 中的位置，可直接传入
        data_read_and_fill_fornum()。
    """

    target_dataset_name = str(target_dataset_name).strip().upper()

    normalized_dataset_names = [
        str(name).strip().upper()
        for name in dataset_name
    ]

    if target_dataset_name not in normalized_dataset_names:
        raise ValueError(
            f"Dataset '{target_dataset_name}' "
            f"was not found in dataset_name."
        )

    return normalized_dataset_names.index(target_dataset_name)


DATASET_S02 = get_dataset_index('s02')
DATASET_U2B = get_dataset_index('u2b')
DATASET_U3B = get_dataset_index('u3b')
DATASET_V1A = get_dataset_index('v1a')
DATASET_V2A = get_dataset_index('v2a')
DATASET_V3A = get_dataset_index('v3a')
DATASET_VXX = get_dataset_index('vxx')

print(
    {
        's02': DATASET_S02,
        'u2b': DATASET_U2B,
        'u3b': DATASET_U3B,
        'v1a': DATASET_V1A,
        'v2a': DATASET_V2A,
        'v3a': DATASET_V3A,
        'vxx': DATASET_VXX
    }
)


# ============================================================================
# 1. 基础读取和编码函数
# ============================================================================

N_SAMPLE = len(GDM_ID)


def load_numeric(dataset_name, variable_name):
    """
    Load one variable using data_read_and_fill_fornum().
    Missing values remain np.nan.
    """
    dataset_number = get_dataset_index(dataset_name)

    values = np.asarray(
        data_read_and_fill_fornum(
            dataset_number,
            variable_name,
            GDM_ID
        ),
        dtype=float
    ).reshape(-1)

    if len(values) != N_SAMPLE:
        raise ValueError(
            f"{variable_name}: expected {N_SAMPLE} samples, "
            f"but obtained {len(values)}."
        )

    return values


def recode_yes_no(values):
    """
    Yes_No_v1:
        1 = Yes
        2 = No
        3 = Not done/none recorded

    Output:
        1 = Yes
        0 = No
        NaN = missing / unknown / not done
    """
    values = np.asarray(values, dtype=float)

    output = np.full(values.shape, np.nan, dtype=float)
    output[values == 1] = 1.0
    output[values == 2] = 0.0

    return output


def recode_present_absent(values):
    """
    Present_Absent_v2:
        1 = Present
        2 = Not present

    Output:
        1 = Present
        0 = Not present
        NaN = missing
    """
    values = np.asarray(values, dtype=float)

    output = np.full(values.shape, np.nan, dtype=float)
    output[values == 1] = 1.0
    output[values == 2] = 0.0

    return output


def recode_checked(values):
    """
    Checked:
        1 = Checked
        0 = Not checked
    """
    values = np.asarray(values, dtype=float)

    output = np.full(values.shape, np.nan, dtype=float)
    output[values == 1] = 1.0
    output[values == 0] = 0.0

    return output


def combine_any_binary(binary_arrays, strict_missing=True):
    """
    Combine several binary variables into one 'any positive' variable.

    strict_missing=True:
        any 1                          -> 1
        all variables observed as 0   -> 0
        otherwise                     -> NaN

    This is suitable for combining several infection diagnoses.
    """
    stacked = np.vstack(binary_arrays).astype(float)

    output = np.full(stacked.shape[1], np.nan, dtype=float)

    any_positive = np.any(stacked == 1, axis=0)

    if strict_missing:
        all_negative = np.all(stacked == 0, axis=0)
    else:
        any_observed = np.any(np.isfinite(stacked), axis=0)
        all_negative = (
            any_observed &
            ~any_positive
        )

    output[any_positive] = 1.0
    output[all_negative] = 0.0

    return output


def binary_transition(previous, current, previous_value, current_value):
    """
    Create a binary transition indicator.

    Example:
        previous_value=0, current_value=1
        identifies a new positive state.
    """
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)

    output = np.full(previous.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(previous) &
        np.isfinite(current)
    )

    output[valid] = (
        (previous[valid] == previous_value) &
        (current[valid] == current_value)
    ).astype(float)

    return output


def persistent_positive(previous, current):
    """
    1 when both previous and current visits are positive.
    """
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)

    output = np.full(previous.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(previous) &
        np.isfinite(current)
    )

    output[valid] = (
        (previous[valid] == 1) &
        (current[valid] == 1)
    ).astype(float)

    return output


# ============================================================================
# 2. U1特征：症状、宫颈手术史、既往妊娠史
# ============================================================================

cramping_visit_1 = recode_yes_no(
    load_numeric('v1a', 'V1AD07')
)

prior_cervical_cone = recode_yes_no(
    load_numeric('v1a', 'V1AD12a')
)

prior_cervical_leep = recode_yes_no(
    load_numeric('v1a', 'V1AD12b')
)

prior_cervical_cryotherapy = recode_yes_no(
    load_numeric('v1a', 'V1AD12c')
)


# Including current pregnancy, how many times pregnant
number_total_pregnancies = load_numeric(
    'v1a',
    'V1AE01'
)

number_prior_pregnancies = number_total_pregnancies - 1

number_prior_pregnancies[
    number_prior_pregnancies < 0
] = np.nan

no_prior_pregnancy = (
    np.isfinite(number_prior_pregnancies) &
    (number_prior_pregnancies == 0)
)


# ============================================================================
# 3. 读取最多10次既往妊娠
# ============================================================================

prior_loss_list = []
prior_loss_ga_list = []
prior_cervical_insufficiency_list = []
prior_prom_list = []
prior_abruption_list = []
prior_d_and_c_list = []
prior_cerclage_list = []

for pregnancy_number in range(1, 11):

    prior_loss_list.append(
        recode_checked(
            load_numeric(
                'v1a',
                f'V1AE2_02c_{pregnancy_number}'
            )
        )
    )

    prior_loss_ga_list.append(
        load_numeric(
            'v1a',
            f'V1AE2_03a_{pregnancy_number}'
        )
    )

    prior_cervical_insufficiency_list.append(
        recode_checked(
            load_numeric(
                'v1a',
                f'V1AE2_04e_{pregnancy_number}'
            )
        )
    )

    prior_prom_list.append(
        recode_checked(
            load_numeric(
                'v1a',
                f'V1AE2_04f_{pregnancy_number}'
            )
        )
    )

    prior_abruption_list.append(
        recode_checked(
            load_numeric(
                'v1a',
                f'V1AE2_04g_{pregnancy_number}'
            )
        )
    )

    prior_d_and_c_list.append(
        recode_checked(
            load_numeric(
                'v1a',
                f'V1AE2_05b_{pregnancy_number}'
            )
        )
    )

    prior_cerclage_list.append(
        recode_checked(
            load_numeric(
                'v1a',
                f'V1AE2_05d_{pregnancy_number}'
            )
        )
    )


prior_loss_matrix = np.vstack(prior_loss_list)
prior_loss_ga_matrix = np.vstack(prior_loss_ga_list)


# ============================================================================
# 4. 构建既往妊娠汇总变量
# ============================================================================

number_prior_pregnancy_losses = np.full(
    N_SAMPLE,
    np.nan,
    dtype=float
)

has_observed_prior_loss_data = np.any(
    np.isfinite(prior_loss_matrix),
    axis=0
)

number_prior_pregnancy_losses[
    has_observed_prior_loss_data
] = np.nansum(
    prior_loss_matrix[:, has_observed_prior_loss_data],
    axis=0
)

# 当前妊娠是第一次妊娠，明确设为0
number_prior_pregnancy_losses[
    no_prior_pregnancy
] = 0.0


maximum_ga_prior_pregnancy_loss = np.full(
    N_SAMPLE,
    np.nan,
    dtype=float
)

for sample_index in range(N_SAMPLE):

    if no_prior_pregnancy[sample_index]:
        maximum_ga_prior_pregnancy_loss[sample_index] = 0.0
        continue

    loss_mask = (
        prior_loss_matrix[:, sample_index] == 1
    )

    if np.any(loss_mask):

        loss_ga = prior_loss_ga_matrix[
            loss_mask,
            sample_index
        ]

        loss_ga = loss_ga[
            np.isfinite(loss_ga)
        ]

        if len(loss_ga) > 0:
            maximum_ga_prior_pregnancy_loss[
                sample_index
            ] = np.max(loss_ga)

    elif np.any(
        np.isfinite(prior_loss_matrix[:, sample_index])
    ):
        maximum_ga_prior_pregnancy_loss[
            sample_index
        ] = 0.0


any_prior_cervical_insufficiency = combine_any_binary(
    prior_cervical_insufficiency_list,
    strict_missing=False
)

any_prior_prom = combine_any_binary(
    prior_prom_list,
    strict_missing=False
)

any_prior_abruption = combine_any_binary(
    prior_abruption_list,
    strict_missing=False
)

any_prior_d_and_c = combine_any_binary(
    prior_d_and_c_list,
    strict_missing=False
)

any_prior_cerclage = combine_any_binary(
    prior_cerclage_list,
    strict_missing=False
)

# 第一次妊娠者明确没有既往事件
for history_variable in [
    any_prior_cervical_insufficiency,
    any_prior_prom,
    any_prior_abruption,
    any_prior_d_and_c,
    any_prior_cerclage
]:
    history_variable[no_prior_pregnancy] = 0.0


# ============================================================================
# 5. U1感染、基础疾病及辅助生殖
# ============================================================================

uti_visit_1 = recode_yes_no(
    load_numeric('vxx', 'VXXB01am_V1a')
)

gonorrhea_visit_1 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bd1_V1a')
)

chlamydia_visit_1 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bd2_V1a')
)

trichomonas_visit_1 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bd8_V1a')
)

bacterial_vaginosis_visit_1 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bf_V1a')
)

genital_infection_visit_1 = combine_any_binary(
    [
        gonorrhea_visit_1,
        chlamydia_visit_1,
        trichomonas_visit_1,
        bacterial_vaginosis_visit_1
    ]
)

ivf_current_pregnancy = recode_yes_no(
    load_numeric('s02', 'S02C02')
)

artificial_insemination = recode_yes_no(
    load_numeric('s02', 'S02C03')
)

chronic_hypertension = recode_yes_no(
    load_numeric('vxx', 'VXXB01aa_V1a')
)

kidney_disease = recode_yes_no(
    load_numeric('vxx', 'VXXB01al_V1a')
)

fibroids = recode_yes_no(
    load_numeric('vxx', 'VXXB01bc2_V1a')
)


# ============================================================================
# 6. U1解释性名称列表
# 保留你要求的原始顺序
# ============================================================================

PTB_add_U1 = [
    'Cramping at visit 1',
    'Prior cervical cone',
    'Prior cervical LEEP',
    'Prior cervical cryotherapy',
    'Number of prior pregnancy losses',
    'Maximum GA of prior pregnancy loss',
    'Prior cervical insufficiency',
    'Prior premature rupture of membranes',
    'Prior placental abruption',
    'Prior D&C or D&E',
    'Prior cervical cerclage',
    'Urinary tract infection at visit 1',
    'Genital tract infection at visit 1',
    'IVF for current pregnancy',
    'Artificial insemination for current pregnancy',
    'Chronic hypertension',
    'Kidney disease',
    'Fibroids'
]


factor_add_U1 = pd.DataFrame(
    {
        'Cramping at visit 1':
            cramping_visit_1,

        'Prior cervical cone':
            prior_cervical_cone,

        'Prior cervical LEEP':
            prior_cervical_leep,

        'Prior cervical cryotherapy':
            prior_cervical_cryotherapy,

        'Number of prior pregnancy losses':
            number_prior_pregnancy_losses,

        'Maximum GA of prior pregnancy loss':
            maximum_ga_prior_pregnancy_loss,

        'Prior cervical insufficiency':
            any_prior_cervical_insufficiency,

        'Prior premature rupture of membranes':
            any_prior_prom,

        'Prior placental abruption':
            any_prior_abruption,

        'Prior D&C or D&E':
            any_prior_d_and_c,

        'Prior cervical cerclage':
            any_prior_cerclage,

        'Urinary tract infection at visit 1':
            uti_visit_1,

        'Genital tract infection at visit 1':
            genital_infection_visit_1,

        'IVF for current pregnancy':
            ivf_current_pregnancy,

        'Artificial insemination for current pregnancy':
            artificial_insemination,

        'Chronic hypertension':
            chronic_hypertension,

        'Kidney disease':
            kidney_disease,

        'Fibroids':
            fibroids
    }
)[PTB_add_U1]


# ============================================================================
# 7. U2新增特征
# ============================================================================

contractions_visit_2 = recode_yes_no(
    load_numeric('v2a', 'V2AD02')
)

weeks_since_contractions_visit_2 = load_numeric(
    'v2a',
    'V2AD02a'
)

cervical_funnel_visit_2 = recode_present_absent(
    load_numeric('u2b', 'U2BB03')
)

cervical_debris_visit_2 = recode_yes_no(
    load_numeric('u2b', 'U2BB05')
)

family_history_sptb = recode_yes_no(
    load_numeric('v2a', 'V2AE06c')
)

family_history_pprom = recode_yes_no(
    load_numeric('v2a', 'V2AE06d')
)

uti_visit_2 = recode_yes_no(
    load_numeric('vxx', 'VXXB01am_V2a')
)

gonorrhea_visit_2 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bd1_V2a')
)

chlamydia_visit_2 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bd2_V2a')
)

trichomonas_visit_2 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bd8_V2a')
)

bacterial_vaginosis_visit_2 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bf_V2a')
)

genital_infection_visit_2 = combine_any_binary(
    [
        gonorrhea_visit_2,
        chlamydia_visit_2,
        trichomonas_visit_2,
        bacterial_vaginosis_visit_2
    ]
)

bleeding_days_visit_2 = load_numeric(
    'v2a',
    'V2AD03a'
)


PTB_add_U2 = [
    'Cramping/contractions at visit 2',
    'Weeks since contractions started at visit 2',
    'Cervical funnel at visit 2',
    'Cervical debris at visit 2',
    'Family history of spontaneous PTB',
    'Family history of preterm PROM',
    'Urinary tract infection at visit 2',
    'Genital tract infection at visit 2',
    'Bleeding days at visit 2'
]


factor_add_U2_new = pd.DataFrame(
    {
        'Cramping/contractions at visit 2':
            contractions_visit_2,

        'Weeks since contractions started at visit 2':
            weeks_since_contractions_visit_2,

        'Cervical funnel at visit 2':
            cervical_funnel_visit_2,

        'Cervical debris at visit 2':
            cervical_debris_visit_2,

        'Family history of spontaneous PTB':
            family_history_sptb,

        'Family history of preterm PROM':
            family_history_pprom,

        'Urinary tract infection at visit 2':
            uti_visit_2,

        'Genital tract infection at visit 2':
            genital_infection_visit_2,

        'Bleeding days at visit 2':
            bleeding_days_visit_2
    }
)[PTB_add_U2]


# U2模型可以获得U1和U2全部新增变量
PTB_add_U2_all = PTB_add_U1 + PTB_add_U2

factor_add_U2 = pd.concat(
    [
        factor_add_U1,
        factor_add_U2_new
    ],
    axis=1
)[PTB_add_U2_all]


# ============================================================================
# 8. U3新增特征
# ============================================================================

contractions_visit_3 = recode_yes_no(
    load_numeric('v3a', 'V3AD02')
)

weeks_since_contractions_visit_3 = load_numeric(
    'v3a',
    'V3AD02a'
)

cervical_funnel_visit_3 = recode_present_absent(
    load_numeric('u3b', 'U3BB03')
)

cervical_debris_visit_3 = recode_yes_no(
    load_numeric('u3b', 'U3BB05')
)

uti_visit_3 = recode_yes_no(
    load_numeric('vxx', 'VXXB01am_V3a')
)

gonorrhea_visit_3 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bd1_V3a')
)

chlamydia_visit_3 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bd2_V3a')
)

trichomonas_visit_3 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bd8_V3a')
)

bacterial_vaginosis_visit_3 = recode_yes_no(
    load_numeric('vxx', 'VXXB01bf_V3a')
)

genital_infection_visit_3 = combine_any_binary(
    [
        gonorrhea_visit_3,
        chlamydia_visit_3,
        trichomonas_visit_3,
        bacterial_vaginosis_visit_3
    ]
)

bleeding_days_visit_3 = load_numeric(
    'v3a',
    'V3AD03a'
)


# ============================================================================
# 9. U2→U3宫颈变化变量
# ============================================================================

cervical_length_visit_2 = load_numeric(
    'u2b',
    'U2BB02'
)

cervical_length_visit_3 = load_numeric(
    'u3b',
    'U3BB02'
)

ga_week_u2b = load_numeric(
    'u2b',
    'U2BA02_WK'
)

ga_day_u2b = load_numeric(
    'u2b',
    'U2BA02_DY'
)

ga_week_u3b = load_numeric(
    'u3b',
    'U3BA02_WK'
)

ga_day_u3b = load_numeric(
    'u3b',
    'U3BA02_DY'
)

ga_u2b_decimal = ga_week_u2b + ga_day_u2b / 7.0
ga_u3b_decimal = ga_week_u3b + ga_day_u3b / 7.0

ga_interval_u2_u3 = (
    ga_u3b_decimal -
    ga_u2b_decimal
)


# 正值代表宫颈从U2到U3变短
cervical_shortening_u2_u3 = (
    cervical_length_visit_2 -
    cervical_length_visit_3
)

invalid_cervical_change = (
    ~np.isfinite(cervical_length_visit_2) |
    ~np.isfinite(cervical_length_visit_3)
)

cervical_shortening_u2_u3[
    invalid_cervical_change
] = np.nan


cervical_shortening_rate = np.full(
    N_SAMPLE,
    np.nan,
    dtype=float
)

valid_shortening_rate = (
    np.isfinite(cervical_shortening_u2_u3) &
    np.isfinite(ga_interval_u2_u3) &
    (ga_interval_u2_u3 > 0)
)

cervical_shortening_rate[
    valid_shortening_rate
] = (
    cervical_shortening_u2_u3[
        valid_shortening_rate
    ] /
    ga_interval_u2_u3[
        valid_shortening_rate
    ]
)


new_cervical_funnel_visit_3 = binary_transition(
    previous=cervical_funnel_visit_2,
    current=cervical_funnel_visit_3,
    previous_value=0,
    current_value=1
)

new_cervical_debris_visit_3 = binary_transition(
    previous=cervical_debris_visit_2,
    current=cervical_debris_visit_3,
    previous_value=0,
    current_value=1
)

persistent_contractions_u2_u3 = persistent_positive(
    contractions_visit_2,
    contractions_visit_3
)


PTB_add_U3 = [
    'Cramping/contractions at visit 3',
    'Weeks since contractions started at visit 3',
    'Cervical funnel at visit 3',
    'Cervical debris at visit 3',
    'Urinary tract infection at visit 3',
    'Genital tract infection at visit 3',
    'Bleeding days at visit 3',
    'Cervical shortening from visit 2 to visit 3',
    'Cervical shortening rate from visit 2 to visit 3',
    'New cervical funnel at visit 3',
    'New cervical debris at visit 3',
    'Persistent contractions from visit 2 to visit 3'
]


factor_add_U3_new = pd.DataFrame(
    {
        'Cramping/contractions at visit 3':
            contractions_visit_3,

        'Weeks since contractions started at visit 3':
            weeks_since_contractions_visit_3,

        'Cervical funnel at visit 3':
            cervical_funnel_visit_3,

        'Cervical debris at visit 3':
            cervical_debris_visit_3,

        'Urinary tract infection at visit 3':
            uti_visit_3,

        'Genital tract infection at visit 3':
            genital_infection_visit_3,

        'Bleeding days at visit 3':
            bleeding_days_visit_3,

        'Cervical shortening from visit 2 to visit 3':
            cervical_shortening_u2_u3,

        'Cervical shortening rate from visit 2 to visit 3':
            cervical_shortening_rate,

        'New cervical funnel at visit 3':
            new_cervical_funnel_visit_3,

        'New cervical debris at visit 3':
            new_cervical_debris_visit_3,

        'Persistent contractions from visit 2 to visit 3':
            persistent_contractions_u2_u3
    }
)[PTB_add_U3]


PTB_add_U3_all = (
    PTB_add_U1 +
    PTB_add_U2 +
    PTB_add_U3
)

factor_add_U3 = pd.concat(
    [
        factor_add_U1,
        factor_add_U2_new,
        factor_add_U3_new
    ],
    axis=1
)[PTB_add_U3_all]


# 完整的U1-U3新增特征
factor_add = factor_add_U3.copy()


# ============================================================================
# 10. 添加PublicID作为DataFrame index
# ============================================================================

if len(GDM_ID) != len(factor_add):
    raise ValueError(
        'The PTB-specific feature dataframe does not align with GDM_ID: '
        f'{len(factor_add)} versus {len(GDM_ID)} subjects.'
    )

participant_index = np.asarray(
    GDM_ID
).reshape(-1)

for current_dataframe in [
    factor_add_U1,
    factor_add_U2_new,
    factor_add_U2,
    factor_add_U3_new,
    factor_add_U3,
    factor_add
]:
    current_dataframe.index = participant_index
    current_dataframe.index.name = 'PublicID'


# ============================================================================
# 11. 检查
# ============================================================================

print(
    '\nfactor_add_U1 shape:',
    factor_add_U1.shape
)

print(
    'factor_add_U2_new shape:',
    factor_add_U2_new.shape
)

print(
    'factor_add_U2 cumulative shape:',
    factor_add_U2.shape
)

print(
    'factor_add_U3_new shape:',
    factor_add_U3_new.shape
)

print(
    'factor_add_U3 cumulative shape:',
    factor_add_U3.shape
)

print(
    'factor_add complete shape:',
    factor_add.shape
)

print(
    '\nMissing values in factor_add:'
)

print(
    factor_add.isna().sum()
)


# ============================================================================
# 12. 保存DataFrame
# ============================================================================

factor_add_U1.to_csv(
    os.path.join(INTERMEDIATE_OUTPUT_DIR, "factor_add_PTB_U1.csv"),
    index=True
)

factor_add_U2.to_csv(
    os.path.join(INTERMEDIATE_OUTPUT_DIR, "factor_add_PTB_U2_cumulative.csv"),
    index=True
)

factor_add_U3.to_csv(
    os.path.join(INTERMEDIATE_OUTPUT_DIR, "factor_add_PTB_U3_cumulative.csv"),
    index=True
)

factor_add.to_pickle(
    os.path.join(INTERMEDIATE_OUTPUT_DIR, "factor_add_PTB_all_visits.pkl")
)


# ============================================================================
# 13. 转换为模型可使用的numpy矩阵
# ============================================================================

# 样本 × 特征
factor_add_U1_samples_by_features = (
    factor_add_U1.to_numpy(dtype=float)
)

factor_add_U2_samples_by_features = (
    factor_add_U2.to_numpy(dtype=float)
)

factor_add_U3_samples_by_features = (
    factor_add_U3.to_numpy(dtype=float)
)


# 特征 × 样本
# 如果你原来的factor每一行为一个特征，使用下面这些
factor_add_U1_np = (
    factor_add_U1.to_numpy(dtype=float).T
)

factor_add_U2_np = (
    factor_add_U2.to_numpy(dtype=float).T
)

factor_add_U3_np = (
    factor_add_U3.to_numpy(dtype=float).T
)

factor_add_np = factor_add_U3_np


print(
    '\nfactor_add_U1_np:',
    factor_add_U1_np.shape
)

print(
    'factor_add_U2_np:',
    factor_add_U2_np.shape
)

print(
    'factor_add_U3_np:',
    factor_add_U3_np.shape
)


# ============================================================================
# 14. Merge PTB-specific features into the original feature-by-subject matrix
# ============================================================================
# The original factor_all_num matrix is feature x subject. factor_add_U3_np has
# the same orientation and contains all added U1, U2 and U3 variables in
# cumulative order. The visit-specific FEATURE_SETS below determine which rows
# are available to each model.
factor_all_num_base = np.asarray(factor_all_num)
factor_add_all_visits_np = np.asarray(factor_add_U3_np, dtype=float)

if factor_all_num_base.ndim != 2:
    raise ValueError(
        'The original factor_all_num must be a two-dimensional '
        'feature-by-subject matrix.'
    )

if factor_add_all_visits_np.ndim != 2:
    raise ValueError(
        'factor_add_U3_np must be a two-dimensional feature-by-subject matrix.'
    )

if factor_all_num_base.shape[1] != factor_add_all_visits_np.shape[1]:
    raise ValueError(
        'The original and PTB-specific matrices contain different numbers of '
        f'subjects: {factor_all_num_base.shape[1]} versus '
        f'{factor_add_all_visits_np.shape[1]}.'
    )

if factor_add_all_visits_np.shape[0] != len(PTB_add_U3_all):
    raise ValueError(
        'The number of rows in factor_add_U3_np does not match the cumulative '
        f'PTB feature-name list: {factor_add_all_visits_np.shape[0]} versus '
        f'{len(PTB_add_U3_all)}.'
    )

if list(factor_add_U3.columns) != list(PTB_add_U3_all):
    raise ValueError(
        'factor_add_U3 columns do not follow PTB_add_U3_all order.'
    )

overlapping_added_names = sorted(
    set(Name_explain_full_base).intersection(PTB_add_U3_all)
)
if overlapping_added_names:
    raise ValueError(
        'The following PTB-specific feature names already exist in the original '
        'feature matrix: ' + ', '.join(overlapping_added_names)
    )

# Update visit-specific model feature names cumulatively.
Name_explainx_U1 = Name_explainx_U1_base + list(PTB_add_U1)
Name_explainx_U2 = Name_explainx_U2_base + list(PTB_add_U2_all)
Name_explainx_U3 = Name_explainx_U3_base + list(PTB_add_U3_all)

# Update the complete row-name list and physically append the new feature rows.
Name_explain_full = Name_explain_full_base + list(PTB_add_U3_all)
factor_all_num = np.concatenate(
    [factor_all_num_base, factor_add_all_visits_np],
    axis=0,
)

FEATURE_SETS = {
    'U1': Name_explainx_U1,
    'U2': Name_explainx_U2,
    'U3': Name_explainx_U3,
}

print('\nMerged feature matrix checks:')
print('Original feature x subject matrix:', factor_all_num_base.shape)
print('Added PTB feature x subject matrix:', factor_add_all_visits_np.shape)
print('Combined feature x subject matrix:', factor_all_num.shape)
print(
    'Requested U1/U2/U3 feature counts:',
    len(Name_explainx_U1),
    len(Name_explainx_U2),
    len(Name_explainx_U3),
)
# -----------------------------------------------------------------------------
# Reproducibility and analysis settings
# -----------------------------------------------------------------------------
# This replaces: for GGamma in range(20)
# Change this single value when a different fixed split is required.
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
    "PTB_LR_nestedCV_performance_metrics",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

alphas = np.linspace(0.0, 1.0, 200)

# -----------------------------------------------------------------------------
# Subject-level trajectory analysis settings
# -----------------------------------------------------------------------------
# 90% confidence (alpha=0.10) is the primary operating point.
# 80% confidence (alpha=0.20) is retained as a prespecified sensitivity analysis. Add/remove values here without changing the analysis functions.
TRAJECTORY_CONFIDENCE_LEVELS = (0.95, 0.90, 0.85, 0.80)
PRIMARY_TRAJECTORY_CONFIDENCE = 0.90
TRAJECTORY_STATE_ORDER = ["L", "R", "H"]
RAW_CP_STATE_ORDER = ["L", "R", "H", "E"]
MIN_GROUP_SIZE_FOR_INTERPRETATION = 20

# Preserve the original feature-by-subject matrix before selecting U1/U2/U3.
factor_all_num_full = np.asarray(factor_all_num)
label1 = np.asarray(PTB_relabel0, dtype=int)

if factor_all_num_full.ndim != 2:
    raise ValueError("factor_all_num must be a two-dimensional feature-by-subject array.")

if factor_all_num_full.shape[0] != len(Name_explain_full):
    raise ValueError(
        "The number of rows in factor_all_num does not match Name_explain_full: "
        f"{factor_all_num_full.shape[0]} versus {len(Name_explain_full)}."
    )

if len(label1) != len(PTB_index):
    raise ValueError(
        "The PTB label length does not match PTB_index: "
        f"{len(label1)} versus {len(PTB_index)}."
    )


# =============================================================================
# PTB subtype alignment and U3 landmark safety mechanism
# =============================================================================
# The all-cause PTB label continues to be determined exclusively from GA_WK.
# A09A03a1 is used only after a pregnancy has been classified as PTB:
#   1 = spontaneous PTB (sPTB)
#   2 = medically indicated PTB (iPTB)
# Missing/other codes among PTB pregnancies are retained as unclassified PTB.
# Term births remain term births irrespective of A09A03a1 missingness.
PTB_SUBTYPE_TERM = "Term birth"
PTB_SUBTYPE_SPTB = "sPTB"
PTB_SUBTYPE_IPTB = "iPTB"
PTB_SUBTYPE_OTHER = "Other/unclassified PTB"
PTB_SUBTYPE_ORDER = [
    PTB_SUBTYPE_TERM,
    PTB_SUBTYPE_SPTB,
    PTB_SUBTYPE_IPTB,
    PTB_SUBTYPE_OTHER,
]

ptb_subtype_raw_full = np.asarray(PTB_A09_raw_sub, dtype=float)
if len(ptb_subtype_raw_full) != len(GDM_ID):
    raise ValueError(
        "PTB_A09_raw_sub does not align with GDM_ID: "
        f"{len(ptb_subtype_raw_full)} versus {len(GDM_ID)}."
    )

PTB_subtype = np.full(len(label1), PTB_SUBTYPE_TERM, dtype=object)
PTB_subtype_raw_aligned = ptb_subtype_raw_full[np.asarray(PTB_index, dtype=int)]

for local_index in range(len(label1)):
    if label1[local_index] == 0:
        PTB_subtype[local_index] = PTB_SUBTYPE_TERM
    elif PTB_subtype_raw_aligned[local_index] == 1.0:
        PTB_subtype[local_index] = PTB_SUBTYPE_SPTB
    elif PTB_subtype_raw_aligned[local_index] == 2.0:
        PTB_subtype[local_index] = PTB_SUBTYPE_IPTB
    else:
        PTB_subtype[local_index] = PTB_SUBTYPE_OTHER

# Align gestational age with the master PTB/term cohort. Completed weeks plus
# recorded days are used when days are valid; otherwise completed weeks are
# retained. Delivery before 30+0 weeks is equivalent to completed GA <=29 weeks.
GA_WK_aligned = np.asarray(GA_WK, dtype=float)[np.asarray(PTB_index, dtype=int)]
GA_DAYS_aligned = np.asarray(GA_DAYS, dtype=float)[np.asarray(PTB_index, dtype=int)]
GA_total_weeks_aligned = GA_WK_aligned.astype(float).copy()
valid_ga_days_aligned = (
    np.isfinite(GA_DAYS_aligned)
    & (GA_DAYS_aligned >= 0)
    & (GA_DAYS_aligned <= 6)
)
GA_total_weeks_aligned[valid_ga_days_aligned] = (
    GA_WK_aligned[valid_ga_days_aligned]
    + GA_DAYS_aligned[valid_ga_days_aligned] / 7.0
)

U3_LANDMARK_START_WEEKS = 30.0
U3_delivered_before_landmark = GA_total_weeks_aligned < U3_LANDMARK_START_WEEKS
U3_landmark_eligible = ~U3_delivered_before_landmark

# D is a structural state, not a conformal prediction. It means that the
# pregnancy had already ended before the conservative U3 landmark and was
# therefore excluded from all U3 training, calibration and testing operations.
DELIVERED_STATE = "D"
TRAJECTORY_STATE_ORDER = ["L", "R", "H", DELIVERED_STATE]
RAW_CP_STATE_ORDER = ["L", "R", "H", "E", DELIVERED_STATE]

subtype_counts = pd.Series(PTB_subtype).value_counts().reindex(
    PTB_SUBTYPE_ORDER,
    fill_value=0,
)
early_subtype_counts = pd.Series(
    PTB_subtype[U3_delivered_before_landmark]
).value_counts().reindex(PTB_SUBTYPE_ORDER, fill_value=0)

print("\nPTB subtype classification based on GA_WK followed by A09A03a1:")
print(subtype_counts.to_string())
print(
    "U3 landmark exclusion (delivery before 30+0 weeks / completed GA <=29):",
    int(U3_delivered_before_landmark.sum()),
    "/",
    len(label1),
)
print("Subtype distribution among pregnancies excluded before U3:")
print(early_subtype_counts.to_string())


def build_raw_feature_dataframe(feature_names):
    """Select one visit-specific feature set and return subjects x features."""
    missing_names = [name for name in feature_names if name not in Name_explain_full]
    if missing_names:
        raise ValueError(
            "The following requested features are absent from Name_explain_full: "
            + ", ".join(missing_names)
        )

    # This intentionally follows the original Name_explain.index(item) logic.
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
    selected = selected[:, PTB_index].T

    return pd.DataFrame(selected, columns=feature_names, dtype=float)


def fit_preprocessor(X_train):
    """Fit manuscript-specified preprocessing using training data only.

    The source pipeline's analysis matrix is retained. Predictors are KNN-imputed
    (k=5), then scaled using training-set 5th/95th percentiles and clipped to
    [0, 1]. Variables with >20% missingness are excluded. No correlation-based
    or other additional feature-selection step is applied here.
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
    """Return sqrt(n_negative / n_positive) from the current training data."""
    y = np.asarray(y, dtype=int)
    n_positive = int(np.sum(y == 1))
    n_negative = int(np.sum(y == 0))

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "Both classes are required to calculate the square-root class weight."
        )

    return float(np.sqrt(n_negative / n_positive))


def make_binary_sample_weight(y, positive_weight):
    """Create per-subject weights with class 0=1 and class 1=positive_weight."""
    y = np.asarray(y, dtype=int)
    sample_weight = np.ones(len(y), dtype=float)
    sample_weight[y == 1] = float(positive_weight)
    return sample_weight


def make_logistic_model(positive_weight):
    """Elastic-net logistic-regression baseline with square-root class weighting."""
    return LogisticRegression(
        C=0.1,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.3,
        class_weight={0: 1.0, 1: float(positive_weight)},
        max_iter=50000,
        random_state=FIXED_GAMMA,
    )


def make_catboost_model(positive_weight):
    """Conservative CatBoost classifier for the structured PTB feature matrix."""
    if CatBoostClassifier is None:
        raise ImportError(
            "CatBoost is required for the three-model comparison. "
            "Install it in the active environment with: pip install catboost"
        )

    return CatBoostClassifier(
        iterations=600,
        depth=4,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1.0, float(positive_weight)],
        l2_leaf_reg=5.0,
        random_seed=FIXED_GAMMA,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def make_ebm_model():
    """Explainable Boosting Machine with a limited interaction budget."""
    if ExplainableBoostingClassifier is None:
        raise ImportError(
            "InterpretML is required for the EBM comparison. "
            "Install it in the active environment with: pip install interpret"
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
    """Create one of the models enabled by MODEL_SELECTION."""
    model_key = str(model_key).upper()
    if model_key == "LR":
        return make_logistic_model(positive_weight)
    if model_key == "EBM":
        return make_ebm_model()
    raise ValueError(f"Unsupported model key: {model_key!r}.")


def fit_selected_model(model, model_key, X, y, positive_weight):
    """Fit LR or EBM with the same class-weighting convention."""
    model_key = str(model_key).upper()
    if model_key == "LR":
        model.fit(X, y)
        return model
    if model_key == "EBM":
        model.fit(
            X,
            y,
            sample_weight=make_binary_sample_weight(y, positive_weight),
        )
        return model
    raise ValueError(f"Unsupported model key: {model_key!r}.")


def validate_model_dependencies():
    """Fail early only when a selected model dependency is unavailable."""
    if "EBM" in ACTIVE_MODEL_KEYS and ExplainableBoostingClassifier is None:
        raise ImportError(
            "MODEL_SELECTION requires EBM, but InterpretML is unavailable. "
            "Install it with: pip install interpret, or set MODEL_SELECTION = 'LR'."
        )

def bootstrap_auc_ci(
    y_true,
    y_score,
    n_bootstrap=BOOTSTRAP_ITERATIONS,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    """Calculate a percentile 95% CI by bootstrapping the final OOF predictions."""
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

        # AUC is undefined if a bootstrap sample contains only one class.
        if np.unique(y_boot).size < 2:
            continue

        bootstrap_aucs.append(
            metrics.roc_auc_score(y_boot, y_score[sample_index])
        )

    if not bootstrap_aucs:
        raise RuntimeError("No valid bootstrap sample contained both outcome classes.")

    lower_ci, upper_ci = np.percentile(bootstrap_aucs, [2.5, 97.5])
    return float(lower_ci), float(upper_ci), np.asarray(bootstrap_aucs)


def calculate_subtype_auc_from_mixed_model(
    full_predictions,
    analysis_mask,
):
    """
    Evaluate sPTB and iPTB using predictions from the single model trained on
    all PTB cases versus term birth.

    For each subtype, the positive group is that subtype and the negative group
    is term births. The other PTB subtype and unclassified PTB cases are omitted.
    No subtype-specific model is retrained.
    """
    full_predictions = np.asarray(full_predictions, dtype=float)
    analysis_mask = np.asarray(analysis_mask, dtype=bool)

    subtype_results = {}
    subtype_definitions = [
        ("sPTB vs term birth", PTB_SUBTYPE_SPTB),
        ("iPTB vs term birth", PTB_SUBTYPE_IPTB),
    ]

    for evaluation_name, subtype_name in subtype_definitions:
        evaluation_mask = (
            analysis_mask
            & (
                (PTB_subtype == PTB_SUBTYPE_TERM)
                | (PTB_subtype == subtype_name)
            )
        )
        y_subtype = (PTB_subtype[evaluation_mask] == subtype_name).astype(int)
        scores_subtype = full_predictions[evaluation_mask]

        if np.unique(y_subtype).size < 2:
            subtype_results[evaluation_name] = {
                "subtype": subtype_name,
                "n": int(evaluation_mask.sum()),
                "n_positive": int(y_subtype.sum()),
                "n_negative": int((y_subtype == 0).sum()),
                "roc_auc": np.nan,
                "auc_ci_lower": np.nan,
                "auc_ci_upper": np.nan,
                "bootstrap_aucs": np.asarray([], dtype=float),
            }
            continue

        auc_value = metrics.roc_auc_score(y_subtype, scores_subtype)
        ci_lower, ci_upper, bootstrap_aucs = bootstrap_auc_ci(
            y_subtype,
            scores_subtype,
        )
        subtype_results[evaluation_name] = {
            "subtype": subtype_name,
            "n": int(evaluation_mask.sum()),
            "n_positive": int(y_subtype.sum()),
            "n_negative": int((y_subtype == 0).sum()),
            "roc_auc": float(auc_value),
            "auc_ci_lower": ci_lower,
            "auc_ci_upper": ci_upper,
            "bootstrap_aucs": bootstrap_aucs,
        }

    return subtype_results




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

def calculate_mondrian_cp_rates(
    model_calibrated,
    X_calib,
    y_calib,
    X_test,
    y_test,
):
    """Retain the original Mondrian conformal-prediction calculation."""
    calib_class0 = y_calib == 0
    calib_class1 = y_calib == 1

    if np.any(calib_class0):
        scores_calib0 = (
            1.0
            - model_calibrated.predict_proba(X_calib.loc[calib_class0])[:, 0]
        )
    else:
        scores_calib0 = np.array([])

    if np.any(calib_class1):
        scores_calib1 = (
            1.0
            - model_calibrated.predict_proba(X_calib.loc[calib_class1])[:, 1]
        )
    else:
        scores_calib1 = np.array([])

    proba_test = model_calibrated.predict_proba(X_test)[:, 1]

    fold_single_correct_rates = []
    fold_single_error_rates = []
    fold_multi_pred_rates = []
    fold_empty_pred_rates = []
    positive_accuracy = []
    neg_error_rate = []

    for alpha in alphas:
        quantile_class0 = (
            finite_sample_conformal_quantile(scores_calib0, alpha)
            if len(scores_calib0) > 0
            else 1.0
        )
        quantile_class1 = (
            finite_sample_conformal_quantile(scores_calib1, alpha)
            if len(scores_calib1) > 0
            else 1.0
        )

        single_correct = 0
        single_error = 0
        multi_pred = 0
        empty_pred = 0
        pred_sets = []

        for probability_1, true_label in zip(proba_test, y_test):
            probability_0 = 1.0 - probability_1
            pred_set = []

            if (1.0 - probability_0) <= quantile_class0:
                pred_set.append(0)
            if (1.0 - probability_1) <= quantile_class1:
                pred_set.append(1)

            if len(pred_set) == 1:
                if pred_set[0] == true_label:
                    single_correct += 1
                else:
                    single_error += 1
            elif len(pred_set) == 0:
                empty_pred += 1
            else:
                multi_pred += 1

            pred_sets.append(pred_set)

        total_samples = len(y_test)
        fold_single_correct_rates.append(single_correct / total_samples)
        fold_single_error_rates.append(single_error / total_samples)
        fold_multi_pred_rates.append(multi_pred / total_samples)
        fold_empty_pred_rates.append(empty_pred / total_samples)

        positive_mask = y_test == 1
        negative_mask = y_test == 0

        positive_single_correct = sum(
            len(pred_sets[index]) == 1 and pred_sets[index][0] == 1
            for index in np.where(positive_mask)[0]
        )
        negative_single_as_positive = sum(
            len(pred_sets[index]) == 1 and pred_sets[index][0] == 1
            for index in np.where(negative_mask)[0]
        )

        positive_accuracy.append(
            positive_single_correct / positive_mask.sum()
            if positive_mask.sum() > 0
            else np.nan
        )
        neg_error_rate.append(
            negative_single_as_positive / negative_mask.sum()
            if negative_mask.sum() > 0
            else np.nan
        )

    return {
        "single_correct": np.asarray(fold_single_correct_rates),
        "single_error": np.asarray(fold_single_error_rates),
        "multiple": np.asarray(fold_multi_pred_rates),
        "empty": np.asarray(fold_empty_pred_rates),
        "positive_accuracy": np.asarray(positive_accuracy),
        "negative_error": np.asarray(neg_error_rate),
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


def plot_conformal_result(result):
    """Display the retained conformal-prediction stack plot without saving it."""
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

    stacked = np.vstack(
        [
            result["cp_single_correct"],
            result["cp_single_error"],
            result["cp_multiple"],
            result["cp_empty"],
        ]
    )
    cumulative = np.cumsum(stacked, axis=0)

    for curve in cumulative:
        ax.plot(alphas, curve, color="white", linewidth=1.5)

    optimal_alpha = alphas[np.argmax(result["cp_single_correct"])]
    ax.axvline(
        x=optimal_alpha,
        color="white",
        linestyle="--",
        linewidth=1.5,
    )
    ax.text(
        optimal_alpha + 0.02,
        0.95,
        f"{optimal_alpha:.2f}",
        color="black",
        fontsize=15,
    )

    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(
        f"{PRIMARY_MODEL_DISPLAY} Conformal Prediction ({result['visit']}, no isotonic)",
        fontsize=14,
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Significance (1 - Confidence)", fontsize=12)
    ax.set_ylabel("Fraction", fontsize=12)
    plt.tight_layout()
    plt.show()



def _safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def _wilson_interval(event_n, total_n, z=1.959963984540054):
    """Wilson 95% interval for a binomial event rate."""
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


# Landmark-aware trajectory functions are defined below before execution.

def build_aligned_co_outcomes():
    """
    Build descriptive non-PTB outcomes aligned to the PTB trajectory cohort.

    The main modelling outcome is all-cause PTB based on gestational age at
    delivery. LGA90 is retained as a fetal-growth co-outcome. For CMAE08,
    code 1 is treated as FGR, code 2 as no FGR, and other/missing codes remain
    NaN.
    """
    full_index = np.asarray(PTB_index, dtype=int)

    fgr_raw = np.asarray(fgr, dtype=float)
    fgr_binary = np.full(fgr_raw.shape, np.nan, dtype=float)
    fgr_binary[fgr_raw == 1.0] = 1.0
    fgr_binary[fgr_raw == 2.0] = 0.0

    co_outcomes = {
        "GDM": np.asarray(GDM_glm, dtype=float)[full_index],
        "PE": np.asarray(PE_label, dtype=float)[full_index],
        "FGR": fgr_binary[full_index],
        "LGA90": np.asarray(LGA90_ARIS, dtype=float)[full_index],
    }

    for outcome_name, values in co_outcomes.items():
        if len(values) != len(label1):
            raise ValueError(
                f"Co-outcome {outcome_name} does not align with the PTB cohort."
            )

    return co_outcomes


# Landmark-aware co-outcome and print functions are defined below.

# =============================================================================
# Landmark-aware overrides for U3 and subtype-aware trajectory analyses
# =============================================================================


def _make_stratified_master_splits(analysis_mask):
    """Create stratified CV splits and map them back to master cohort indices."""
    analysis_mask = np.asarray(analysis_mask, dtype=bool)
    eligible_indices = np.flatnonzero(analysis_mask)
    y_eligible = label1[eligible_indices]

    class_counts = np.bincount(y_eligible, minlength=2)
    if np.any(class_counts < N_SPLITS):
        raise ValueError(
            "Each class must contain at least N_SPLITS subjects in the visit-specific "
            f"risk set. Counts={class_counts.tolist()}, N_SPLITS={N_SPLITS}."
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


def run_visit_model(visit_name, feature_names, cv_splits, analysis_mask=None):
    """
    Run one visit-specific analysis while preserving master-cohort alignment.

    U1 and U2 use the full PTB/term cohort. U3 uses only pregnancies delivered
    at 30+0 weeks or later. Excluded U3 subjects retain master-array positions,
    receive NaN probabilities and the structural state D.
    """
    X_raw = build_raw_feature_dataframe(feature_names)

    if len(X_raw) != len(label1):
        raise ValueError(
            f"{visit_name}: feature rows and labels differ: "
            f"{len(X_raw)} versus {len(label1)}."
        )

    if analysis_mask is None:
        analysis_mask = np.ones(len(label1), dtype=bool)
    analysis_mask = np.asarray(analysis_mask, dtype=bool)
    if analysis_mask.shape != (len(label1),):
        raise ValueError(
            f"{visit_name}: analysis_mask must have length {len(label1)}."
        )

    excluded_mask = ~analysis_mask
    model_names = list(ACTIVE_MODEL_DISPLAY_NAMES)
    oof_predictions_by_model = {
        model_name: np.full(len(label1), np.nan, dtype=float)
        for model_name in model_names
    }
    oof_cp_probability = np.full(len(label1), np.nan, dtype=float)
    oof_fold_id = np.full(len(label1), np.nan, dtype=float)
    oof_score_state = {
        float(c): np.full(len(label1), "", dtype="<U1")
        for c in TRAJECTORY_CONFIDENCE_LEVELS
    }
    oof_cp_state = {
        float(confidence): np.full(len(label1), "", dtype="<U1")
        for confidence in TRAJECTORY_CONFIDENCE_LEVELS
    }
    oof_cp_raw_state = {
        float(confidence): np.full(len(label1), "", dtype="<U1")
        for confidence in TRAJECTORY_CONFIDENCE_LEVELS
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
                f"{visit_name}: a CV fold contains a subject outside its landmark risk set."
            )

        print(
            f"{visit_name}: Processing Fold {fold_number}/{N_SPLITS} "
            f"(train={len(train_index)}, test={len(test_index)})"
        )

        X_train_raw = X_raw.iloc[train_index]
        X_test_raw = X_raw.iloc[test_index]
        y_train = label1[train_index]
        y_test = label1[test_index]

        # AUC models use the complete visit-specific outer-training fold.
        X_train_auc, auc_preprocessor = fit_preprocessor(X_train_raw)
        X_test_auc = transform_preprocessor(X_test_raw, auc_preprocessor)
        features_after_preprocessing.append(X_train_auc.shape[1])

        positive_weight_auc = calculate_sqrt_positive_weight(y_train)
        fold_positive_weights.append(positive_weight_auc)

        for model_key in ACTIVE_MODEL_KEYS:
            model_display = MODEL_DISPLAY_NAMES[model_key]
            auc_model = make_selected_model(model_key, positive_weight_auc)
            fit_selected_model(
                auc_model,
                model_key,
                X_train_auc,
                y_train,
                positive_weight_auc,
            )
            oof_predictions_by_model[model_display][test_index] = (
                auc_model.predict_proba(X_test_auc)[:, 1]
            )

        # Proper-training/calibration split for Mondrian conformal prediction.
        # The calibration labels are used only to form class-specific conformal
        # scores; no isotonic probability-calibration model is fitted.
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
        fit_selected_model(
            cp_model,
            PRIMARY_MODEL_KEY,
            X_proper,
            y_proper,
            positive_weight_cp,
        )

        fold_cp = calculate_mondrian_cp_rates(
            model_calibrated=cp_model,
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
            raise RuntimeError(
                f"{visit_name}: {model_name} has missing OOF predictions in the risk set."
            )
        if np.any(~np.isnan(predictions[excluded_mask])):
            raise RuntimeError(
                f"{visit_name}: excluded subjects unexpectedly received model predictions."
            )

    if np.isnan(oof_cp_probability[analysis_mask]).any():
        raise RuntimeError(
            f"{visit_name}: some eligible subjects have no CP probability."
        )
    if np.any(~np.isnan(oof_cp_probability[excluded_mask])):
        raise RuntimeError(
            f"{visit_name}: excluded subjects unexpectedly received CP probabilities."
        )

    for confidence in TRAJECTORY_CONFIDENCE_LEVELS:
        confidence = float(confidence)
        if np.any(oof_cp_state[confidence][analysis_mask] == ""):
            raise RuntimeError(
                f"{visit_name}: missing CP state within the visit risk set at "
                f"confidence={confidence:.2f}."
            )
        if np.any(excluded_mask) and np.any(
            oof_cp_state[confidence][excluded_mask] != DELIVERED_STATE
        ):
            raise RuntimeError(
                f"{visit_name}: excluded subjects were not marked as D."
            )

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
            raise RuntimeError(f"{visit_name}: excluded subjects were not marked as D in score states.")

    y_evaluation = label1[analysis_mask]
    model_auc_results = {}
    for model_name, full_predictions in oof_predictions_by_model.items():
        predictions = full_predictions[analysis_mask]
        fpr, tpr, threshold = metrics.roc_curve(y_evaluation, predictions)
        auc_value = metrics.auc(fpr, tpr)
        ci_lower, ci_upper, bootstrap_aucs = bootstrap_auc_ci(
            y_evaluation,
            predictions,
        )
        subtype_auc_results = calculate_subtype_auc_from_mixed_model(
            full_predictions=full_predictions,
            analysis_mask=analysis_mask,
        )
        model_auc_results[model_name] = {
            "roc_auc": float(auc_value),
            "auc_ci_lower": ci_lower,
            "auc_ci_upper": ci_upper,
            "oof_prediction": full_predictions,
            "evaluation_prediction": predictions,
            "evaluation_label": y_evaluation,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": threshold,
            "bootstrap_aucs": bootstrap_aucs,
            "subtype_auc_results": subtype_auc_results,
        }
        print(
            f"{visit_name} {model_name} all-PTB AUC = {auc_value:.4f} "
            f"(bootstrap 95% CI {ci_lower:.4f}-{ci_upper:.4f}; "
            f"risk-set N={analysis_mask.sum()})"
        )
        for subtype_evaluation, subtype_result in subtype_auc_results.items():
            print(
                f"  {subtype_evaluation}, using the same all-PTB OOF scores: "
                f"AUC={subtype_result['roc_auc']:.4f} "
                f"(95% CI {subtype_result['auc_ci_lower']:.4f}-"
                f"{subtype_result['auc_ci_upper']:.4f}; "
                f"N={subtype_result['n']}, "
                f"events={subtype_result['n_positive']})"
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


def classify_trajectory_group(states):
    """Collapse a complete landmark trajectory into interpretable groups."""
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
    """Link U1/U2/U3 decisions on the unchanged master subject index."""
    confidence = float(confidence)
    subject_ids = np.asarray(PTB_ID_relabel0, dtype=object)
    if len(subject_ids) != len(label1):
        raise ValueError(
            "PTB_ID_relabel0 does not align with the master PTB label vector."
        )

    trajectory_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "PTB": label1.astype(int),
            "PTB_subtype": PTB_subtype,
            "PTB_subtype_raw": PTB_subtype_raw_aligned,
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
        raise RuntimeError(
            "The U3 D markers do not exactly match the landmark exclusion mask."
        )

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

    full_path_mask = trajectory_df["U3_landmark_eligible"].to_numpy()
    changed_any = np.full(len(trajectory_df), np.nan, dtype=float)
    number_of_changes = np.full(len(trajectory_df), np.nan, dtype=float)
    changed_any[full_path_mask] = (
        ~(
            (trajectory_df.loc[full_path_mask, "U1_state"].to_numpy()
             == trajectory_df.loc[full_path_mask, "U2_state"].to_numpy())
            &
            (trajectory_df.loc[full_path_mask, "U2_state"].to_numpy()
             == trajectory_df.loc[full_path_mask, "U3_state"].to_numpy())
        )
    ).astype(float)
    number_of_changes[full_path_mask] = (
        (trajectory_df.loc[full_path_mask, "U1_state"].to_numpy()
         != trajectory_df.loc[full_path_mask, "U2_state"].to_numpy()).astype(int)
        +
        (trajectory_df.loc[full_path_mask, "U2_state"].to_numpy()
         != trajectory_df.loc[full_path_mask, "U3_state"].to_numpy()).astype(int)
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


def _trajectory_subset_masks(trajectory_df):
    return {
        "All samples": pd.Series(True, index=trajectory_df.index),
        "PTB-positive only": trajectory_df["PTB"] == 1,
        "PTB-negative only": trajectory_df["PTB"] == 0,
        "sPTB only": trajectory_df["PTB_subtype"] == PTB_SUBTYPE_SPTB,
        "iPTB only": trajectory_df["PTB_subtype"] == PTB_SUBTYPE_IPTB,
        "Other/unclassified PTB only": (
            trajectory_df["PTB_subtype"] == PTB_SUBTYPE_OTHER
        ),
    }


def summarize_state_prevalence(trajectory_df):
    """Report D explicitly while keeping an at-risk denominator for L/R/H."""
    rows = []
    for subset_name, subset_mask in _trajectory_subset_masks(trajectory_df).items():
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
                            if state != DELIVERED_STATE
                            else np.nan
                        ),
                        "Empty raw sets": int((raw_states == "E").sum())
                        if state == "R"
                        else 0,
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
        y = trajectory_df.loc[at_risk, "PTB"].astype(int)
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


def summarize_early_delivery_before_u3(trajectory_df):
    early = trajectory_df.loc[trajectory_df["delivered_before_U3"]].copy()
    if early.empty:
        return pd.DataFrame(), pd.DataFrame()

    subtype_table = (
        early.groupby("PTB_subtype", dropna=False)
        .size()
        .reindex(PTB_SUBTYPE_ORDER, fill_value=0)
        .rename("N")
        .reset_index()
    )
    subtype_table["Percent"] = 100.0 * subtype_table["N"] / len(early)

    path_table = (
        early.groupby(["PTB_subtype", "U1_U2_trajectory"], dropna=False)
        .size()
        .rename("N")
        .reset_index()
        .sort_values(["PTB_subtype", "N"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return subtype_table, path_table


def summarize_exact_trajectories(
    trajectory_df,
    positive_only=False,
    subtype=None,
    require_u3_eligibility=True,
):
    analysis_df = trajectory_df
    if require_u3_eligibility:
        analysis_df = analysis_df.loc[analysis_df["U3_landmark_eligible"]]
    if positive_only:
        analysis_df = analysis_df.loc[analysis_df["PTB"] == 1]
    if subtype is not None:
        analysis_df = analysis_df.loc[analysis_df["PTB_subtype"] == subtype]

    total_n = len(analysis_df)
    reference_df = trajectory_df.loc[trajectory_df["U3_landmark_eligible"]]
    overall_event_rate = float(reference_df["PTB"].mean())
    rows = []
    for trajectory, group in analysis_df.groupby("trajectory", sort=False):
        n_group = len(group)
        event_n = int(group["PTB"].sum())
        event_rate = _safe_rate(event_n, n_group)
        ci_lower, ci_upper = _wilson_interval(event_n, n_group)
        rows.append(
            {
                "Trajectory": trajectory,
                "N": n_group,
                "Percent": 100.0 * _safe_rate(n_group, total_n),
                "PTB events": event_n,
                "PTB event rate": event_rate,
                "Event-rate CI lower": ci_lower,
                "Event-rate CI upper": ci_upper,
                "Enrichment vs U3 risk set": (
                    event_rate / overall_event_rate
                    if overall_event_rate > 0 and not np.isnan(event_rate)
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["N", "PTB event rate"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_trajectory_groups(trajectory_df):
    analysis_df = trajectory_df.loc[trajectory_df["U3_landmark_eligible"]]
    overall_event_rate = float(analysis_df["PTB"].mean())
    rows = []
    for group_name, group in analysis_df.groupby("trajectory_group", sort=False):
        n_group = len(group)
        event_n = int(group["PTB"].sum())
        event_rate = _safe_rate(event_n, n_group)
        ci_lower, ci_upper = _wilson_interval(event_n, n_group)
        rows.append(
            {
                "Trajectory group": group_name,
                "N": n_group,
                "Percent of U3 risk set": 100.0 * n_group / len(analysis_df),
                "PTB events": event_n,
                "PTB event rate": event_rate,
                "Event-rate CI lower": ci_lower,
                "Event-rate CI upper": ci_upper,
                "Enrichment vs U3 risk set": (
                    event_rate / overall_event_rate
                    if overall_event_rate > 0 and not np.isnan(event_rate)
                    else np.nan
                ),
                "Interpret cautiously": n_group < MIN_GROUP_SIZE_FOR_INTERPRETATION,
            }
        )
    return pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)


def summarize_subtype_composition_by_trajectory(trajectory_df):
    """Compare sPTB and iPTB within each clinical trajectory group."""
    group_order = [
        "Delivered before U3",
        "Stable low",
        "Resolved to low",
        "Persistent review",
        "Review at Visit 3",
        "High de-escalated",
        "Emerging high",
        "Persistent high",
        "Other/reversal",
    ]
    rows = []
    for group_name in group_order:
        group = trajectory_df.loc[trajectory_df["trajectory_group"] == group_name]
        if group.empty:
            continue
        ptb = group.loc[group["PTB"] == 1]
        s_n = int((ptb["PTB_subtype"] == PTB_SUBTYPE_SPTB).sum())
        i_n = int((ptb["PTB_subtype"] == PTB_SUBTYPE_IPTB).sum())
        other_n = int((ptb["PTB_subtype"] == PTB_SUBTYPE_OTHER).sum())
        ptb_n = len(ptb)
        rows.append(
            {
                "Trajectory group": group_name,
                "N": len(group),
                "PTB events": ptb_n,
                "PTB event rate": _safe_rate(ptb_n, len(group)),
                "sPTB": s_n,
                "iPTB": i_n,
                "Other/unclassified PTB": other_n,
                "sPTB among PTB (%)": 100.0 * _safe_rate(s_n, ptb_n),
                "iPTB among PTB (%)": 100.0 * _safe_rate(i_n, ptb_n),
                "Other among PTB (%)": 100.0 * _safe_rate(other_n, ptb_n),
            }
        )
    return pd.DataFrame(rows)


def summarize_transition_matrix(
    trajectory_df,
    from_visit,
    to_visit,
    positive_only=False,
    subtype=None,
):
    analysis_df = trajectory_df
    if positive_only:
        analysis_df = analysis_df.loc[analysis_df["PTB"] == 1]
    if subtype is not None:
        analysis_df = analysis_df.loc[analysis_df["PTB_subtype"] == subtype]

    valid_mask = (
        analysis_df[f"{from_visit}_state"].isin(["L", "R", "H"])
        & analysis_df[f"{to_visit}_state"].isin(["L", "R", "H"])
    )
    analysis_df = analysis_df.loc[valid_mask]

    count_table = pd.crosstab(
        analysis_df[f"{from_visit}_state"],
        analysis_df[f"{to_visit}_state"],
    ).reindex(index=["L", "R", "H"], columns=["L", "R", "H"], fill_value=0)
    row_percent = count_table.div(
        count_table.sum(axis=1).replace(0, np.nan),
        axis=0,
    ) * 100.0
    return count_table, row_percent


def summarize_longitudinal_metrics(trajectory_df):
    rows = []
    for subset_name, subset_mask in _trajectory_subset_masks(trajectory_df).items():
        subset = trajectory_df.loc[subset_mask]
        if subset.empty:
            continue

        full_path = subset.loc[subset["U3_landmark_eligible"]]
        if len(full_path) > 0:
            rows.extend(
                [
                    {
                        "Subset": subset_name,
                        "Transition": "Across U1-U3",
                        "Metric": "Any state change",
                        "Numerator": int(full_path["changed_any"].sum()),
                        "Denominator": len(full_path),
                        "Rate": float(full_path["changed_any"].mean()),
                    },
                    {
                        "Subset": subset_name,
                        "Transition": "Across U1-U3",
                        "Metric": "Two state changes",
                        "Numerator": int((full_path["number_of_changes"] == 2).sum()),
                        "Denominator": len(full_path),
                        "Rate": float((full_path["number_of_changes"] == 2).mean()),
                    },
                ]
            )

        for from_visit, to_visit in [("U1", "U2"), ("U2", "U3")]:
            transition_subset = subset.loc[
                subset[f"{from_visit}_state"].isin(["L", "R", "H"])
                & subset[f"{to_visit}_state"].isin(["L", "R", "H"])
            ]
            current = transition_subset[f"{from_visit}_state"]
            following = transition_subset[f"{to_visit}_state"]
            y_true = transition_subset["PTB"]

            review_mask = current == "R"
            determinate_next = following.isin(["L", "H"])
            correct_next = (
                ((following == "L") & (y_true == 0))
                | ((following == "H") & (y_true == 1))
            )
            not_high_mask = current != "H"
            all_valid = pd.Series(True, index=transition_subset.index)

            metric_specs = [
                ("Uncertainty resolution (R to L/H)", review_mask & determinate_next, review_mask),
                ("Correct uncertainty resolution", review_mask & determinate_next & correct_next, review_mask),
                ("R to L", review_mask & (following == "L"), review_mask),
                ("R to H", review_mask & (following == "H"), review_mask),
                ("High-risk emergence", not_high_mask & (following == "H"), not_high_mask),
                ("State stability", current == following, all_valid),
            ]
            for metric_name, numerator_mask, denominator_mask in metric_specs:
                denominator = int(denominator_mask.sum())
                numerator = int(numerator_mask.sum())
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


def summarize_co_outcome_enrichment(trajectory_df, co_outcomes):
    """Describe co-outcomes without treating D as an ordinary model state."""
    eligible = trajectory_df["U3_landmark_eligible"]
    changed = trajectory_df["changed_any"].fillna(0).astype(bool)
    group_masks = {
        "Whole PTB modelling cohort": pd.Series(True, index=trajectory_df.index),
        "U3 landmark risk set": eligible,
        "Delivered before U3": trajectory_df["delivered_before_U3"],
        "Any state change among U3-eligible": eligible & changed,
        "Stable low (L-L-L)": eligible & (trajectory_df["trajectory"] == "L-L-L"),
        "End high at U3": eligible & (trajectory_df["U3_state"] == "H"),
        "Emerging high by U3": eligible & (trajectory_df["U1_state"] != "H") & (trajectory_df["U3_state"] == "H"),
        "Persistent high (H-H-H)": eligible & (trajectory_df["trajectory"] == "H-H-H"),
        "Persistent review (R-R-R)": eligible & (trajectory_df["trajectory"] == "R-R-R"),
        "Review resolved by U3": eligible & (trajectory_df["U1_state"] == "R") & trajectory_df["U3_state"].isin(["L", "H"]),
        "R-to-H at either transition": eligible & (
            ((trajectory_df["U1_state"] == "R") & (trajectory_df["U2_state"] == "H"))
            | ((trajectory_df["U2_state"] == "R") & (trajectory_df["U3_state"] == "H"))
        ),
    }

    rows = []
    for outcome_name, values in co_outcomes.items():
        outcome_series = pd.Series(values, index=trajectory_df.index, dtype=float)
        valid_all = outcome_series.notna()
        baseline_rate = outcome_series.loc[valid_all].mean()
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
                    "Cohort prevalence": float(baseline_rate),
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
    """Print landmark-aware all-cause and subtype-comparative trajectory tables."""
    eligible = trajectory_df.loc[trajectory_df["U3_landmark_eligible"]]
    early = trajectory_df.loc[trajectory_df["delivered_before_U3"]]

    print("\n" + "=" * 100)
    print(
        f"SUBJECT-LEVEL PTB TRAJECTORY ANALYSIS: confidence={confidence:.0%}, "
        f"alpha={1.0-confidence:.2f}"
    )
    print("=" * 100)
    print(
        f"Master cohort N={len(trajectory_df)}, PTB events={int(trajectory_df['PTB'].sum())} "
        f"({trajectory_df['PTB'].mean():.2%}); "
        f"U3 landmark risk set N={len(eligible)}, PTB events={int(eligible['PTB'].sum())} "
        f"({eligible['PTB'].mean():.2%}); "
        f"delivered before U3 N={len(early)}"
    )

    coverage_summary = summarize_empirical_coverage(trajectory_df, confidence)
    print("\n0) Empirical marginal and class-conditional coverage:")
    print(
        coverage_summary.to_string(
            index=False,
            formatters={
                "Empirical coverage": lambda value: f"{value:.4f}",
                "95% CI lower": lambda value: f"{value:.4f}",
                "95% CI upper": lambda value: f"{value:.4f}",
                "Difference from nominal": lambda value: f"{value:+.4f}",
            },
        )
    )

    state_summary = summarize_state_prevalence(trajectory_df)
    print("\n1) L/R/H states and structural D markers by visit:")
    print(
        state_summary.to_string(
            index=False,
            formatters={
                "Percent of subset": lambda value: f"{value:.2f}",
                "Percent among at-risk": lambda value: "" if pd.isna(value) else f"{value:.2f}",
            },
        )
    )

    early_subtype, early_path = summarize_early_delivery_before_u3(trajectory_df)
    print("\n2) Pregnancies delivered before the U3 landmark (marked D, not compared with U3 L/R/H):")
    if early_subtype.empty:
        print("None")
    else:
        print("Subtype composition:")
        print(early_subtype.to_string(index=False, formatters={"Percent": lambda value: f"{value:.2f}"}))
        print("U1-U2 decision paths before delivery:")
        print(early_path.to_string(index=False))

    exact_summary = summarize_exact_trajectories(trajectory_df)
    print("\n3) Exact U1-U2-U3 trajectories in the U3 landmark risk set only:")
    print(
        exact_summary.to_string(
            index=False,
            formatters={
                "Percent": lambda value: f"{value:.2f}",
                "PTB event rate": lambda value: f"{value:.4f}",
                "Event-rate CI lower": lambda value: f"{value:.4f}",
                "Event-rate CI upper": lambda value: f"{value:.4f}",
                "Enrichment vs U3 risk set": lambda value: f"{value:.2f}",
            },
        )
    )

    positive_summary = summarize_exact_trajectories(trajectory_df, positive_only=True)
    sptb_summary = summarize_exact_trajectories(trajectory_df, subtype=PTB_SUBTYPE_SPTB)
    iptb_summary = summarize_exact_trajectories(trajectory_df, subtype=PTB_SUBTYPE_IPTB)
    print("\n4a) Exact trajectory distribution among all PTB events remaining in the U3 risk set:")
    print(positive_summary[["Trajectory", "N", "Percent"]].to_string(index=False, formatters={"Percent": lambda value: f"{value:.2f}"}))
    print("\n4b) Exact trajectory distribution among sPTB events remaining in the U3 risk set:")
    print(sptb_summary[["Trajectory", "N", "Percent"]].to_string(index=False, formatters={"Percent": lambda value: f"{value:.2f}"}))
    print("\n4c) Exact trajectory distribution among iPTB events remaining in the U3 risk set:")
    print(iptb_summary[["Trajectory", "N", "Percent"]].to_string(index=False, formatters={"Percent": lambda value: f"{value:.2f}"}))

    grouped_summary = summarize_trajectory_groups(trajectory_df)
    print("\n5) Clinically grouped trajectories and observed all-cause PTB incidence in the U3 risk set:")
    print(
        grouped_summary.to_string(
            index=False,
            formatters={
                "Percent of U3 risk set": lambda value: f"{value:.2f}",
                "PTB event rate": lambda value: f"{value:.4f}",
                "Event-rate CI lower": lambda value: f"{value:.4f}",
                "Event-rate CI upper": lambda value: f"{value:.4f}",
                "Enrichment vs U3 risk set": lambda value: f"{value:.2f}",
            },
        )
    )

    subtype_composition = summarize_subtype_composition_by_trajectory(trajectory_df)
    print("\n6) sPTB versus iPTB composition within clinical trajectory groups:")
    print(
        subtype_composition.to_string(
            index=False,
            formatters={
                "PTB event rate": lambda value: f"{value:.4f}",
                "sPTB among PTB (%)": lambda value: f"{value:.2f}",
                "iPTB among PTB (%)": lambda value: f"{value:.2f}",
                "Other among PTB (%)": lambda value: f"{value:.2f}",
            },
        )
    )

    transition_tables = {}
    transition_subsets = [
        ("all", "all samples", False, None),
        ("positive", "PTB-positive only", True, None),
        ("sptb", "sPTB only", False, PTB_SUBTYPE_SPTB),
        ("iptb", "iPTB only", False, PTB_SUBTYPE_IPTB),
    ]
    for subset_key, subset_label, positive_only, subtype in transition_subsets:
        for from_visit, to_visit in [("U1", "U2"), ("U2", "U3")]:
            counts, row_percent = summarize_transition_matrix(
                trajectory_df,
                from_visit,
                to_visit,
                positive_only=positive_only,
                subtype=subtype,
            )
            transition_key = f"{subset_key}_{from_visit}_{to_visit}"
            transition_tables[transition_key + "_counts"] = counts
            transition_tables[transition_key + "_row_percent"] = row_percent
            note = " (D excluded)" if to_visit == "U3" else ""
            print(f"\n7) {from_visit}->{to_visit} transition counts ({subset_label}){note}:")
            print(counts.to_string())
            print(f"{from_visit}->{to_visit} row percentages ({subset_label}){note}:")
            print(row_percent.to_string(float_format=lambda value: f"{value:.2f}"))

    longitudinal_metrics = summarize_longitudinal_metrics(trajectory_df)
    print("\n8) Longitudinal change and uncertainty-resolution metrics (D excluded where U3 is involved):")
    print(longitudinal_metrics.to_string(index=False, formatters={"Rate": lambda value: f"{value:.4f}"}))

    co_outcome_summary = None
    if co_outcomes is not None:
        co_outcome_summary = summarize_co_outcome_enrichment(trajectory_df, co_outcomes)
        print("\n9) Descriptive co-outcome prevalence by landmark-aware trajectory stratum:")
        print(
            co_outcome_summary.to_string(
                index=False,
                formatters={
                    "Prevalence": lambda value: f"{value:.4f}",
                    "Cohort prevalence": lambda value: f"{value:.4f}",
                    "Prevalence ratio": lambda value: f"{value:.2f}",
                },
            )
        )

    return {
        "subject_level": trajectory_df,
        "coverage_summary": coverage_summary,
        "state_summary": state_summary,
        "early_delivery_subtype_summary": early_subtype,
        "early_delivery_U1_U2_path_summary": early_path,
        "exact_trajectory_summary": exact_summary,
        "positive_only_trajectory_summary": positive_summary,
        "sPTB_trajectory_summary": sptb_summary,
        "iPTB_trajectory_summary": iptb_summary,
        "grouped_trajectory_summary": grouped_summary,
        "subtype_composition_by_trajectory": subtype_composition,
        "transition_tables": transition_tables,
        "longitudinal_metrics": longitudinal_metrics,
        "co_outcome_summary": co_outcome_summary,
    }


# =============================================================================
# Visit-specific fixed CV splits and landmark-aware model execution
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
        "outcome": "PTB", "visit": visit_name, "n": int(len(y)), "events": int(y.sum()),
        "prevalence": float(y.mean()), "AUROC": float(auroc), "AUROC_CI95_low": float(auc_lo),
        "AUROC_CI95_high": float(auc_hi), "AUPRC": float(auprc), "AUPRC_CI95_low": float(pr_lo),
        "AUPRC_CI95_high": float(pr_hi), "Brier_score": float(brier),
        "Brier_CI95_low": float(brier_lo), "Brier_CI95_high": float(brier_hi),
        "calibration_intercept": cal_i, "calibration_slope": cal_s,
        "mean_predicted_probability": float(np.mean(p)),
    }
    subject = pd.DataFrame({
        "subject_id": np.asarray(PTB_ID_relabel0, dtype=object), "outcome": label1, "visit": visit_name,
        "eligible": analysis_mask, "outer_fold": fold_id, "oof_probability": oof,
    })
    return summary, subject, pd.concat(tuning_rows, ignore_index=True), pd.DataFrame(selected_rows)


# Delivery-based visit eligibility: V2 >=22+0 weeks; V3 uses the source pipeline U3 eligibility mask.
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

summary_path = os.path.join(OUTPUT_DIR, "PTB_nestedCV_performance_summary.csv")
subject_path = os.path.join(OUTPUT_DIR, "PTB_nestedCV_subject_level_oof_predictions.csv")
tuning_path = os.path.join(OUTPUT_DIR, "PTB_nestedCV_inner_tuning_all_candidates.csv")
selected_path = os.path.join(OUTPUT_DIR, "PTB_nestedCV_selected_hyperparameters_by_fold.csv")
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
