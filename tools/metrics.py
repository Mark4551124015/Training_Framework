# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d
from sklearn import metrics
import os
import numpy as np


def thresholding(pred_arr, threshold=0.5):
    pred_arr = np.where(pred_arr > threshold, 1, 0).astype(np.uint8)
    return pred_arr

def max_fusion(x, y):
    assert x.shape == y.shape
    
    return np.maximum(x, y)


def extract_mask(pred_arr, gt_arr, mask_arr=None):
    # we want to make them into vectors
    pred_vec = pred_arr.flatten()
    gt_vec = gt_arr.flatten()
    
    if mask_arr is not None:
        mask_vec = mask_arr.flatten()
        idx = list(np.where(mask_vec == 0)[0])
        
        pred_vec = np.delete(pred_vec, idx)
        gt_vec = np.delete(gt_vec, idx)
    
    return pred_vec, gt_vec


def calc_auc(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    roc_auc = metrics.roc_auc_score(gt_vec, pred_vec)
    
    return roc_auc


def numeric_score(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    """Computation of statistical numerical scores:
    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives
    return: tuple (FP, FN, TP, TN)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated_gt_arr = cv2.dilate(gt_arr, kernel, iterations=1)
    
    FP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 0)))
    FN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 1)))
    TP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 1)))
    TN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 0)))
    
    return FP, FN, TP, TN


def calc_acc(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    acc = (TP + TN) / (FP + FN + TP + TN)
    
    return acc


def calc_sen(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    sen = TP / (FN + TP + 1e-12)
    
    return sen


def calc_fdr(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    fdr = FP / (FP + TP + 1e-12)
    
    return fdr


def calc_spe(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    spe = TN / (FP + TN + 1e-12)
    
    return spe


def calc_gmean(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    sen = calc_sen(pred_arr, gt_arr, kernel_size=kernel_size)
    spe = calc_spe(pred_arr, gt_arr, kernel_size=kernel_size)
    
    return math.sqrt(sen * spe)


def calc_kappa(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size=kernel_size)
    matrix = np.array([[TP, FP],
                       [FN, TN]])
    n = np.sum(matrix)
    
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    
    return (po - pe) / (1 - pe)


def calc_iou(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    iou = TP / (FP + FN + TP + 1e-12)
    
    return iou


def calc_dice(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    dice = 2.0 * TP / (FP + FN + 2 * TP + 1e-12)
    
    return dice

def count_bytes(file_size):
    '''
    Count the number of parameters in model
    '''
    #param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    def strofsize(integer, remainder, level):
        if integer >= 1024:
            remainder = integer % 1024
            integer //= 1024
            level += 1
            return strofsize(integer, remainder, level)
        else:
            return integer, remainder, level
    
    def MBofstrsize(integer, remainder, level):
        remainder = integer % (1024*1024)
        integer //= (1024*1024)
        level = 2
        return integer, remainder, level

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    #integer, remainder, level = strofsize(int(file_size), 0, 0)
    #if level+1 > len(units):
    #    level = -1
    integer, remainder, level = MBofstrsize(int(file_size), 0, 0)
    return ( '{}.{:>03d} {}'.format(integer, remainder, units[level]) )

# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1.0 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 定义获取当前学习率的函数
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def calc_skel(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def calc_cl(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = calc_skel(v_p,skeletonize(v_l))
        tsens = calc_skel(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = calc_skel(v_p,skeletonize_3d(v_l))
        tsens = calc_skel(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)
