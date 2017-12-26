#coding=utf-8
import numpy as np
def cal_euclidean_distance(vec1,vec2):
    """
    :param vec1:向量1
    :param vec2:向量2
    :return:返回两个向量之间的欧氏距离
    """
    np_vec1, np_vec2 = np.array(vec1), np.array(vec2)
    euclidean_distance = np.sqrt(np.sum(np.square(np_vec1 - np_vec2)))
    return euclidean_distance