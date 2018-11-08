# -*- coding: utf-8 -*-
# @Time    : 18-8-23 下午3:29
# @Author  : HeJi
# @FileName: module.py
# @E-mail: hj@jimhe.cn


from rdkit.Chem import AllChem
import numpy as np
from contextlib import contextmanager
import time
import sys

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

def convert_to_one_hot(y, C):
    y = y.astype(int)
    return np.eye(C)[y.reshape(-1)]


def type_filter(data, column, target_type):
    """

    :param data: pandas dataframe
    :param column: str
    :return: filtered data
    """
    masks = data[column].apply(lambda x: type(x) == target_type)
    data = data[masks]
    data.reset_index(drop=True, inplace = True)
    return data


def reaction_from_smart(smart):
    try:
        return(AllChem.ReactionFromSmarts(smart))
    except:
        return None


def get_reaction_fingerprint(reactions, fp_dim = 2048):
    """

    :param reactions: list or np.array
    :return: reaction fingerprint
    """
    all_reaction_fps = np.zeros([len(reactions), fp_dim], dtype='int8')
    for i, reaction in enumerate(reactions):
        reaction_fps = list(AllChem.CreateDifferenceFingerprintForReaction(reaction))
        all_reaction_fps[i] = reaction_fps
    return all_reaction_fps


def get_product_fingerprint(products, fp_dim = 100000):
    """

    :param products: list or np.array
    :return: product fingerprint
    """
    all_product_fps = np.zeros([len(products), fp_dim], dtype='int8')
    for i, product in enumerate(products):
        product_fps = list(AllChem.GetMorganFingerprintAsBitVect(product, radius = 2, nBits = fp_dim))
        all_product_fps[i] = product_fps
    return all_product_fps


def get_top_k_index(target_array, k):
    return np.argpartition(np.abs(target_array), -k)[-k:]



def dbg(*objects, file=sys.stderr, flush=True, **kwargs):
    "Helper function to print to stderr and flush"
    print(*objects, file=file, flush=flush, **kwargs)
