# -*- coding: utf-8 -*-
# @Time    : 18-8-23 下午3:29
# @Author  : HeJi
# @FileName: module.py
# @E-mail: hj@jimhe.cn


import numpy as np
import pandas as pd
from utils import reaction_from_smart, get_reaction_fingerprint, get_product_fingerprint, convert_to_one_hot, get_top_k_index
from rdkit import Chem
import os
import gc
import argparse
import keras
import multiprocessing
from multiprocessing import Process
from sklearn.preprocessing import LabelEncoder
import math
from multiprocessing import Pool
from keras.utils import to_categorical

def np_loader(path):
    return np.load(path)


class DataGenerator_v2(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path, shuffle=True, batch_size = 1024, file_indexes = (0,1,2,3,4), used_indexes = None, relabel = None):
        'Initialization'
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_indexes = file_indexes
        self.used_indexes = used_indexes
        assert (data_path is not None)

        #pool1 = multiprocessing.Pool(len(file_indexes))
        #pool2 = multiprocessing.Pool(len(file_indexes))
        #pool3 = multiprocessing.Pool(len(file_indexes))
        #pool4 = multiprocessing.Pool(len(file_indexes))

        self.products = {i : np.load(data_path + "/product_fps_" + str(i) + ".npy") for i in file_indexes}
        self.labels = {i: np.load(data_path+ "/labels_"+ str(i) + ".npy") for i in file_indexes}
        #self.n_classes = np.concatenate(list(self.labels.values())).shape[0]

        #self.pos_products = {i: pool1.apply_async(np_loader, (pos_data_path + "/product_fps_pos_" + str(i) + ".npy",)).get() for i in file_indexes}
        #self.pos_reactions = {i: pool1.apply_async(np_loader, (pos_data_path + "/reaction_fps_pos_" + str(i) + ".npy",)).get() for i in file_indexes}
        lengths = []

        for file in self.file_indexes:
            lengths.append(self.products[file].shape[0])
        self.width = self.products[file_indexes[0]].shape[-1]

        #self.neg_products = {i: pool1.apply_async(np_loader, (neg_data_path + "/product_fps_neg_" + str(i) + ".npy",)).get() for i in file_indexes}
        #self.neg_reactions = {i: pool1.apply_async(np_loader, (neg_data_path + "/reaction_fps_neg_" + str(i) + ".npy",)).get() for i in file_indexes}
        #pool1.close()
        #pool1.join()

        self.length = np.sum(lengths)

        self.y = np.concatenate([self.labels[i] for i in file_indexes])
        if relabel is not None:
            self.y = relabel.transform(self.y)

        #lbl = LabelEncoder()
        #self.y = lbl.fit_transform(self.y)
        unique_classes = np.unique(self.y)
        self.n_classes = len(np.unique(unique_classes))


        self.on_epoch_end()

        self.hash = {}

        accumulate_sum = np.add.accumulate(lengths)
        #print("pos_accumulate_sum", pos_accumulate_sum)
        #print("neg_accumulate_sum", neg_accumulate_sum)
        for i in range(int(self.length)):
            for ind, j in enumerate(accumulate_sum):
                if i < j:
                    if ind == 0:
                        self.hash[i] = [self.file_indexes[ind], i]
                    else:
                        self.hash[i] = [self.file_indexes[ind], i - accumulate_sum[ind-1]]
                    break


    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.list_IDs) / self.batch_size))
        if self.used_indexes is not None:
            return len(self.used_indexes) // self.batch_size
        else:
            return len(self.y) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)]

        # Find list of IDs

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.used_indexes is not None:
            self.indexes = self.used_indexes
        else:
            self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_loc_from_hashs(self, indexes):
        all_data = np.zeros([len(indexes), 2], dtype=int)
        for i, ind in enumerate(indexes):
            all_data[i] = self.hash[ind]
        return all_data


    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        locs = self.get_loc_from_hashs(indexes)

        #print(locs_pos)

        product_batch = np.zeros([len(indexes), self.width])

        y_batch = self.y[indexes]

        y_batch = to_categorical(y_batch, num_classes=self.n_classes)


        for i, loc in enumerate(locs):
            try:
                product_batch[i] = self.products[loc[0]][loc[1]]
            except:
                print(self.products)
                print(loc)
                print("*"*100)


        return product_batch, y_batch


def build_data(all_smarts, fp_dim):
    """

    :param data_path: str, path to reaction files, xls format
    :return:
    """
    #str_masks = np.array(list(map(lambda x: type(x) == str, all_smarts)))
    #all_smarts = all_smarts[str_masks]

    all_product_smiles = []
    useful_filter = []
    all_product_mols = []
    for smt in all_smarts:
        product_smile = smt.split(">>")[-1]
        all_product_smiles.append(product_smile)
        mol = Chem.MolFromSmiles(product_smile)
        if mol is not None:
            all_product_mols.append(mol)
            useful_filter.append(True)
        else:
            useful_filter.append(False)

    product_fps = get_product_fingerprint(all_product_mols, fp_dim = fp_dim)

    return product_fps, np.array(useful_filter)


def generate_negative_samples(smarts, n):
    """

    :param smarts: np.array or list
    :param n: num of samples want to generated
    :return: negative samples
    """
    count = 0
    product_smiles = []
    reactant_smiles = []
    for smt in smarts:
        a = tuple(smt.split(">>"))
        product_smiles.append(a[-1])
        reactant_smiles.append(a[0])
    product_smiles = np.array(product_smiles)
    reactant_smiles = np.array(reactant_smiles)
    neg_samples = []
    while True:
        shuffled_smiles = np.random.permutation(product_smiles)
        masks = shuffled_smiles != product_smiles
        neg_sap = [react_smi+'>>'+pro_smi for react_smi, pro_smi in zip(reactant_smiles[masks],shuffled_smiles[masks])]
        coo_num = masks.sum()
        neg_samples.extend(neg_sap)
        count += coo_num
        if count>n:
            break
    return np.array(neg_samples)


def generate_negative_samples_v2(smarts, n):
    """

    :param smarts: np.array or list
    :param n: num of samples want to generated
    :return: negative samples
    """
    product_smiles = []
    reactant_smiles = []
    for smt in smarts:
        a = tuple(smt.split(">>"))
        product_smiles.append(a[-1])
        reactant_smiles.append(a[0])
    #product_smiles = np.array(product_smiles)
    #reactant_smiles = np.array(reactant_smiles)
    neg_samples = []

    length = len(product_smiles)
    batch_size = 100000
    while True:
        random_array1 = np.random.choice(length, batch_size)
        random_array2 = np.random.choice(length, batch_size)
        for (a, b) in zip(random_array1, random_array2):
            if product_smiles[a] != product_smiles[b]:
                neg_sap = reactant_smiles[a]+'>>'+product_smiles[b]
                neg_samples.append(neg_sap)
        if len(neg_samples)>=n:
            break
    return neg_samples


def get_all_reaction_files(data_path):
    all_files = np.array(os.listdir(data_path))
    files_mask = np.array(list(map(lambda x: os.path.isfile(os.path.join(data_path, x)), all_files)))
    true_files = all_files[files_mask]
    xls_mask = np.array(list(map(lambda x: x.split('.')[-1] == 'xls', true_files)))
    xls_files = true_files[xls_mask]
    return xls_files


def get_all_smarts(data_path):
    """

    :param path_to_data_dir:
    :return:
    """
    all_smarts = np.array([])
    all_files = get_all_reaction_files(data_path)
    for file in all_files:
        absolute_path = os.path.join(data_path, file)
        #print(absolute_path)
        try:
            smarts = pd.read_table(absolute_path, usecols=["Reaction"])["Reaction"].values
            #print(smarts)
            all_smarts = np.concatenate([all_smarts, smarts])
        except:
            print("This file cannot be read by pandas for some reason", absolute_path)
    str_masks = np.array(list(map(lambda x: type(x) == str, all_smarts)))
    all_smarts = all_smarts[str_masks]
    non_product_masks = np.array(list(map(lambda x: x.split('>>')[-1] != '', all_smarts)))
    non_reactant_masks = np.array(list(map(lambda x: x.split('>>')[0] != '', all_smarts)))
    masks = np.bitwise_and(non_product_masks , non_reactant_masks)
    all_smarts = all_smarts[masks]
    return all_smarts


def rule_filter(labels, occurance_threshold = 2):
    count = np.bincount(labels)
    coverage_rate = count[count>=occurance_threshold].sum()/len(labels)
    print("coverage_rate", coverage_rate)
    keeped_labels = np.where(count>=occurance_threshold)[0]
    the_filter = np.array([labels[i] in keeped_labels for i in range(len(labels))])
    return the_filter

def clean_data(data):
    smart1_filter = []
    smart2_filter = []
    data.reset_index(drop=True, inplace=True)
    smarts1 = data.clean_reaction_center_radius_smarts0.values
    smarts2 = data.clean_reaction_center_radius_smarts1.values
    for i in range(len(smarts1)):
        if smarts1[i][:2] == ">>":
            smart1_filter.append(False)
        else:
            smart1_filter.append(True)
        if smarts2[i][:2] == ">>":
            smart2_filter.append(False)
        else:
            smart2_filter.append(True)
    smart1_filter = np.array(smart1_filter)
    smart2_filter = np.array(smart2_filter)
    new_data = data.iloc[np.logical_and(smart1_filter, smart2_filter)]
    new_data.reset_index(drop=True, inplace=True)
    return new_data


def make_useful_data(data, save_path):
    data.reset_index(drop=True, inplace = True)
    data = data[["mapped_reaction_smiles","clean_reaction_center_radius_smarts0","clean_reaction_center_radius_smarts1"]]
    data = clean_data(data)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    data.reset_index(drop=True, inplace=True)
    lbl0 = LabelEncoder()
    lbl1 = LabelEncoder()
    lbl_exp = LabelEncoder()
    lbl_roll = LabelEncoder()
    expand_rule_labels = lbl0.fit_transform(data.clean_reaction_center_radius_smarts0.values)
    rollout_rule_labels = lbl1.fit_transform(data.clean_reaction_center_radius_smarts1.values)
    expand_rule_filter = rule_filter(labels=expand_rule_labels, occurance_threshold=2)
    rollout_rule_filter = rule_filter(labels=rollout_rule_labels, occurance_threshold=3)
    data_expand_network = data[expand_rule_filter]
    data_expand_network.reset_index(drop=True, inplace=True)
    oexpand_rule_labels = lbl0.inverse_transform(expand_rule_labels[expand_rule_filter])
    lbl_exp.fit_transform(oexpand_rule_labels)
    data_expand_network["y"] = lbl_exp.fit_transform(oexpand_rule_labels)
    del data_expand_network["clean_reaction_center_radius_smarts1"]
    data_expand_network.columns = ["mapped_reaction_smiles", "reaction_center_radius0", 'y']

    data_rollout_network = data[rollout_rule_filter]
    data_rollout_network.reset_index(drop=True, inplace=True)
    orollout_rule_labels = lbl1.inverse_transform(rollout_rule_labels[rollout_rule_filter])
    data_rollout_network["y"] = lbl_roll.fit_transform(orollout_rule_labels)
    del data_rollout_network["clean_reaction_center_radius_smarts0"]
    data_rollout_network.columns = ["mapped_reaction_smiles", "reaction_center_radius1", 'y']

    data_expand_network.to_csv(os.path.join(save_path, "data_expand_network.csv"), index=False)
    data_rollout_network.to_csv(os.path.join(save_path, "data_rollout_network.csv"), index = False)

    np.save(os.path.join(save_path, "expand_rule_label_encoder.npy"), lbl_exp.classes_)
    np.save(os.path.join(save_path, "rollout_rule_label_encoder.npy"), lbl_roll.classes_)


def get_args():
    '''
    get arguments
    :return:
    '''
    parser = argparse.ArgumentParser(description='generator')
    parser.add_argument('--data_path', type=str,
                        default='/home/stein/Documents/chemical_reaction_data/synthesis_reaction/useful_data/data_expand_network.csv',
                        help='data_path')
    parser.add_argument('--save_path', type=str,
                        default='useful_data/expnet',
                        help='save_path')
    parser.add_argument('--num_data_batchs', type=int,
                        default=1,
                        help="num_data_batchs")
    parser.add_argument('--fp_dim', type=int, default=100000)
    parser.add_argument('--k', type=int, default=10000, help="num_top_k_indexes")

    return parser.parse_args()

def run(smiles, labels, index, size, save_path, save_column_indexes, batch_size = 10240, fp_dim = 1000000):
    size = int(math.ceil(len(smiles) / size))
    start = size * index
    end = (index + 1) * size if (index + 1) * size < len(smiles) else len(smiles)
    temp_smiles = smiles[start:end]
    temp_labels = labels[start:end]
    all_data = []
    all_labels = []
    num_batches = len(temp_smiles) // batch_size
    for i in range(num_batches):
        batch_smiles = temp_smiles[i * batch_size:(i + 1) * batch_size]
        batch_labels = temp_labels[i * batch_size:(i + 1) * batch_size]
        product_fps, useful_filter = build_data(batch_smiles, fp_dim=fp_dim)
        print("product_fps.shape", product_fps.shape)
        product_fps = product_fps[:,save_column_indexes]
        batch_labels = batch_labels[useful_filter]
        np.save(save_path + "/product_fps_" + str(index)+"_" + str(i) + ".npy", product_fps)
        np.save(save_path + "/labels_" + str(index)+"_" + str(i)+ ".npy", batch_labels)
        all_data.append(product_fps)
        all_labels.append(batch_labels)
    batch_smiles = temp_smiles[num_batches * batch_size:]
    batch_labels = temp_labels[num_batches * batch_size:]
    product_fps, useful_filter = build_data(batch_smiles, fp_dim=fp_dim)
    print("product_fps.shape", product_fps.shape)
    product_fps = product_fps[:, save_column_indexes]
    batch_labels = batch_labels[useful_filter]
    np.save(save_path + "/product_fps_" + str(index) + "_" + str(num_batches) + ".npy", product_fps)
    np.save(save_path + "/labels_" + str(index) + "_" + str(num_batches) + ".npy", batch_labels)
    all_data.append(product_fps)
    all_labels.append(batch_labels)
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels)
    np.save(save_path + "/product_fps_" + str(index) + ".npy", all_data)
    np.save(save_path + "/labels_" + str(index) + ".npy", all_labels)


def run_sum(smiles, index, size, return_dict, batch_size = 64, fp_dim = 1000000):
    size = int(math.ceil(len(smiles) / size))
    start = size * index
    end = (index + 1) * size if (index + 1) * size < len(smiles) else len(smiles)
    temp_smiles = smiles[start:end]
    sum_product_fps = np.zeros([fp_dim, ])
    num_batches = len(temp_smiles)//batch_size
    for i in range(num_batches):
        batch_smiles = temp_smiles[i*batch_size:(i+1)*batch_size]
        product_fps, _ = build_data(batch_smiles, fp_dim=fp_dim)
        sum_product_fps += np.sum(product_fps, axis=0)
    batch_smiles = temp_smiles[num_batches * batch_size:]
    product_fps, _ = build_data(batch_smiles, fp_dim=fp_dim)
    sum_product_fps += np.sum(product_fps, axis=0)
    #np.save(save_path + "/sum_product_fps_" + str(index) + ".npy", sum_product_fps)
    print("sum_product_fps.shape", sum_product_fps.shape)
    return_dict[index] = sum_product_fps



if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    data = pd.read_csv(args.data_path)
    data.reset_index(drop=True, inplace=True)
    #data = data.iloc[:10000]
    all_smiles = data["mapped_reaction_smiles"].values
    all_labels = data["y"].values
    fp_dim = args.fp_dim
    num_top_k_indexes = args.k

    #num_batchs = args.num_data_batchs

    #all_smarts_pos = pd.read_csv("data/pos/all_smarts_pos.csv")

    #all_smarts_neg = pd.read_csv("data/neg/all_smarts_neg.csv")

    used_cpu = multiprocessing.cpu_count() // 2

    #p = Pool(used_cpu)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(used_cpu):
        #p = Process(target=run, args=(all_smiles,all_labels, i, used_cpu, args.save_path))
        p = Process(target=run_sum, args=(all_smiles, i, used_cpu, return_dict, 1024, fp_dim))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    value_list = list(return_dict.values())
    result_sum = np.zeros_like(value_list[0])
    for val in value_list:
        result_sum += val

    top_k_indexes = get_top_k_index(result_sum, k = num_top_k_indexes)

    
    print(top_k_indexes)
    np.save(args.save_path+"/top_"+str(num_top_k_indexes)+"_indexes.npy", top_k_indexes)

    top_k_indexes = np.load(args.save_path+"/top_10000_indexes.npy")
    for i in range(used_cpu):
        p = Process(target=run, args=(all_smiles, all_labels, i, used_cpu, args.save_path, top_k_indexes,1024, fp_dim))
        p.start()


    """
    batch_size = len(all_smiles)//num_batchs

    for i in range(0, num_batchs):

        if i == num_batchs - 1:

            smiles = list(all_smiles)[i*batch_size:]
            labels = all_labels[i*batch_size:]

        else:

            smiles = list(all_smiles)[i * batch_size:(i + 1) * batch_size]
            labels = all_labels[i * batch_size:(i + 1) * batch_size]

        product_fps, useful_filter = build_data(smiles, fp_dim=100000)

        labels = labels[useful_filter]
        np.save(args.save_path+"/product_fps_"+str(i)+".npy", product_fps)
        np.save(args.save_path + "/labels_" + str(i) + ".npy", labels)

        del product_fps, useful_filter, labels
        gc.collect()
    """

