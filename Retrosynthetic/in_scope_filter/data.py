import numpy as np
import pandas as pd
from utils import reaction_from_smart, get_reaction_fingerprint, get_product_fingerprint
from rdkit import Chem
import os
import gc
import argparse
import keras
import multiprocessing


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pos_data_path, neg_data_path, shuffle=True, batch_size = 1024, file_indexes = (0,1,2,3,4)):
        'Initialization'
        self.pos_data_path = pos_data_path
        self.neg_data_path = neg_data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        assert (pos_data_path is not None and neg_data_path is not None)
        self.all_product = np.load(pos_data_path + "/product_fps_pos_" + str(file_indexes[0]) + ".npy")
        self.all_reaction = np.load(pos_data_path + "/reaction_fps_pos_" + str(file_indexes[0]) + ".npy")
        self.all_labels = np.tile([1], self.all_product.shape[0])

        neg_product = np.load(neg_data_path + "/product_fps_neg_" + str(file_indexes[0]) + ".npy")
        neg_reaction = np.load(neg_data_path + "/reaction_fps_neg_" + str(file_indexes[0]) + ".npy")
        neg_labels = np.tile([0], neg_product.shape[0])

        self.all_product = np.concatenate([self.all_product, neg_product], axis=0)
        self.all_reaction = np.concatenate([self.all_reaction, neg_reaction], axis=0)
        self.all_labels = np.concatenate([self.all_labels, neg_labels], axis=0)

        del neg_product, neg_labels, neg_reaction
        gc.collect()

        for i in range(1, len(file_indexes)):
            pos_product_pitch = np.load(pos_data_path + "/product_fps_pos_" + str(file_indexes[i]) + ".npy")
            self.all_product = np.concatenate([self.all_product, pos_product_pitch], axis=0)
            pos_labels_pitch = np.tile([1], pos_product_pitch.shape[0])
            self.all_labels = np.concatenate([self.all_labels, pos_labels_pitch], axis=0)
            del pos_product_pitch, pos_labels_pitch
            gc.collect()

            pos_reaction_pitch = np.load(pos_data_path + "/reaction_fps_pos_" + str(file_indexes[i]) + ".npy")
            self.all_reaction = np.concatenate([self.all_reaction, pos_reaction_pitch], axis=0)
            del pos_reaction_pitch
            gc.collect()

            neg_product_pitch = np.load(neg_data_path + "/product_fps_neg_" + str(file_indexes[i]) + ".npy")
            self.all_product = np.concatenate([self.all_product, neg_product_pitch])
            neg_labels_pitch = np.tile([0], neg_product_pitch.shape[0])
            self.all_labels = np.concatenate([self.all_labels, neg_labels_pitch])
            del neg_product_pitch, neg_labels_pitch
            gc.collect()

            neg_reaction_pitch = np.load(neg_data_path + "/reaction_fps_neg_" + str(file_indexes[i]) + ".npy")
            self.all_reaction = np.concatenate([self.all_reaction, neg_reaction_pitch], axis=0)
            del neg_reaction_pitch
            gc.collect()



    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.list_IDs) / self.batch_size))
        return len(self.all_labels) // batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]


        # Find list of IDs

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.all_labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        product_batch = self.all_product[indexes]
        reaction_batch = self.all_reaction[indexes]
        y_batch = self.all_labels[indexes]
        return [product_batch, reaction_batch], y_batch

def np_loader(path):
    return np.load(path)


class DataGenerator_v2(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pos_data_path, neg_data_path, shuffle=True, batch_size = 1024, file_indexes = (0,1,2,3,4)):
        'Initialization'
        self.pos_data_path = pos_data_path
        self.neg_data_path = neg_data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_indexes = file_indexes
        assert (pos_data_path is not None and neg_data_path is not None)

        #pool1 = multiprocessing.Pool(len(file_indexes))
        #pool2 = multiprocessing.Pool(len(file_indexes))
        #pool3 = multiprocessing.Pool(len(file_indexes))
        #pool4 = multiprocessing.Pool(len(file_indexes))

        self.pos_products = {i : np.load(pos_data_path + "/product_fps_pos_" + str(i) + ".npy") for i in file_indexes}
        self.pos_reactions = {i: np.load(pos_data_path + "/reaction_fps_pos_" + str(i) + ".npy") for i in file_indexes}

        #self.pos_products = {i: pool1.apply_async(np_loader, (pos_data_path + "/product_fps_pos_" + str(i) + ".npy",)).get() for i in file_indexes}
        #self.pos_reactions = {i: pool1.apply_async(np_loader, (pos_data_path + "/reaction_fps_pos_" + str(i) + ".npy",)).get() for i in file_indexes}
        pos_lengths = []
        for file in self.file_indexes:
            pos_lengths.append(self.pos_products[file].shape[0])

        self.neg_products = {i : np.load(neg_data_path + "/product_fps_neg_" + str(i) + ".npy") for i in file_indexes}
        self.neg_reactions = {i : np.load(neg_data_path + "/reaction_fps_neg_" + str(i) + ".npy") for i in file_indexes}

        #self.neg_products = {i: pool1.apply_async(np_loader, (neg_data_path + "/product_fps_neg_" + str(i) + ".npy",)).get() for i in file_indexes}
        #self.neg_reactions = {i: pool1.apply_async(np_loader, (neg_data_path + "/reaction_fps_neg_" + str(i) + ".npy",)).get() for i in file_indexes}
        #pool1.close()
        #pool1.join()
        neg_lengths = []
        for file in self.file_indexes:
            neg_lengths.append(self.neg_products[file].shape[0])

        self.pos_length = np.sum(pos_lengths)
        self.neg_length = np.sum(neg_lengths)

        self.y_pos = np.tile([1], self.pos_length)
        self.y_neg = np.tile([0], self.neg_length)

        self.on_epoch_end()

        self.hash_pos = {}
        self.hash_neg = {}

        pos_accumulate_sum = np.add.accumulate(pos_lengths)
        neg_accumulate_sum = np.add.accumulate(neg_lengths)
        #print("pos_accumulate_sum", pos_accumulate_sum)
        #print("neg_accumulate_sum", neg_accumulate_sum)
        for i in range(int(self.pos_length)):
            for ind, j in enumerate(pos_accumulate_sum):
                if i < j:
                    if ind == 0:
                        self.hash_pos[i] = [self.file_indexes[ind], i]
                    else:
                        self.hash_pos[i] = [self.file_indexes[ind], i - pos_accumulate_sum[ind-1]]
                    break


        for i in range(int(self.neg_length)):
            for ind, j in enumerate(neg_accumulate_sum):
                if i < j:
                    if ind == 0:
                        self.hash_neg[i] = [self.file_indexes[ind], i]
                    else:
                        self.hash_neg[i] = [self.file_indexes[ind], i - neg_accumulate_sum[ind-1]]
                    break

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.list_IDs) / self.batch_size))
        return min(len(self.y_pos) // self.batch_size, len(self.y_neg) // self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_pos = self.indexes_pos[index*int(self.batch_size/2):(index+1)*int(self.batch_size/2)]
        indexes_neg = self.indexes_neg[index*int(self.batch_size/2):(index+1)*int(self.batch_size/2)]

        # Find list of IDs

        # Generate data
        X, y = self.__data_generation(indexes_pos, indexes_neg)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes_pos = np.arange(len(self.y_pos))
        self.indexes_neg = np.arange(len(self.y_neg))
        if self.shuffle == True:
            np.random.shuffle(self.indexes_pos)
            np.random.shuffle(self.indexes_neg)

    def get_loc_from_pos_hashs(self, indexes):
        all_data = np.zeros([len(indexes), 2], dtype=int)
        for i, ind in enumerate(indexes):
            all_data[i] = self.hash_pos[ind]
        return all_data

    def get_loc_from_neg_hashs(self, indexes):
        all_data = np.zeros([len(indexes), 2], dtype=int)
        for i, ind in enumerate(indexes):
            all_data[i] = self.hash_neg[ind]
        return all_data


    def __data_generation(self, indexes_pos, indexes_neg):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        locs_pos = self.get_loc_from_pos_hashs(indexes_pos)
        locs_neg = self.get_loc_from_neg_hashs(indexes_neg)
        #print(locs_pos)

        product_batch_pos = np.zeros([len(indexes_pos), 16384])
        reaction_batch_pos = np.zeros([len(indexes_pos), 2048])
        y_batch_pos = self.y_pos[indexes_pos]

        product_batch_neg = np.zeros([len(indexes_neg), 16384])
        reaction_batch_neg = np.zeros([len(indexes_neg), 2048])
        y_batch_neg = self.y_neg[indexes_neg]

        for i, loc in enumerate(locs_pos):
            try:
                product_batch_pos[i] = self.pos_products[loc[0]][loc[1]]
                reaction_batch_pos[i] = self.pos_reactions[loc[0]][loc[1]]
            except:
                print(self.pos_products)
                print(loc)
                print("*"*100)

        for i, loc in enumerate(locs_neg):
            product_batch_neg[i] = self.neg_products[loc[0]][loc[1]]
            reaction_batch_neg[i] = self.neg_reactions[loc[0]][loc[1]]

        product_batch = np.concatenate([product_batch_pos, product_batch_neg], axis=0)
        reaction_batch = np.concatenate([reaction_batch_pos, reaction_batch_neg], axis=0)
        y_batch = np.concatenate([y_batch_pos, y_batch_neg], axis=0)

        return [product_batch, reaction_batch], y_batch


def build_data(all_smarts):
    """

    :param data_path: str, path to reaction files, xls format
    :return:
    """
    #str_masks = np.array(list(map(lambda x: type(x) == str, all_smarts)))
    #all_smarts = all_smarts[str_masks]

    all_product_smiles = []
    all_reactions = []
    useful_smarts = []
    all_product_mols = []
    for smt in all_smarts:
        product_smile = smt.split(">>")[-1]
        all_product_smiles.append(product_smile)
        reaction = reaction_from_smart(smt)
        mol = Chem.MolFromSmiles(product_smile)
        if reaction is not None and mol is not None:
            all_reactions.append(reaction)
            all_product_mols.append(mol)
            useful_smarts.append(smt)

    product_fps = get_product_fingerprint(all_product_mols, fp_dim=16384)
    reaction_fps = get_reaction_fingerprint(all_reactions, fp_dim=2048)

    return product_fps, reaction_fps, np.array(useful_smarts)


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



def get_args():
    '''
    get arguments
    :return:
    '''
    parser = argparse.ArgumentParser(description='generator')
    parser.add_argument('--data_path', type=str,
                        default='/home/stein/Downloads/all_reaction_data/AllReaction_20180816',
                        help='data_path')
    parser.add_argument('--save_path', type=str,
                        default='debug_data',
                        help='save_path')
    parser.add_argument('--num_data_batchs', type=int,
                        default=500,
                        help="num_data_batchs")

    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.save_path+"/pos"):
        os.mkdir(args.save_path+"/pos")
    if not os.path.exists(args.save_path+"/neg"):
        os.mkdir(args.save_path+"/neg")

    all_smarts_pos = get_all_smarts(data_path=args.data_path)
    all_smarts_neg = generate_negative_samples_v2(smarts=all_smarts_pos, n = len(all_smarts_pos))

    num_batchs = args.num_data_batchs

    #all_smarts_pos = pd.read_csv("data/pos/all_smarts_pos.csv")

    #all_smarts_neg = pd.read_csv("data/neg/all_smarts_neg.csv")

    batch_size = min(len(all_smarts_neg)//num_batchs, len(all_smarts_pos)//num_batchs)

    for i in range(0, num_batchs):

        if i == num_batchs - 1:

            smarts_pos = list(all_smarts_pos)[i*batch_size:]

            smarts_neg = list(all_smarts_neg)[i*batch_size:]

        else:

            smarts_pos = list(all_smarts_pos)[i * batch_size:(i + 1) * batch_size]

            smarts_neg = list(all_smarts_neg)[i * batch_size:(i + 1) * batch_size]

        product_fps_pos, reaction_fps_pos, useful_smarts_pos = build_data(smarts_pos)

        np.save(args.save_path+"/pos/product_fps_pos_"+str(i)+".npy", product_fps_pos)
        np.save(args.save_path+"/pos/reaction_fps_pos_"+str(i)+".npy", reaction_fps_pos)
        np.save(args.save_path+"/pos/useful_smarts_pos_"+str(i)+".npy", useful_smarts_pos)

        del product_fps_pos, reaction_fps_pos, useful_smarts_pos
        gc.collect()

        product_fps_neg, reaction_fps_neg, useful_smarts_neg = build_data(smarts_neg)

        np.save(args.save_path+"/neg/product_fps_neg_"+str(i)+".npy", product_fps_neg)
        np.save(args.save_path+"/neg/reaction_fps_neg_"+str(i)+".npy", reaction_fps_neg)
        np.save(args.save_path+"/neg/useful_smarts_neg_"+str(i)+".npy", useful_smarts_neg)
        del product_fps_neg, reaction_fps_neg, useful_smarts_neg
        gc.collect()

