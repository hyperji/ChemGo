# -*- coding: utf-8 -*-
# @Time    : 18-8-23 下午3:29
# @Author  : HeJi
# @FileName: module.py
# @E-mail: hj@jimhe.cn


import numpy as np
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, BatchNormalization, Add, Dropout, Dot, Highway, Activation
import argparse
import gc
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from .data import DataGenerator_v2, rule_filter
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from rdkit import Chem
from utils import get_product_fingerprint

def get_args():
    '''
    get arguments
    :return:
    '''
    parser = argparse.ArgumentParser(description='generator')
    parser.add_argument('--data_path', type=str,
                        default='/home/stein/Documents/chemical_reaction_data/synthesis_reaction/useful_data/expnet',
                        help='data_path')
    parser.add_argument('--save_path', type=str,
                        default='expnet_models',
                        help='save_path')
    parser.add_argument('--model_name', type=str, default="expnet.h5")
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument("--n_epoches", type=int, default=10)


    return parser.parse_args()


class Expand_Network(object):

    def __init__(self, n_classes, n_feats = 10000, model_path = None):
        self.model_path = model_path
        self.n_classes = n_classes
        self.n_feats = n_feats
        if model_path is not None:
            self.model = load_model(model_path)
        else:
            self.model = self.default_model()


    def default_model(self):
        product_ecfp4 = Input(shape=(self.n_feats,))
        product = Dense(512, activation='elu')(product_ecfp4)
        product = Dropout(0.3)(product)

        product = Highway(activation='elu')(product)
        product = Dropout(rate=0.1)(product)
        product = Highway(activation='elu')(product)
        product = Dropout(rate=0.1)(product)
        product = Highway(activation='elu')(product)
        product = Dropout(rate=0.1)(product)
        product = Highway(activation='elu')(product)
        product = Dropout(rate=0.1)(product)
        product = Highway(activation='elu')(product)
        product = Dropout(rate=0.1)(product)

        product = Dense(self.n_classes, activation="relu")(product)

        Y = Activation('softmax')(product)

        model = Model(input=product_ecfp4, output=Y)
        return model


    def fit(self, data_path = None, file_indexes = (0,1,2,3,4), save_path = "saved_model", model_name = "expnet.h5",
              batch_size = 1024, epochs = 100, shuffle = True, valid_ratio = 0.2):
        assert (data_path is not None)
        all_product = np.load(data_path + "/product_fps_"+str(file_indexes[0])+".npy")
        all_labels = [np.load(data_path+ "/labels_"+ str(file_indexes[i]) + ".npy") for i in range(len(file_indexes))]
        all_labels = np.concatenate(all_labels)
        print(all_product.shape)
        print(all_labels.shape)

        gc.collect()

        for i in range(1, len(file_indexes)):
            product_pitch = np.load(data_path + "/product_fps_" + str(file_indexes[i]) +  ".npy")
            all_product = np.concatenate([all_product, product_pitch], axis=0)
            del product_pitch
            gc.collect()

        train_idx, test_idx = train_test_split(np.array(list(range(all_labels.shape[0]))), test_size=valid_ratio)


        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=2)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model_checkpoint = ModelCheckpoint(os.path.join(save_path, model_name), monitor='val_acc', verbose=1,
                                           save_best_only=True,
                                           mode='auto')
        hist = self.model.fit(all_product[train_idx], y=to_categorical(all_labels[train_idx],num_classes=self.n_classes),
                         validation_data=(all_product[test_idx], to_categorical(all_labels[test_idx],num_classes=self.n_classes)),
                         shuffle=shuffle, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping, model_checkpoint])


    def fit_generator(self, data_path,
                      file_indexes,
                      train_indexes,
                      vali_indexes = None,
                      save_path = "saved_model",
                      model_name = "expnet.h5",
                      batch_size = 1024,
                      epochs = 100,
                      validation_steps = 10,
                      shuffle = True,
                      relabel = None):

        default_generator = DataGenerator_v2(data_path, shuffle, batch_size, file_indexes=file_indexes, used_indexes = train_indexes, relabel = relabel)
        #vali_generator = DataGenerator_v2(pos_data_path, neg_data_path, shuffle, batch_size, vali_file_indexes)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=2)
        model_checkpoint = ModelCheckpoint(os.path.join(save_path, model_name), monitor='val_acc', verbose=1,
                                           save_best_only=True,
                                           mode='auto')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        if vali_indexes is not None:
            if len(vali_indexes) < 10*batch_size:
                vali_batch_size = len(vali_indexes)
            else:
                vali_batch_size = 10*batch_size
            vali_generator = DataGenerator_v2(data_path, shuffle, batch_size = vali_batch_size,
                                              file_indexes = file_indexes, used_indexes = vali_indexes, relabel = relabel)
            self.model.fit_generator(default_generator, validation_data=vali_generator,validation_steps =validation_steps, epochs=epochs, callbacks=[early_stopping, model_checkpoint])
        else:
            self.model.fit_generator(default_generator, epochs=epochs)
            self.save(save_path=save_path)

    def predict(self, product_fps):
        return self.model.predict(product_fps)

    def run(self, state, unsolved_indexes, feat_indexes):
        unsolved_mols_smiles = [state.mols[j] for j in unsolved_indexes]
        unsolved_mols = [Chem.MolFromSmiles(i) for i in unsolved_mols_smiles]
        mol_fps = get_product_fingerprint(unsolved_mols, fp_dim=1000000)
        mol_fps = mol_fps[:,feat_indexes]
        return self.predict(mol_fps)

    def run_many(self, states, all_unsolved_indexes, feat_indexes):
        all_results = []
        length = len(states)
        length_per_state = [len(states[p].mols) for p in range(length)]
        cumsum = np.cumsum(length_per_state)
        all_unsolved_mols_smiles = []
        for i in range(length):
            unsolved_mols_smiles = [states[i].mols[j] for j in all_unsolved_indexes[i]]
            all_unsolved_mols_smiles += unsolved_mols_smiles
        unsolved_mols = [Chem.MolFromSmiles(i) for i in all_unsolved_mols_smiles]
        mol_fps = get_product_fingerprint(unsolved_mols, fp_dim=1000000)
        mol_fps = mol_fps[:, feat_indexes]
        preds = self.predict(mol_fps)
        all_results.append(preds[:cumsum[0]])
        for ii in range(1, len(cumsum)):
            partial_result = preds[cumsum[ii - 1]:cumsum[ii]]
            all_results.append(partial_result)
        return all_results


    def save(self, save_path):
        self.model.save(save_path)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    args = get_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    file_indexes = list(range(16))
    all_labels = [np.load(args.data_path + "/labels_" + str(i) + ".npy") for i in file_indexes]
    all_labels = np.concatenate(all_labels)
    lbl = LabelEncoder()
    try:
        lbl.classes_ = np.load(args.data_path + '/expnet_relabel.npy')
    except:
        print("No previous relabelor exists, creating one from scratch")
        all_labels = lbl.fit_transform(all_labels)
        np.save(args.data_path + "/expnet_relabel.npy", lbl.classes_)
    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)

    try:
        train_indexes = np.load(args.data_path + 'train_indexes.npy')
        vali_indexes = np.load(args.data_path + 'vali_indexes.npy')
    except:
        print("No previous train_indexes or vali_indexes exists, creating them from scratch")
        filters = rule_filter(all_labels, occurance_threshold=10)
        the_idnexes = np.where(filters == True)[0]
        _, vali_indexes = train_test_split(the_idnexes, test_size=0.2)
        train_indexes = np.array([i for i in range(all_labels.shape[0]) if i not in vali_indexes])
        np.save(args.data_path + "/train_indexes.npy", train_indexes)
        np.save(args.data_path, "/vali_indexes.npy", vali_indexes)
    expnet = Expand_Network(n_classes=num_classes, n_feats=10000)
    expnet.fit_generator(data_path=args.data_path,
                         file_indexes=file_indexes,
                         train_indexes=train_indexes,
                         vali_indexes=vali_indexes,
                         batch_size=args.batch_size,
                         epochs=args.n_epoches,
                         shuffle=True,
                         save_path=args.save_path,
                         model_name=args.model_name,
                         validation_steps=1,
                         relabel=lbl)
