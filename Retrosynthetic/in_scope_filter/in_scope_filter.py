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
from .data import DataGenerator, DataGenerator_v2


class In_Scope_Filter(object):

    def __init__(self, model_path = None):
        if model_path is not None:
            self.model = load_model(model_path)
        else:
            self.model = self.default_model()
        self.model_path = model_path

    def default_model(self):
        product_ecfp4 = Input(shape=(16384,))
        reaction_ecfp4 = Input(shape=(2048,))
        product = Dense(activation='elu', units=1024)(product_ecfp4)
        reaction = Dense(activation='elu', units=1024)(reaction_ecfp4)
        product = Dropout(0.3)(product)

        product = Highway(activation='elu')(product)
        product = Highway(activation='elu')(product)
        product = Highway(activation='elu')(product)
        product = Highway(activation='elu')(product)
        product = Highway(activation='elu')(product)

        cosine_similarities = Dot(normalize=True, axes=-1)([product, reaction])

        Y = Activation('sigmoid')(cosine_similarities)

        model = Model(input=[product_ecfp4, reaction_ecfp4], output=Y)
        return model


    def fit(self, pos_data_path = None, neg_data_path = None, file_indexes = (0,1,2,3,4), save_path = "saved_model",
              batch_size = 1024, epochs = 100, shuffle = True, valid_ratio = 0.2):
        assert (pos_data_path is not None and neg_data_path is not None)
        all_product = np.load(pos_data_path + "/product_fps_pos_"+str(file_indexes[0])+".npy")
        all_reaction = np.load(pos_data_path + "/reaction_fps_pos_"+str(file_indexes[0])+".npy")
        all_labels = np.tile([1], all_product.shape[0])

        neg_product = np.load(neg_data_path + "/product_fps_neg_"+str(file_indexes[0])+".npy")
        neg_reaction = np.load(neg_data_path + "/reaction_fps_neg_"+str(file_indexes[0])+".npy")
        neg_labels = np.tile([0], neg_product.shape[0])

        all_product = np.concatenate([all_product, neg_product], axis=0)
        all_reaction = np.concatenate([all_reaction, neg_reaction], axis=0)
        all_labels = np.concatenate([all_labels, neg_labels], axis=0)

        del neg_product, neg_labels, neg_reaction
        gc.collect()

        for i in range(1, len(file_indexes)):
            pos_product_pitch = np.load(pos_data_path + "/product_fps_pos_" + str(file_indexes[i]) +  ".npy")
            all_product = np.concatenate([all_product, pos_product_pitch], axis=0)
            pos_labels_pitch = np.tile([1], pos_product_pitch.shape[0])
            all_labels = np.concatenate([all_labels, pos_labels_pitch], axis=0)
            del pos_product_pitch, pos_labels_pitch
            gc.collect()

            pos_reaction_pitch = np.load(pos_data_path + "/reaction_fps_pos_" + str(file_indexes[i]) + ".npy")
            all_reaction = np.concatenate([all_reaction, pos_reaction_pitch], axis=0)
            del pos_reaction_pitch
            gc.collect()

            neg_product_pitch = np.load(neg_data_path + "/product_fps_neg_" + str(file_indexes[i]) + ".npy")
            all_product = np.concatenate([all_product, neg_product_pitch])
            neg_labels_pitch = np.tile([0], neg_product_pitch.shape[0])
            all_labels = np.concatenate([all_labels, neg_labels_pitch])
            del neg_product_pitch, neg_labels_pitch
            gc.collect()

            neg_reaction_pitch = np.load(neg_data_path + "/reaction_fps_neg_" + str(file_indexes[i]) + ".npy")
            all_reaction = np.concatenate([all_reaction, neg_reaction_pitch], axis=0)
            del neg_reaction_pitch
            gc.collect()

        train_idx, test_idx = train_test_split(np.array(list(range(all_labels.shape[0]))), test_size=valid_ratio)


        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=2)
        model_checkpoint = ModelCheckpoint(save_path + '/my_model.h5', monitor='val_acc', verbose=1,
                                           save_best_only=True,
                                           mode='auto')

        hist = self.model.fit([all_product[train_idx], all_reaction[train_idx]], y=all_labels[train_idx],
                         validation_data=([all_product[test_idx], all_reaction[test_idx]], all_labels[test_idx]),
                         shuffle=shuffle, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping, model_checkpoint])


    def fit_generator(self, pos_data_path = None,
                      neg_data_path = None,
                      train_file_indexes = (0,1,2,3,4),
                      vali_file_indexes = None,
                      save_path = "saved_model.h5",
                      batch_size = 1024,
                      epochs = 100,
                      shuffle = True):

        default_generator = DataGenerator_v2(pos_data_path, neg_data_path, shuffle, batch_size, train_file_indexes)
        #vali_generator = DataGenerator_v2(pos_data_path, neg_data_path, shuffle, batch_size, vali_file_indexes)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=2)
        model_checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=1,
                                           save_best_only=True,
                                           mode='auto')
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if vali_file_indexes is not None:
            vali_generator = DataGenerator_v2(pos_data_path, neg_data_path, shuffle, batch_size = 10*batch_size,
                                              file_indexes = vali_file_indexes)
            self.model.fit_generator(default_generator, validation_data=vali_generator, epochs=epochs, callbacks=[early_stopping, model_checkpoint])
        else:
            self.model.fit_generator(default_generator, epochs=epochs)
            self.save(save_path=save_path)

    def predict(self, product_fps, reaction_fps):
        return self.model.predict([product_fps, reaction_fps])


    def run(self, single_product_fps, many_reaction_fps, threshold = 0.5):
        assert (single_product_fps.shape[0] == 1)
        tiled_product_fps = np.tile(single_product_fps, (many_reaction_fps.shape[0], 1))
        #results = np.argmax(self.predict(product_fps=tiled_product_fps, reaction_fps=many_reaction_fps), axis=-1)
        #results.dtype = np.bool
        return (self.predict(product_fps=tiled_product_fps, reaction_fps=many_reaction_fps) > threshold).flatten()



    def save(self, save_path):
        self.model.save(save_path)










