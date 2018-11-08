from .in_scope_filter import In_Scope_Filter
import argparse
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
import argparse
from sklearn.model_selection import train_test_split
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def get_args():
    '''
    get arguments
    :return:
    '''
    parser = argparse.ArgumentParser(description='generator')
    parser.add_argument('--pos_data_path', type=str,
                        default='/home/stein/PycharmProjects/In_Scope_Filter/data/pos',
                        help='pos_data_path')
    parser.add_argument('--neg_data_path', type=str,
                        default='/home/stein/PycharmProjects/In_Scope_Filter/data/neg',
                        help='neg_data_path')
    parser.add_argument('--save_dir', type=str,
                        default='saved_model',
                        help='save_model_path')
    return parser.parse_args()

if __name__ == '__main__':

    arg = get_args()
    if not os.path.exists(arg.save_dir):
        os.mkdir(arg.save_dir)
    isf = In_Scope_Filter()
    args = get_args()
    isf.fit_generator(args.pos_data_path, args.neg_data_path, train_file_indexes=(0,1,2,3,4),
                          save_path=args.save_dir+'/my_model.h5', epochs=10)
    isf.save(args.save_dir+'/my_model.h5')






