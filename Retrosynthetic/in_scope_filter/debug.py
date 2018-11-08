from .in_scope_filter import In_Scope_Filter
from utils import timer
from .data import DataGenerator_v2
import argparse


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
    parser.add_argument('--save_path', type=str,
                        default='save',
                        help='save_model_path')
    return parser.parse_args()


if __name__ =="__main__":
    #isf = In_Scope_Filter()
    args = get_args()
    #isf.fit_generator(args.pos_data_path, args.neg_data_path, train_file_indexes=(0,), vali_file_indexes=(1,), save_path = args.save_path+"/omg.h5")
    #isf.save(args.save_path)
    with timer("DataGenerator"):
        dg = DataGenerator_v2(args.pos_data_path, args.neg_data_path, file_indexes=(0,1,2))


