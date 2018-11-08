#!/usr/bin/env python
# encoding: utf-8
# author :yuanpeng
# created time: 2018年08月23日 星期四 10时16分40秒


import os
import pandas as pd
import json
import argparse
#from ECFP_featurizer import get_product_fp
from reaction import get_reaction_center, Reaction, reset_reaction_mapped_num
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD   #Communicator对象包含所有进程
size = comm.Get_size()
rank = comm.Get_rank()



EMPTY_LIST = [None, '', 'None', '\t', '\t\t']

def extract_mapped_reaction(file_path):
    mapped_reaction_list = []
    reaction_smiles_list = []
    with open(file_path, 'r') as file_r:
        for line in file_r.readlines():
            smiles_list = line.split('\t')
            if '>>' in smiles_list[-1] and len(smiles_list[-1]) > 1:
                mapped_reaction = smiles_list[1].strip().strip('"').split('|')[0].strip()
                mapped_reaction_list.append(mapped_reaction)
                reaction_smiles_list.append(smiles_list[0].strip())
    return mapped_reaction_list, reaction_smiles_list



def save_reaction_center(mapped_reaction_list, save_csv=False, save_name = "reaction_center.csv"):

    mapped_reaction_list = list(set(mapped_reaction_list))
    valid_mapped_reaction_list = []
    valid_reaction_center_list = []
    valid_reaction_center_radius_list0 = []
    valid_reaction_center_radius_list1 = []
    valid_reaction_center_radius_list2 = []
    valid_count = 0
    for ii, mapped_reaction in enumerate(mapped_reaction_list):

        if ii % 1000==0:
            print ('*'*1000)
            print (str(ii)+ ' done')
        try:
            reaction_center, reaction_rule_smiles_radius =  get_reaction_center(mapped_reaction, draw_picture=False)
            valid_reaction_center_radius_list0.append(reaction_rule_smiles_radius[0])
            valid_reaction_center_radius_list1.append(reaction_rule_smiles_radius[1])
            valid_reaction_center_radius_list2.append(reaction_rule_smiles_radius[2])
            valid_reaction_center_list.append(reaction_center)
            valid_mapped_reaction_list.append(mapped_reaction)
            valid_count += 1
        except:
            pass
        print ('valid rate:', valid_count/float(ii+1))
    print ('valid sample number', len(valid_mapped_reaction_list))
    valid_reaction_center_set_list = list(set(valid_reaction_center_list))
    reaction_center_to_label = dict((reaction_center, index) for index, reaction_center in enumerate(valid_reaction_center_set_list))

    synthesis_dataframe = {}
    synthesis_dataframe['mapped_reaction_smiles'] = valid_mapped_reaction_list
    synthesis_dataframe['reaction_center_smiles'] = valid_reaction_center_list
    synthesis_dataframe['reaction_center_radius_smiles0'] = valid_reaction_center_radius_list0
    synthesis_dataframe['reaction_center_radius_smiles1'] = valid_reaction_center_radius_list1
    synthesis_dataframe['reaction_center_radius_smiles2'] = valid_reaction_center_radius_list2



    synthesis_dataframe['label'] = [reaction_center_to_label[reaction_center] for reaction_center in valid_reaction_center_list]

    dataframe = pd.DataFrame(synthesis_dataframe)
    if save_csv:
        dataframe.to_csv(save_name, index=False)

    return dataframe


def clean_dataframe(dataframe, k=3, save_json=False):
    reaction_center_list = dataframe['reaction_center_smiles']
    reaction_center_set = set(reaction_center_list)

    occurred_count_reaction_center = {}
    for reaction_center in reaction_center_list:
        if reaction_center in occurred_count_reaction_center.keys():
            occurred_count_reaction_center[reaction_center] += 1
        else:
            occurred_count_reaction_center[reaction_center] = 1

    occurred_3_reaction_center_list = [reaction_center for reaction_center in occurred_count_reaction_center.keys() if occurred_count_reaction_center[reaction_center] > 2]
    occurred_50_reaction_center_list = [reaction_center for reaction_center in occurred_count_reaction_center.keys() if occurred_count_reaction_center[reaction_center] > 50]

    label_occurred_3 = dict((reaction_center, index) for index, reaction_center in enumerate(occurred_3_reaction_center_list))
    label_occurred_50 = dict((reaction_center, index) for index, reaction_center in enumerate(occurred_50_reaction_center_list))
    if save_json:
        with open("reaction_center_3_to_label.json", 'w') as file_w:
            json.dump(label_occurred_3, file_w)
        with open("reaction_center_50_to_label.json", 'w') as file_w:
            json.dump(label_occurred_50, file_w)

    if k==3:
        clean_3_dataframe = dataframe[dataframe['reaction_center_smiles'] in occurred_3_reaction_center_list]
        return clean_3_dataframe
    elif k==50:
        clean_50_dataframe = dataframe[dataframe['reaction_center_smiles'] in occurred_50_reaction_center_list]
        return clean_50_dataframe
    else:
        return dataframe


def clean_mapped_num(synthesis_smiles):
    synthesis_rule = Reaction(synthesis_smiles)
    reset_reaction_mapped_num(synthesis_rule)
    reaction_rule_smiles = synthesis_rule.ReactionToSmiles()
    reaction_rule_smarts = synthesis_rule.ReactionToSmarts()
    return reaction_rule_smiles, reaction_rule_smarts

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



if __name__=="__main__":
    print("rank = %d,size = %d" % (rank, size))
    process_data_map = ["first_mapped_reaction.smi", "second_mapped_reaction.smi", "third_mapped_reaction.smi", "fourth_mapped_reaction.smi"]

    data_folder_path = "/home/stein/Documents/chemical_reaction_data/mapped_reaction"
    mapped_reaction_list, reaction_smiles_list = extract_mapped_reaction(file_path=data_folder_path+'/'+process_data_map[rank])
    print(len(mapped_reaction_list))
    dataframe = save_reaction_center(mapped_reaction_list, save_csv=False, save_name="reation_center_"+process_data_map[rank].split('.')[0]+".csv")

    #synthesis_data_csv = '/home/yuanpeng/Documents/dataset/mapped_reaction/fourth_synthesis_data.csv'
    #dataframe = pd.read_csv(synthesis_data_csv)

    reaction_center_radius_smiles0 =  dataframe['reaction_center_radius_smiles0']
    reaction_center_radius_smiles1 = dataframe['reaction_center_radius_smiles1']
    reaction_center_radius_smiles2 = dataframe['reaction_center_radius_smiles2']
    reaction_center_smiles_list = dataframe["reaction_center_smiles"]
    clean_reaction_center_radius_smiles0 = []
    clean_reaction_center_radius_smarts0 = []
    clean_reaction_center_radius_smiles1 = []
    clean_reaction_center_radius_smarts1 = []
    clean_reaction_center_radius_smiles2 = []
    clean_reaction_center_radius_smarts2 = []
    for index, synthesis_smiles in enumerate(reaction_center_radius_smiles0):
        if index % 100 == 0:
            print (index)
        clean_reaction_rule_radius_smiles, clean_reaction_rule_radius_smarts = clean_mapped_num(synthesis_smiles)
        clean_reaction_center_radius_smiles0.append(clean_reaction_rule_radius_smiles)
        clean_reaction_center_radius_smarts0.append(clean_reaction_rule_radius_smarts)

    for index, synthesis_smiles in enumerate(reaction_center_radius_smiles1):
        if index % 100 == 0:
            print (index)
        clean_reaction_rule_radius_smiles, clean_reaction_rule_radius_smarts = clean_mapped_num(synthesis_smiles)
        clean_reaction_center_radius_smiles1.append(clean_reaction_rule_radius_smiles)
        clean_reaction_center_radius_smarts1.append(clean_reaction_rule_radius_smarts)

    for index, synthesis_smiles in enumerate(reaction_center_radius_smiles2):
        if index % 100 == 0:
            print (index)
        clean_reaction_rule_radius_smiles, clean_reaction_rule_radius_smarts = clean_mapped_num(synthesis_smiles)
        clean_reaction_center_radius_smiles2.append(clean_reaction_rule_radius_smiles)
        clean_reaction_center_radius_smarts2.append(clean_reaction_rule_radius_smarts)


    clean_reaction_center_smiles = []
    clean_reaction_center_smarts = []
    for index, synthesis_smiles in enumerate(reaction_center_smiles_list):
        if index % 100 == 0:
            print (index)
        clean_reaction_rule_smiles, clean_reaction_rule_smarts = clean_mapped_num(synthesis_smiles)
        clean_reaction_center_smiles.append(clean_reaction_rule_smiles)
        clean_reaction_center_smarts.append(clean_reaction_rule_smarts)
    dataframe['clean_reaction_center_radius_smiles0'] = clean_reaction_center_radius_smiles0
    dataframe['clean_reaction_center_radius_smiles1'] = clean_reaction_center_radius_smiles1
    dataframe['clean_reaction_center_radius_smiles2'] = clean_reaction_center_radius_smiles2
    dataframe["clean_reaction_center_radius_smarts0"] = clean_reaction_center_radius_smarts0
    dataframe["clean_reaction_center_radius_smarts1"] = clean_reaction_center_radius_smarts1
    dataframe["clean_reaction_center_radius_smarts2"] = clean_reaction_center_radius_smarts2
    print("clean_reaction_center_radius_smarts2",clean_reaction_center_radius_smarts2)

    dataframe['clean_reaction_center_smiles'] = clean_reaction_center_smiles
    dataframe['clean_reaction_center_smarts'] = clean_reaction_center_smarts

    dataframe.to_csv("final_"+process_data_map[rank].split('.')[0]+".csv", index=False)



