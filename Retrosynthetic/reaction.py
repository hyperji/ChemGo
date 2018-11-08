#!/usr/bin/env python
# encoding: utf-8
# author :yuanpeng
# created time: 2018年08月10日 星期五 11时55分43秒


import os
import sys
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem import rdChemReactions
from rdkit import Chem, RDConfig

from molecular import Chemical


class Reaction(object):

    def __init__(self, input_reaction, input_format='smiles'):
        self.input_format = input_format
        self.input_reaction = input_reaction

        if self.input_format == 'smiles':
            self.rxn = AllChem.ReactionFromSmarts(self.input_reaction)
        elif self.input_format == 'reaction_file':
            self.rxn = AllChem.ReactionFromRxnFile(self.input_reaction)
        elif self.input_format == 'reaction_mol':
            self.rxn = AllChem.ReactionFromMolecule(self.input_reaction)
        elif self.input_format == '':
            self.rxn = AllChem.ReactionFromRxnBlock(self.input_reaction)

    def Compute2DCoordsForReaction(self):
        AllChem.Compute2DCoordsForReaction(self.rxn)


    def CreateDifferenceFingerprintForReaction(self):
        difference_reaction_fp = AllChem.CreateDifferenceFingerprintForReaction(self.rxn)
        return difference_reaction_fp


    def CreateStructuralFingerprintForReaction(self):
        structural_reaction_fp = AllChem.CreateStructuralFingerprintForReaction(self.rxn)
        return structural_reaction_fp


    def HasAgentTemplateSubstructMatch(self, queryReaction):
        '''
        tests if the agents of a queryReaction are the same as those of a reaction
        :param reaction_rxn:
        :return:
        '''
        return AllChem.HasAgentTemplateSubstructMatch(self.rxn, queryReaction)


    def HasProductTemplateSubstructMatch(self, queryReaction):
        '''

        :param reaction_rxn:
        :return:
        '''
        return AllChem.HasProductTemplateSubstructMatch(self.rxn, queryReaction)


    def HasReactantTemplateSubstructMatch(self, queryReaction):
        '''

        :param reaction_rxn:
        :return:
        '''
        return AllChem.HasReactantTemplateSubstructMatch(self.rxn, queryReaction)



    def HasReactionAtomMapping(self):
        '''

        :param reaction_rxn:
        :return:
        '''
        return AllChem.HasReactionAtomMapping(self.rxn)


    def HasReactionSubstructMatch(self, queryReaction):
        '''

        :return:
        '''
        return AllChem.HasReactionSubstructMatch(self.rxn, queryReaction)


    def IsReactionTemplateMoleculeAgent(self, mol, agentThreshold):
        '''
        tests if a molecule can be classified as an agent depending on the ratio of mapped atoms and a give threshold
        :return:
        '''
        return AllChem.IsReactionTemplateMoleculeAgent(mol, agentThreshold=agentThreshold)


    def ReactionToMolecule(self):
        '''

        :return:
        '''
        return AllChem.ReactionToMolecule(self.rxn)


    def ReactionToRxnBlock(self):
        '''

        :return:
        '''
        return AllChem.ReactionToRxnBlock(self.rxn)


    def ReactionToSmarts(self):
        '''

        :return:
        '''
        return AllChem.ReactionToSmarts(self.rxn)


    def ReactionToSmiles(self):
        '''

        :return:
        '''
        return AllChem.ReactionToSmiles(self.rxn)

    def RemoveMappingNumbersFromReactions(self):
        '''

        :return: None
        '''
        AllChem.RemoveMappingNumbersFromReactions(self.rxn)


    def SanitizeRxn(self):
        self.rxn.SanitizeRxn()


    def addAgentTemplate(self, mol):
        return self.rxn.AddAgentTemplate(mol)


    def AddProductTemplate(self, mol):
        return self.rxn.AddProductTemplate(mol)


    def AddReactantTemplate(self, mol):
        return self.rxn.AddReactantTemplate(mol)


    def AddRecursiveQueriesToReaction(self, queries):
        return self.rxn.AddRecursiveQueriesToReaction(queries)


    def GetAgentTemplate(self, reagent_index):
        return self.rxn.GetAgentTemplate(reagent_index)


    def GetAgents(self):
        return self.rxn.GetAgents()


    def GetNumAgentTemplates(self):
        return self.rxn.GetNumAgentTemplates()


    def GetNumProductTemplates(self):
        return self.rxn.GetNumProductTemplates()


    def GetNumReactantTemplates(self):
        return self.rxn.GetNumReactantTemplates()


    def GetProductTemplate(self, product_index):
        return self.rxn.GetProductTemplate(product_index)


    def GetProducts(self):
        return self.rxn.GetProducts()


    def GetReactantTemplate(self, reactant_index):
        return self.rxn.GetReactantTemplate(reactant_index)


    def GetReactants(self):
        return self.rxn.GetReactants()


    def GetReactingAtoms(self):
        return self.rxn.GetReactingAtoms()


    def initialize(self):
        self.rxn.Initialize()


    def IsInitialized(self):
        return self.rxn.IsInitialized()


    def IsMoleculeAgent(self, mol):
        '''
        whether or not the molecule has a substructure match to one of the agents.
        returns whether or not the molecule has a substructure match to one of the agents.

        :param mol:
        :return: bool
        '''
        return self.rxn.IsMoleculeAgent(mol)


    def IsMoleculeProduct(self, mol):
        return self.rxn.IsMoleculeProduct(mol)


    def IsMoleculeReactant(self, mol):
        return self.rxn.IsMoleculeReactant(mol)


    def RemoveAgentTemplates(self, **kwargs):
        self.rxn.RemoveAgentTemplates(**kwargs)


    def RemoveUnmappedProductTemplates(self, **kwargs):
        self.rxn.RemoveUnmappedProductTemplates(**kwargs)


    def RemoveUnmappedReactantTemplates(self, **kwargs):
        self.rxn.RemoveUnmappedReactantTemplates(**kwargs)


    def RunReactant(self, AtomPairsParameters, unsigned_int):
        '''
        apply the reaction to a single reactant.

        :param AtomPairsParameters:
        :param unsigned_int:
        :return:
        '''

        return self.rxn.RunReactant(AtomPairsParameters, unsigned_int)


    def RunReactants(self, mol_tuple):
        '''

        :param mol_tuple:
        :return:
        '''
        return self.rxn.RunReactants(mol_tuple)


    def ToBinary(self):
        '''

        :return:
        '''
        return self.ToBinary()


    def Validate(self, **kwargs):
        return self.rxn.Validate(**kwargs)


    def get_react_atoms(self, reactant_list, product):
        """
        find the feature changed atom of reactant of product
        :param: reactants, product in mol object
        :return: a list contain list of changed atoms in reactant and list of changed atoms in product
                [[[changed atom in R1],[changed atom in R2]], [changed atoms in P1]]
        """
        # create list storing the changed atoms of each reactant
        R_atom_list = []
        P_atom_list = []
        for i in range(len(reactant_list)):
            r_atom_list = reactant_list[i].get_changed_atoms(product)
            R_atom_list.append(r_atom_list)
            p_atom_list = product.get_changed_atoms(reactant_list[i])
            P_atom_list.append(p_atom_list)
        # for product changed atoms, we need find the union of every changed list when compare with every reactant
        new_P_atom_list = P_atom_list[0]
        for j in range(len(P_atom_list)):
            # p_atom_list contains two lists, each list is the different atoms compare P with R
            # eg: P_atom_list = [[4,5,13~19][1~12,19]]
            # should find the common atoms:[4,5,19]
            new_P_atom_list = list(set(new_P_atom_list).union(set(P_atom_list[j])))
        return R_atom_list, new_P_atom_list


    def get_reactant_product(self, r=2, n=100):

        rdkit_reactant_list = self.GetReactants()
        rdkit_product_list = self.GetProducts()
        reactant_list = []
        product_list = []

        for rdkit_reactant in rdkit_reactant_list:
            reactant = Chemical(rdkit_reactant, r, n)
            reactant_list.append(reactant)

        for rdkit_product in rdkit_product_list:
            product = Chemical(rdkit_product, r, n)
            product_list.append(product)
        return reactant_list, product_list

    def extend_react_atoms_version1(self, reactant_list, product, method='radius', radius = 0):
        """
        extent the changed atom to neigh with radius 1
        :param: reactants, product in mol object
        :param function that used to get the extend reaction atoms, defaults = radius
        :return: same form list after extended
        """
        methods = {
            'radius': Chemical.extend_atoms_r,
            'environment': Chemical.extend_atoms_e,
        }
        get_method = methods[method]
        R_changed, P_changed = self.get_react_atoms(reactant_list, product)
        print ('changed_atom', R_changed, P_changed)
        R_extend = []
        P_extend = []
        # for Reactants:
        for i in range(len(reactant_list)):
            try:
                new_center = get_method(reactant_list[i], R_changed[i], radius = radius)
            except:
                new_center = get_method(reactant_list[i], R_changed[i])
            R_extend.append(new_center)
            P_extend += new_center
        # for products:
        # P_extend = extend_atoms_r(product, ratio, P_changed)
        # the above method for P has the bug that if the aromatic ring broken during reaction,
        # the product reaction center will extend to some unrelated groups:
        # the new method first comcatenate the extend center for reactants
        # and then find union set in product.
        product_AAM_list = product.get_AAM_list()
        P_extend = list(set(P_extend).intersection(set(product_AAM_list)))
        P_extend.extend(P_changed)
        P_extend = list(set(P_extend))
        return R_extend, P_extend


    def get_reaction_rule(self, reactant_list, product, method='radius',limit_symbol=False, radius = 0):
        """
        get the reaction_rule in a mol representation, which is consist of extended changed atoms
        first find the bonds that in the reaction core  as connection and then construct a new mol
        :param: all reactants and products
        :param: the method to get the extend reaction atoms, default = radius
        :return reaction_rule
        """

        R_center, P_center = self.extend_react_atoms_version1(reactant_list, product, method=method, radius = radius)
        R_core_smiles = []
        P_core_smiles = []
        # for every reactants:
        for i in range(len(reactant_list)):
            # find all the bonds between the atoms in the list like AAM=[1,2,3,14] => idx=[x,x,x,x]
            list = R_center[i]
            dict = Chemical.get_AAM_dict(reactant_list[i])
            atoms_to_use = []
            # limit the atom symbol to generate SMILES
            atom_symbol_list = [atom.GetSymbol() for atom in reactant_list[i].mol.GetAtoms()]
            for j in list:
                idx = dict[j]
                atoms_to_use.append(idx)
            if len(atoms_to_use) == 0:
                pass
            else:
                if limit_symbol == True:
                    smiles = Chem.MolFragmentToSmiles(reactant_list[i].mol, atoms_to_use, atomSymbols=atom_symbol_list)
                else:
                    smiles = Chem.MolFragmentToSmiles(reactant_list[i].mol, atoms_to_use)
                R_core_smiles.append(smiles)
        list = P_center
        atoms_to_use = []
        dict = Chemical.get_AAM_dict(product)
        atom_symbol_list = [atom.GetSymbol() for atom in product.mol.GetAtoms()]
        for j in list:
            idx = dict[j]
            atoms_to_use.append(idx)
        if limit_symbol == True:
            smiles = Chem.MolFragmentToSmiles(product.mol, atoms_to_use, atomSymbols=atom_symbol_list)
        else:
            smiles = Chem.MolFragmentToSmiles(product.mol, atoms_to_use)
        P_core_smiles.append(smiles)
        # combine every core_smiles to form a reaction_rule
        SMILES = ""
        for i in range(len(R_core_smiles)):
            if i == 0:
                SMILES += str(R_core_smiles[i])
            else:
                SMILES += '.' + str(R_core_smiles[i])
        SMILES += '>>'
        for j in range(len(P_core_smiles)):
            if j == 0:
                SMILES += str(P_core_smiles[j])
            else:
                SMILES += '.' + str(P_core_smiles[j])
        return SMILES


    def get_changed_atom_mapped_num(self):
        reactant_list = self.GetReactants()
        atom_list = self.GetReactingAtoms()
        print ('atom_list:', atom_list)
        #atom_list = reaction_rxn.get_react_atoms(reactant_list, product_list)

        mapped_num_atom = []
        for ii, reactant_changed_atom in enumerate(atom_list):
            mapped_num_atom.append([])
            reactant = reactant_list[ii]
            for atom_index in reactant_changed_atom:
                atom = reactant.GetAtomWithIdx(atom_index)
                mapped_num_atom[ii].append(atom.GetAtomMapNum())
        return mapped_num_atom


    def draw_reaction(self):
        reaction_img = Draw.ReactionToImage(self.rxn)
        return reaction_img


def convert_product_and_reactant(reaction_center_smiles):
    product_smiles = reaction_center_smiles.split('>>')[-1]
    reactant_smiles = reaction_center_smiles.split('>>')[0]

    return product_smiles + '>>' + reactant_smiles


def single_step_synthesis(synthesis_smiles, product_smiles):
    product_smiles = remove_mapped_num(product_smiles)
    #print ('synthesis_smiles', synthesis_smiles)
    synthesis_rule = Reaction(synthesis_smiles)
    synthesis_rule.RemoveUnmappedProductTemplates()
    synthesis_rule.RemoveUnmappedReactantTemplates()
    #print ('reaction_rule_smiles', reaction_rule_smiles)
    try:
        predict_reactants_mol = synthesis_rule.RunReactants([Chem.MolFromSmiles(product_smiles)])[0]
    except:
        return None
    predict_reactant_smiles = []
    for ii, mol in enumerate(predict_reactants_mol):
        predict_smiles = Chem.MolToSmiles(mol)
        predict_reactant_smiles.append(predict_smiles)
    return predict_reactant_smiles


def single_step_synthesis_v2(mapped_reaction_smiles, reaction_rule_smiles, draw = False, save_path = None):

    reaction_rxn = Reaction(mapped_reaction_smiles)
    reaction_rxn.initialize()
    if draw:
        mapped_reaction_img = Draw.ReactionToImage(reaction_rxn.rxn)

    # product_smiles = Chem.MolToSmiles(Chem.MolFromSmarts(reaction_smiles.split('>>')[-1]))
    reaction_rxn.RemoveMappingNumbersFromReactions()
    products = reaction_rxn.GetProducts()
    products_smiles = [Chem.MolToSmiles(oo) for oo in products]
    longest_product_smiles = products_smiles[0]
    #for ps in products_smiles:
    #    if len(longest_product_smiles)<len(ps):
    #        longest_product_smiles = ps

    #print("product_smiles", longest_product_smiles)

    product_rule_smiles = reaction_rule_smiles.split('>>')[-1]
    reactant_rule_smiles = reaction_rule_smiles.split(">>")[0]

    synthesis_reaction_smiles = product_rule_smiles + '>>' + reactant_rule_smiles
    #print("reaction_rule_smiles", synthesis_reaction_smiles)

    synthesis_rule = Reaction(synthesis_reaction_smiles)
    synthesis_rule.RemoveUnmappedProductTemplates()
    synthesis_rule.RemoveUnmappedReactantTemplates()

    predict_reactants_mol = synthesis_rule.RunReactants([Chem.MolFromSmiles(longest_product_smiles)])[0]
    predict_reactant_smiles = []
    reactant_mol_imgs = []
    for ii, mol in enumerate(predict_reactants_mol):
        predict_smiles = Chem.MolToSmiles(mol)
        predict_reactant_smiles.append(predict_smiles)
        if draw:
            predict_img = Draw.MolToImage(Chem.MolFromSmiles(predict_reactant_smiles[ii]))
            reactant_mol_imgs.append(predict_img)
    if draw:
        synthesis_rule_img = Draw.ReactionToImage(synthesis_rule.rxn)
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            mapped_reaction_img.save(os.path.join(save_path, 'mapped_reaction.png'))
            synthesis_rule_img.save(os.path.join(save_path, 'synthesis_rule.png'))
            for i, mol_img in enumerate(reactant_mol_imgs):
                mol_img.save(os.path.join(save_path, 'reactant_mol_imgs_'+str(i)+'.png'))
    return predict_reactant_smiles


def reset_reaction_mapped_num(synthesis_rule):
    # synthesis_rule = Reaction(synthesis_smiles)
    products_list = synthesis_rule.GetProducts()
    reactants_list = synthesis_rule.GetReactants()

    product_match_mapped_num = []
    product_match_mapped_dict = {}
    for product_mol in products_list:
        for atom in product_mol.GetAtoms():
            mapped_num = atom.GetAtomMapNum()
            product_match_mapped_num.append(mapped_num)
            product_match_mapped_dict[mapped_num] = atom

    reactant_match_mapped_num = []
    reactant_match_mapped_dict = {}
    for reactant_mol in reactants_list:
        for atom in reactant_mol.GetAtoms():
            mapped_num = atom.GetAtomMapNum()
            reactant_match_mapped_num.append(mapped_num)
            reactant_match_mapped_dict[mapped_num] = atom

    match_atom_num = set(product_match_mapped_num) & set(reactant_match_mapped_num)

    rest_product_match_num = set(product_match_mapped_num) - match_atom_num
    rest_reactant_match_num = set(reactant_match_mapped_num) - match_atom_num

    for index, mapped_num in enumerate(match_atom_num):
        product_match_mapped_dict[mapped_num].SetAtomMapNum(index+1)
        reactant_match_mapped_dict[mapped_num].SetAtomMapNum(index+1)
    same_mapped_num = len(match_atom_num) + 1
    product_same_mapped_num = same_mapped_num
    reactant_same_mapped_num = same_mapped_num

    for mapped_num in rest_product_match_num:
        product_match_mapped_dict[mapped_num].SetAtomMapNum(product_same_mapped_num)
        product_same_mapped_num += 1

    for mapped_num in rest_reactant_match_num:
        reactant_match_mapped_dict[mapped_num].SetAtomMapNum(reactant_same_mapped_num)
        reactant_same_mapped_num += 1


def get_reaction_center(mapped_reaction_smiles, draw_picture=True):
    reaction_rxn = Reaction(mapped_reaction_smiles)
    reaction_rxn.initialize()
    reactant_list, product_list = reaction_rxn.get_reactant_product()
    reaction_rule_smiles = reaction_rxn.get_reaction_rule(reactant_list, product_list[0], method='environment')

    reaction_rule_smiles_radius0 = reaction_rxn.get_reaction_rule(reactant_list, product_list[0], method='radius', radius=0)
    reaction_rule_smiles_radius1 = reaction_rxn.get_reaction_rule(reactant_list, product_list[0], method='radius', radius=1)
    reaction_rule_smiles_radius2 = reaction_rxn.get_reaction_rule(reactant_list, product_list[0], method='radius', radius=2)
    reaction_rule_rxn = Reaction(reaction_rule_smiles)

    if draw_picture:
        reaction_image = reaction_rxn.draw_reaction()
        reaction_rule_image = reaction_rule_rxn.draw_reaction()
        reaction_image.show()
        reaction_rule_image.show()
    return reaction_rule_smiles, tuple([reaction_rule_smiles_radius0, reaction_rule_smiles_radius1, reaction_rule_smiles_radius2])

def remove_mapped_num(mapped_smiles):
    mol = Chem.MolFromSmiles(mapped_smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    new_smiles = Chem.MolToSmiles(mol)
    return new_smiles




if __name__ == '__main__':

    reaction_smiles ='C(C)(C)(C)C1=CC=C(C=C1)B(O)O' \
             '.BrC1=C(C=O)C=CC=C1' \
             '>>C(C)(C)(C)C1=CC=C(C=C1)C=1C(=CC=CC1)C=O'

    reaction_smiles_0 = "[O:1]=[C:2]1[CH2:3][S:4][C:5](=[O:6])[NH:7]1.[CH3:8][CH:9]([CH2:10][O:11][c:12]1[cH:13][cH:14][c:15]([CH:16]=[O:17])[cH:18][cH:19]1)[CH:20]1[O:21][C@H:22]([CH2:23][O:24][CH2:25][c:26]2[cH:27][cH:28][cH:29][cH:30][cH:31]2)[C@H:32]2[O:33][C:34]([CH3:35])([CH3:36])[O:37][C@@H:38]12>>[CH3:8][CH:9]([CH2:10][O:11][c:12]1[cH:13][cH:14][c:15](\[CH:16]=[C:3]2/[S:4][C:5](=[O:6])[NH:7][C:2]2=[O:1])[cH:18][cH:19]1)[CH:20]1[O:21][C@H:22]([CH2:23][O:24][CH2:25][c:26]2[cH:27][cH:28][cH:29][cH:30][cH:31]2)[C@H:32]2[O:33][C:34]([CH3:36])([CH3:35])[O:37][C@@H:38]12"
    #reaction_smiles = "[C:1](=[O:2])O.[N:3]>>[C:1](=[O:2])[N:3]"

    reaction_smiles_1 = "[NH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][c:7]1-[c:8]1[cH:9][c:10]([F:11])[c:12]([F:13])[c:14]([F:15])[cH:16]1>>[F:11][c:10]1[cH:9][c:8]([cH:16][c:14]([F:15])[c:12]1[F:13])-[c:7]1[cH:6][cH:5][cH:4][cH:3][c:2]1[I:17]"
    reaction_smiles_2 = "[CH3:1][O:2][c:3]1[cH:14][cH:13][c:10]([CH:11]=[O:12])[cH:9][c:4]1[O:5][CH2:6][CH:7]=[CH2:8].[CH3:15][CH2:16][O:17][C:18](=[O:19])[CH2:20][CH2:21][C:22]([CH3:23])=[O:24]>>[CH3:1][O:2][c:3]1[cH:14][cH:13][c:10](\[CH:11]=[C:20](/[CH2:21][C:22]([CH3:23])=[O:24])[C:18]([OH:17])=[O:19])[cH:9][c:4]1[O:5][CH2:6][CH:7]=[CH2:8]"
    reaction_smiles_3 = "[cH:1]1[cH:6][cH:5][c:4]2[c:3]([cH:2]1)[cH:22][cH:21][c:20]1[cH:19][c:18]3[cH:17][c:16]4[cH:15][c:14]5[c:13]([cH:30][cH:29][c:24]6[cH:25][cH:26][cH:27][cH:28][c:23]56)[cH:12][c:11]4[cH:10][c:9]3[cH:8][c:7]21>>[Cl:31][c:6]1[cH:1][c:2]([Cl:32])[c:3]2[cH:22][cH:21][c:20]([Cl:33])[c:7]([Cl:34])[c:4]2[cH:5]1"
    reaction_smiles_4 = "[CH3:1][C@@H:2]1[CH2:3][CH2:4][C@@H:5]([C:6]([CH3:7])=[CH2:8])[C@@:9]([OH:10])([CH2:11]1)[C:12](=[CH2:13])[c:14]1[cH:15][cH:16][cH:17][cH:18][cH:19]1.[CH3:20][CH2:21][I:22]>>[CH3:20][CH2:21][C@:12]1([CH2:4][CH2:5]\[C:6]([CH3:7])=[CH:8]\[CH2:13][CH2:3][C@@H:2]([CH3:1])[CH2:11][C:9]1=[O:10])[c:14]1[cH:15][cH:16][cH:17][cH:18][cH:19]1"
    reaction_smiles_5 = "[CH3:1][C:2]1([CH3:3])[CH2:4][CH2:5][CH2:6][C@:7]2([CH3:8])[C@@H:9]([CH2:10][OH:11])[C@@:12]3([CH2:13][O:14]3)[CH2:15][CH2:16][CH:17]12>>[CH3:1][C:2]1([CH3:3])[CH2:4][CH2:5][CH2:6][C@@:7]2([CH3:8])[CH:17]1[CH2:16][CH2:15][C@:12]1([CH2:13][O:14]1)[C@@H:9]2[C:10]([OH:11])=[O:18]"

    reaction_smiles_list = [reaction_smiles_0, reaction_smiles_1, reaction_smiles_2, reaction_smiles_3, reaction_smiles_4, reaction_smiles_5]
    reaction_smiles = reaction_smiles_list[1]
    reaction_rxn = Reaction(reaction_smiles)
    reaction_rxn.initialize()

    mapped_num_of_changed_atom = reaction_rxn.get_changed_atom_mapped_num()
    print ("mapped_num_atom", mapped_num_of_changed_atom)

    changed_atom_mapped_num = reaction_rxn.GetReactingAtoms()
    print ('changed_atom_mapped_num', changed_atom_mapped_num)

    reaction_smiles = reaction_rxn.ReactionToSmiles()
    print(reaction_smiles)

    reactant_list, product_list = reaction_rxn.get_reactant_product()
    #changed_atom = reaction_rxn.get_react_atoms(reactant_list, product_list[0])
    #print changed_atom

    #rxn = rdChemReactions.ReactionFromSmarts('[C:1](=[O:2])O.[N:3]>>[C:1](=[O:2])[N:3]')
    reaction_img = Draw.ReactionToImage(reaction_rxn.rxn)
    reaction_img.show()

    reaction_rule_smiles = reaction_rxn.get_reaction_rule(reactant_list, product_list[0], method='radius')
    print ("reaction_rule: ", reaction_rule_smiles)

    reaction_rule = Reaction(reaction_rule_smiles)
    rule_img = Draw.ReactionToImage(reaction_rule.rxn)
    rule_img.show()


    # product_smiles = Chem.MolToSmiles(Chem.MolFromSmarts(reaction_smiles.split('>>')[-1]))
    reaction_rxn.RemoveMappingNumbersFromReactions()
    products = reaction_rxn.GetProducts()[0]
    product_smiles = Chem.MolToSmiles(products)
    print ("products", product_smiles)


    product_rule_smiles = reaction_rule_smiles.split('>>')[-1]
    reactant_rule_smiles = reaction_rule_smiles.split(">>")[0]
    reactant_list = reaction_rule_smiles.split('.')
    synthesis_reaction_smiles = product_rule_smiles+'>>' + reactant_rule_smiles
    print ("reaction_rule_smiles", synthesis_reaction_smiles)
    print ('product_smiles', product_smiles)

    product_img = Draw.MolToImage(Chem.MolFromSmiles(product_smiles))
    product_img.save('product.png')

    synthesis_rule = Reaction(synthesis_reaction_smiles)
    synthesis_rule.RemoveUnmappedProductTemplates()
    synthesis_rule.RemoveUnmappedReactantTemplates()
    # synthesis_rule_img = Draw.ReactionToImage(synthesis_rule.rxn)
    # synthesis_rule_img.show()
    predict_reactants_mol = synthesis_rule.RunReactants([Chem.MolFromSmiles(product_smiles)])[0]
    print ('result:', predict_reactants_mol)
    predict_reactant_smiles = []
    for ii, mol in enumerate(predict_reactants_mol):

        predict_smiles = Chem.MolToSmiles(mol)
        predict_reactant_smiles.append(predict_smiles)


        predict_img = Draw.MolToImage(Chem.MolFromSmiles(predict_reactant_smiles[ii]))
        predict_img.save(str(ii)+'_reslut.png')

    reaction_smiles_1 = "[NH2:1][c:2]1[cH:3][cH:4][cH:5][cH:6][c:7]1-[c:8]1[cH:9][c:10]([F:11])[c:12]([F:13])[c:14]([F:15])[cH:16]1>>[F:11][c:10]1[cH:9][c:8]([cH:16][c:14]([F:15])[c:12]1[F:13])-[c:7]1[cH:6][cH:5][cH:4][cH:3][c:2]1[I:17]"
    """
    reaction_smiles_2 = "[CH3:1][O:2][c:3]1[cH:14][cH:13][c:10]([CH:11]=[O:12])[cH:9][c:4]1[O:5][CH2:6][CH:7]=[CH2:8].[CH3:15][CH2:16][O:17][C:18](=[O:19])[CH2:20][CH2:21][C:22]([CH3:23])=[O:24]>>[CH3:1][O:2][c:3]1[cH:14][cH:13][c:10](\[CH:11]=[C:20](/[CH2:21][C:22]([CH3:23])=[O:24])[C:18]([OH:17])=[O:19])[cH:9][c:4]1[O:5][CH2:6][CH:7]=[CH2:8]"
    reaction_rule_smiles,_ = get_reaction_center(reaction_smiles_2, draw_picture=True)
    print(reaction_rule_smiles)
    """


