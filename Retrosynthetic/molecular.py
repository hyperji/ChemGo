#!/usr/bin/env python
# encoding: utf-8
# author :yuanpeng
# created time: 2018年08月13日 星期一 19时14分17秒


import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

'''
This is a class represent a chemical compound

:param  mol: a mol object of the chemical
:param  radius: radius that limit the reaction core of the chemical
:param  recursion_depth: recursion depth that limit the reaction core of the chemical
'''

class Chemical(object):

    def __init__(self, mol, radius, n=100):
        # Canonize the mol object by RDKit, no need of smiles
        self.smiles = Chem.MolToSmiles(mol)
        self.mol = Chem.MolFromSmiles(self.smiles)

        # for atom in self.mol.GetAtoms():
        #     print ('atom aromatic:', atom.GetIsAromatic())

        self.radius = radius
        self.recursion_depth = n

    def get_AAM_list(self):
        AAM_list = []
        for atom in self.mol.GetAtoms():
            AAM = atom.GetAtomMapNum()
            AAM_list.append(AAM)
        return AAM_list

    def get_AAM_dict(self):
        """
        For a molecule, the idx assign b is different with the AAM
        In order to get bond between two know AAM atoms, we need use idx as connection
        :param mol: molecule mol object
        :return: a dict:{key = AAM, value = idx}
        """
        AAM_dict = dict()
        for atom in self.mol.GetAtoms():
            idx = atom.GetIdx()
            AAM = atom.GetAtomMapNum()
            AAM_dict[AAM] = idx
        return AAM_dict

    def convert_IdxtoAAM(self, idx_list):
        """
        find a AAM_list for a given idx_list
        :param idx_list: a list of atoms index
        :return: a AAM list
        """
        AAM_dict = self.get_AAM_dict()
        AAM_list=[]
        for value in idx_list:
            AAM = [k for k,v in AAM_dict.items() if v == value]
            AAM_list = list(set(AAM_list).union(set(AAM)))
        return AAM_list

    def get_bond_feature(self, id1, id2):
        """
        define a bond feature between two atoms
        which is used in extending the reaction environment
        :param AAM1: the mapped number for one atom
        :param AAM2: the mapped number for another atom
        :return: a smarts representation like "C=C, CC, C=0, CO"
        """
        atom1 = self.mol.GetAtomWithIdx(id1)
        atom2 = self.mol.GetAtomWithIdx(id2)
        bond = self.mol.GetBondBetweenAtoms(id1, id2).GetSmarts()
        if atom1.GetAtomicNum() < atom2.GetAtomicNum():
            list = [atom1.GetSymbol(), atom2.GetSymbol()]
        else: list = [atom2.GetSymbol(), atom1.GetSymbol()]
        bond_feat = list[0] + bond + list[1]
        return bond_feat

    def get_atom_feature(self, atom):
        """
        get one atom feature
        which is used to compare if two mapped atoms are different by feature
        :param: the atom and the molecule that the atom belongs
        :return: a dict represent feature of one atom, use neighbours'
                mapped number(identity because every atom has different number)) and bond type
                {'neigh_map': [the mapped_number of neighbors],
                 'neigh_bond': [the bond type(int) of neighbors]}
        """
        idx = atom.GetIdx()
        feat = dict()
        # old_list is the list that before sorted.
        old_map_list = []
        old_bond_list = []
        for neigh in atom.GetNeighbors():
            idn = neigh.GetIdx()
            neigh_map_num = neigh.GetAtomMapNum()
            neigh_bond_type = int(self.mol.GetBondBetweenAtoms(idx, idn).GetBondType())
            old_map_list.append(neigh_map_num)
            old_bond_list.append(neigh_bond_type)
        # return a list of index of components in map_list
        # the index list is sort by number
        # eg: mapped_list = [12, 5, 7] => index_list = [1, 2, 0]
        # where 1 means the index 1 (refer 5 in mapped_list) should be placed first order
        index = np.argsort(old_map_list)
        length = len(old_bond_list)
        # new_list is the old list after sorted
        new_map_list = []
        new_bond_list = []
        for i in range(length):
            order = index[i]
            new_map_list.append(old_map_list[order])
            new_bond_list.append(old_bond_list[order])
        feat['neigh_map'] = new_map_list
        feat['neigh_bond'] = new_bond_list
        return feat

    def get_mol_feature(self):
        """
        get feature collection of a molecule
        :param: mol object of a molecule
        :return: feature_list of a molecule contain every atom's feature
        """
        neigh_map = dict()  # it is used to extend changed atoms to radius 1
        mol_feat = []
        for atom in self.mol.GetAtoms():
            AAM = atom.GetAtomMapNum()
            atom_feat = self.get_atom_feature(atom)
            mol_feat.append([AAM, atom_feat])
            neigh_map[AAM] = atom_feat['neigh_map']
        return mol_feat, neigh_map

    def get_changed_atoms(self, other_mol):
        """
        find the feature changed atom of self.mol compare with another_mol
        :param: self.mol other_mol
        :return: a list contain list of changed atoms
        """
        feat = Chemical.get_mol_feature(self)[0]  # a list of dict={'AAM','atom_feature'}
        other_feat = Chemical.get_mol_feature(other_mol)[0]
        changed_atoms = []
        leaving_atoms = []
        AAM_dict = self.get_AAM_dict()
        # case1: first find the atoms that exist in both but changed.
        for a in feat:
            for b in other_feat:
                # if the same atoms AAM are found, then compare their feature
                # if the same mapped atoms have different feature, then it is the changed_atoms
                if a[0] == b[0]:
                    if a[1] != b[1]:
                        changed_atoms.append(a[0])
        # case2: then find the atom that is not exit which means it is in the leaving group,
        # but we may not include all of them, but only include the atoms
        # which is the first_neigh for any of the changed_atoms
        for a in feat:
            # initial exit as False because we haven't check all the atoms
            exist = False
            for b in other_feat:
                if a[0] == b[0]:  # if the are the same AAM, then exist
                    exist = True
            if exist == False:
                if self.mol.GetNumAtoms() == 1:
                    leaving_atoms.append(a[0])
                else:
                    # only include the non_exist atom is the neigh for changed atoms:
                    for i in range(len(changed_atoms)):
                        AAM = changed_atoms[i]
                        idn = AAM_dict[AAM]
                        for atom in self.mol.GetAtomWithIdx(idn).GetNeighbors():
                            if a[0]==atom.GetAtomMapNum():
                                leaving_atoms.append(a[0])
        changed_atoms = list(set(changed_atoms).union(set(leaving_atoms)))
        return changed_atoms

    def extend_atoms_r(self, changed_atoms, radius=1):
        n = 0
        extend_center = changed_atoms[:]
        neigh_map = self.get_mol_feature()[1]

        while n < radius:
            n += 1
            for j in range(len(changed_atoms)):
                # for every mapped number, find the neighbor(radius=1) atoms' mapped number
                # we can find the neighbor atoms' mapped number from it's atom_feature
                center_atom_AAM = changed_atoms[j]
                neighbor_atom_AAM = neigh_map[center_atom_AAM]
                extend_center = list(set(extend_center).union(set(neighbor_atom_AAM)))
            # update the center by the extend_center after every iteration
            changed_atoms = extend_center[:]
        return extend_center

    def extend_atoms_e(self, this_center):
        """
        extent the changed atom by environment algorithm
        :param: list of changed atoms of reactants and product
        :param: n the maximun recursion depth, n = 100 if user not defined
        :return: reaction core atoms list
        """
        # print(self.mol.GetBondBetweenAtoms(idx, idn).GetSmarts())
        mapped_list = self.get_AAM_dict()
        # initialize the envir for the first recursion
        last_envir = this_center[:]
        this_envir = this_center[:]
        ring_info = self.mol.GetRingInfo()
        ring_tuple = ring_info.AtomRings()
        r = 0
        while True:
            r += 1
            # to record the first_neigh that is the aromatic atoms
            aromatic_list = []
            first_aromatic_atoms = []
            for i in range((len(this_center))):
                AAM = this_center[i]
                idx = mapped_list[AAM]
                atom = self.mol.GetAtomWithIdx(mapped_list[AAM])
                for first_neigh in atom.GetNeighbors():
                    first_list = []
                    idn1 = first_neigh.GetIdx()
                    primary_bond_feat = self.get_bond_feature(idx, idn1)
                    # case1: if the first_neigh is the changed_atom
                    if first_neigh.GetAtomMapNum() in this_center:
                        pass
                    # case2: if the first_neigh is in aromatic ring and the recursion is the first-time
                    # which means that we don't consider aromatic envir if it is second_neigh.
                    # include all the atoms in the aromatic ring as envir
                    elif first_neigh.GetIsAromatic()==True:
                        first_aromatic_atoms.append(first_neigh.GetAtomMapNum())
                        # add the all atoms in the [same] [aromatic] ring associated with first_neigh
                        for i in range(len(ring_tuple)):
                            # find the ring that the first_neigh is in
                            # but the ring may be aromatic or aliphatic
                            if idn1 in ring_tuple[i]:
                                aromatic = True
                                j = 0
                                while aromatic:
                                    index = ring_tuple[i][j]
                                    atom = self.mol.GetAtomWithIdx(index)
                                    j += 1
                                    if atom.GetIsAromatic() == False:
                                        aromatic = False
                                    if j>=len(ring_tuple[i]):
                                        break
                                if aromatic == True:
                                    first_list = self.convert_IdxtoAAM(list(ring_tuple[i]))
                                    aromatic_list = first_list[:]
                                    this_envir = list(set(this_envir).union(set(first_list)))
                    # case3: if the center_atom is C but the bond is not 'CC':
                    # the first_neigh is not in aromatic ring
                    elif atom.GetSymbol() == 'C' and first_neigh.GetIsAromatic()==False \
                            and primary_bond_feat != "CC":
                        first_list.append(first_neigh.GetAtomMapNum())
                        this_envir = list(set(this_envir).union(set(first_list)))
                    # if the first_neigh is not in aromatic ring
                    # case4: primary_bond is 'CC'
                    # case5: center atom is not C
                    elif first_neigh.GetIsAromatic()==False and (primary_bond_feat == "CC"
                                or atom.GetSymbol() != 'C'):
                        # if the center_atom is not single bond with first_neigh 'C':
                        # then add the first_neigh
                        if (first_neigh.GetSymbol()=='C' and int(self.mol.
                        GetBondBetweenAtoms(idx, idn1).GetBondType())>1) or first_neigh.GetSymbol()!='C':
                            first_list.append(first_neigh.GetAtomMapNum())
                            this_envir = list(set(this_envir).union(set(first_list)))
                        # if the center_atom if the single bond with first_neigh 'C':
                        # then check the second_neigh
                        else:
                            neglect_second = True
                            # at first, we don't know if there is only "CC" for first_neigh
                            # so we need check all the second_neigh bonds
                            # second_neigh should exclued the center atoms_self
                            for second_neigh in first_neigh.GetNeighbors():
                                # only check the second_neigh that is not atom self
                                if second_neigh.GetAtomMapNum() != atom.GetAtomMapNum():
                                    idn2 = second_neigh.GetIdx()  # 12
                                    second_bond_feature = self.get_bond_feature(idn1, idn2)
                                    if second_bond_feature != "CC":
                                        neglect_second = False
                            if neglect_second == False:
                                first_list.append(first_neigh.GetAtomMapNum())
                                this_envir = list(set(this_envir).union(set(first_list)))
            if this_envir == last_envir:
                break
            elif r >= self.recursion_depth:
                break
            else:
                # update the changed_atoms, not include the aromatic atoms but the changed_atom itself.
                # so the next recursion the atoms close to aromatic will not be concerned
                this_center = list(set(this_envir)
                                .difference(set(aromatic_list)))
                last_envir = this_envir[:]
        return this_envir

