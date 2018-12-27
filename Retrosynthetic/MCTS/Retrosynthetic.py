# -*- coding: utf-8 -*-
# @Time    : 18-8-23 下午3:29
# @Author  : HeJi
# @FileName: module.py
# @E-mail: hj@jimhe.cn


from .mcts_policy import MCTS_Policy, MCTS_Policy_old
from .mcts_node import State
from absl import app, flags
import time
import numpy as np
from hyperparams import Hyperparams as hp





def Retrosynthetic(network, in_scope_filter, state, lbl1, lbl2,
                   exp_indexes, build_block_mols, parallel_readouts=None,
                   num_readouts = None, verbosity=0):
    """
    TODO Merge with Reinforcement Learning
    With mcts of AlphaGo Zero 2017, newest version of AlphaGo
    :param network:
    :param state:
    :param verbosity:
    :return:
    """

    Searcher = MCTS_Policy_old(network, in_scope_filter, lbl1, lbl2, exp_indexes, verbosity=verbosity,
                               softpick_move_cutoff=hp.softpick_move_cutoff)

    Searcher.initialize(state)

    Searcher.startup(build_block_mols=build_block_mols)

    # Must run this once at the start to expand the root node.
    moves = []
    associated_mols = []
    while True:
        current_readouts = Searcher.root.N
        # we want to do "X additional readouts", rather than "up to X readouts".
        if parallel_readouts is None:
            parallel_readouts = hp.parallel_readouts
        if num_readouts is None:
            num_readouts = hp.num_readouts

        while Searcher.root.N < current_readouts + num_readouts:
            Searcher.tree_search(build_block_mols, parallel_readouts=parallel_readouts)
        move = Searcher.pick_transformation()

        best_move = Searcher.root.local_global_trans_maps[move]
        associated_mol = Searcher.root.local_trans_mol_maps[move]

        best_move = Searcher.decoding(best_move)
        associated_mols.append(associated_mol)
        moves.append(best_move)


        Searcher.exec_transform(c=move)
        if Searcher.root.is_done(build_block_mols=build_block_mols):
            solved = Searcher.root.state.is_solved(build_block_mols=build_block_mols)
            Searcher.set_result(solved)
            break
    return Searcher, moves, associated_mols

