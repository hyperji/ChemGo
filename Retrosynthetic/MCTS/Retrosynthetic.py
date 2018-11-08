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

FLAGS = flags.FLAGS


def Retrosynthetic_v1(expand_network, rollout_network, in_scope_filter, state,lbl1, lbl2, exp_indexes, rot_indexes, build_block_mols, verbosity=0):
    """
    TODO Merge with Reinforcement Learning
    With mcts of AlphaGo 2016, also used in paper
    Planning chemical syntheses with deep neural networks and symbolic AI

    Want better performance than the paper above? (No Guarantee)
    -- Try Retrosynthetic_v2

    :param expand_network:
    :param rollout_network:
    :param state:
    :param verbosity:
    :return:
    """

    Searcher = MCTS_Policy_old(expand_network,rollout_network, in_scope_filter, lbl1=lbl1, lbl2=lbl2,
                               exp_indexes=exp_indexes, rot_indexes=rot_indexes,verbosity=verbosity)
    Searcher.initialize(state)
    # Must run this once at the start to expand the root node.
    first_node = Searcher.root.select_leaf()
    unsolved_mols, unsolved_indexes = first_node.state.get_unsolved_mols_and_indexes(build_block_mols=build_block_mols)
    probs = expand_network.run(first_node.state, unsolved_indexes = unsolved_indexes, feat_indexes=exp_indexes)
    probs = np.sum(probs,axis=0)
    first_node.incorporate_results(probs, build_block_mols)
    readouts = FLAGS.num_readouts
    while True:
        current_readouts = Searcher.root.N
        # we want to do "X additional readouts", rather than "up to X readouts".
        while Searcher.root.N < current_readouts + readouts:
            Searcher.mcts_policy()
        transformation = Searcher.pick_transformation()
        transformation = Searcher.decoding(transformation)
        Searcher.exec_transform(transformation)
        if Searcher.root.is_done():
            Searcher.set_result(Searcher.root.state.is_solved())
            break
    return Searcher



def Retrosynthetic_v2(network, state, verbosity=0):
    """
    TODO Merge with Reinforcement Learning
    With mcts of AlphaGo Zero 2017, newest version of AlphaGo
    :param network:
    :param state:
    :param verbosity:
    :return:
    """
    Searcher = MCTS_Policy(network,
                        verbosity=verbosity)

    Searcher.initialize(state)

    # Must run this once at the start to expand the root node.
    first_node = Searcher.root.select_leaf()
    prob = network.run(first_node.state)
    first_node.incorporate_results(prob)
    readouts = FLAGS.num_readouts
    while True:
        current_readouts = Searcher.root.N
        # we want to do "X additional readouts", rather than "up to X readouts".
        while Searcher.root.N < current_readouts + readouts:
            Searcher.tree_search()
        transformation = Searcher.pick_transformation()
        Searcher.exec_transform(transformation)
        if Searcher.root.is_done():
            Searcher.set_result(Searcher.root.state.is_solved())
            break
    return Searcher

