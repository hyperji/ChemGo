# -*- coding: utf-8 -*-
# @Time    : 18-8-23 下午3:29
# @Author  : HeJi
# @FileName: module.py
# @E-mail: hj@jimhe.cn


import os
import random
import time
from .mcts_node import MCTSNode, IllegalTransformation
from absl import flags
import numpy as np
from utils import dbg
from rdkit import Chem
from utils import get_product_fingerprint, get_reaction_fingerprint, reaction_from_smart, get_topk_transformation_v2
from hyperparams import Hyperparams as hp
# Ensure that both white and black have an equal number of softpicked moves
#flags.register_validator('softpick_move_cutoff', lambda x: x % 2 == 0)


class IllegalMol(Exception):
    pass


class MCTS_Policy_old(object):
    def __init__(self, expand_network, in_scope_filter,lbl1, lbl2, exp_indexes, num_readouts=1, softpick_move_cutoff = 2, verbosity=0):
        self.expand_network = expand_network
        self.in_scope_filter = in_scope_filter
        self.num_readouts = num_readouts
        self.verbosity = verbosity
        self.root = None
        self.temp_threshold = softpick_move_cutoff
        self.lbl1 =lbl1
        self.lbl2 = lbl2
        self.exp_indexes = exp_indexes
        assert (self.num_readouts > 0)


    def get_state(self):
        return self.root.state if self.root else None

    def get_root(self):
        return self.root

    def get_result_string(self):
        return self.result_string

    def initialize(self, state):
        self.root = MCTSNode(state)
        self.result = 0
        self.result_string = None
        self.comments = []
        self.searches_pi = []
        self.result_string = None

    def suggest_move(self, state):
        ''' Used for playing a single game.
        For parallel play, use initialize_move, select_leaf,
        incorporate_results, and pick_move
        '''
        start = time.time()
        current_readouts = self.root.N
        while self.root.N < current_readouts + self.num_readouts:
            self.mcts_policy()
        if self.verbosity > 0:
            dbg("%d: Searched %d times in %.2f seconds\n\n" % (
                state.n, self.num_readouts, time.time() - start))

        # print some stats on moves considered.
        if self.verbosity > 2:
            dbg(self.root.describe())
            dbg('\n\n')
        if self.verbosity > 3:
            dbg(self.root.state)

        return self.pick_transformation()

    def exec_transform(self, c):
        '''
        Notable side effects:
          - finalizes the probability distribution according to
          this roots visit counts into the class' running tally, `searches_pi`
          - Makes the node associated with this move the root, for future
            `inject_noise` calls.
        '''
        best_move = self.root.local_global_trans_maps[c]
        best_move = self.decoding(best_move)
        mol_index = self.root.local_trans_mol_maps[c]
        self.searches_pi.append(self.root.children_as_pi())
        #self.comments.append(self.root.describe(self.decoding))
        try:
            self.root = self.root.maybe_add_child(best_move, c, mol_index)
        except IllegalTransformation:
            print("Illegal Transformation")
            self.searches_pi.pop()
            #self.comments.pop()
            return False
        self.state = self.root.state  # for showboard
        del self.root.parent.children
        return True  # GTP requires positive result.

    def pick_transformation(self):
        '''Picks a move to play, based on MCTS readout statistics.
        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max.'''
        if self.root.state.n >= self.temp_threshold:
            fcoord = np.argmax(self.root.child_N)
        else:
            cdf = self.root.children_as_pi(squash=True).cumsum()
            cdf /= cdf[-2]  # Prevents passing via softpick.
            selection = random.random()
            fcoord = cdf.searchsorted(selection)
            assert self.root.child_N[fcoord] != 0
        return fcoord

    def decoding(self, transformation):
        return self.lbl1.inverse_transform(self.lbl2.inverse_transform(transformation))

    def encoding(self, smarts):
        return self.lbl2.transform(self.lbl1.transform(smarts))


    def tree_search(self, build_block_mols, parallel_readouts=None):
        if parallel_readouts is None:
            parallel_readouts = hp.parallel_readouts
        leaves = []
        failsafe = 0
        while len(leaves) < parallel_readouts and failsafe < parallel_readouts * 2:
            failsafe += 1
            leaf = self.root.select_leaf(decoder = self.decoding)

            if self.verbosity >= 4:
                print(self.show_path_to_root(leaf))
            # if game is over, override the value estimate with the true score
            if leaf.is_done(build_block_mols):
                value = leaf.state.score(build_block_mols)
                leaf.backup_value(value, up_to=self.root)
                continue
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        if leaves:
            all_unsolved_indexes = []
            for ll in leaves:
                unsolved_mols, unsolved_indexes = ll.state.get_unsolved_mols_and_indexes(
                    build_block_mols=build_block_mols)
                all_unsolved_indexes.append(unsolved_indexes)

            all_results = self.expand_network.run_many(
            [leaf.state for leaf in leaves], all_unsolved_indexes=all_unsolved_indexes, feat_indexes=self.exp_indexes)

            for leaf, all_move_probs in zip(leaves, all_results):
                leaf.revert_virtual_loss(up_to=self.root)
                action_mols_map, topk_probs, topk_transformations = get_topk_transformation_v2(hp.expand_topk,
                                                                                               all_move_probs)
                leaf.incorporate_results(topk_probs,
                                         local_global_trans_maps=topk_transformations,
                                         local_trans_mol_maps=action_mols_map,
                                         build_block_mols=build_block_mols)
        return leaves


    def startup(self, build_block_mols):
        leaf = self.root.select_leaf(decoder = self.decoding)

        unsolved_mols, unsolved_indexes = leaf.state.get_unsolved_mols_and_indexes(
            build_block_mols=build_block_mols)

        if len(unsolved_indexes) == 0:
            return

        all_move_probs = self.expand_network.run(leaf.state, unsolved_indexes=unsolved_indexes,
                                     feat_indexes=self.exp_indexes)
        action_mols_map, topk_probs, topk_transformations = get_topk_transformation_v2(hp.expand_topk,
                                                                                       all_move_probs)
        leaf.incorporate_results(topk_probs,
                                 local_global_trans_maps=topk_transformations,
                                 local_trans_mol_maps=action_mols_map,
                                 build_block_mols=build_block_mols)


        return leaf



    def show_path_to_root(self, node):
        pos = node.state
        diff = node.state.n - self.root.state.n
        if len(pos.recent) == 0:
            return

        path = " ".join(move for move in pos.recent[-diff:])
        if node.state.n >= hp.max_game_length:
            path += " (depth cutoff reached) %0.1f" % node.state.score()
        elif node.state.is_game_over():
            path += " (game over) %0.1f" % node.state.score()
        return path

    def is_done(self, build_block_mols):
        return self.result != 0 or self.root.is_done(build_block_mols)

    def get_num_readouts(self):
        return self.num_readouts

    def set_num_readouts(self, readouts):
        self.num_readouts = readouts

    def set_result(self, is_solved):
        self.result = is_solved
        self.result_string = "Solved" if is_solved else "Not Solved"



class MCTS_Policy(object):
    def __init__(self, network, num_readouts=0, verbosity=0):
        self.network = network
        self.num_readouts = num_readouts or hp.num_readouts
        self.verbosity = verbosity
        self.root = None
        self.temp_threshold = hp.softpick_move_cutoff
        assert (self.num_readouts > 0)


    def get_state(self):
        return self.root.state if self.root else None

    def get_root(self):
        return self.root

    def get_result_string(self):
        return self.result_string

    def initialize(self, state):
        self.root = MCTSNode(state)
        self.result = 0
        self.result_string = None
        self.comments = []
        self.searches_pi = []
        self.result_string = None

    def suggest_move(self, state):
        ''' Used for playing a single game.
        For parallel play, use initialize_move, select_leaf,
        incorporate_results, and pick_move
        '''
        start = time.time()
        current_readouts = self.root.N
        while self.root.N < current_readouts + self.num_readouts:
            self.tree_search()
        if self.verbosity > 0:
            dbg("%d: Searched %d times in %.2f seconds\n\n" % (
                state.n, self.num_readouts, time.time() - start))

        # print some stats on moves considered.
        if self.verbosity > 2:
            dbg(self.root.describe())
            dbg('\n\n')
        if self.verbosity > 3:
            dbg(self.root.state)

        return self.pick_transformation()

    def exec_transform(self, c):
        '''
        Notable side effects:
          - finalizes the probability distribution according to
          this roots visit counts into the class' running tally, `searches_pi`
          - Makes the node associated with this move the root, for future
            `inject_noise` calls.
        '''

        self.searches_pi.append(self.root.children_as_pi())
        self.comments.append(self.root.describe())
        best_transformation = self.root.local_global_trans_maps
        try:
            self.root = self.root.maybe_add_child(c)
        except IllegalTransformation:
            print("Illegal Transformation")
            self.searches_pi.pop()
            self.comments.pop()
            return False
        self.state = self.root.state  # for showboard
        del self.root.parent.children
        return True  # GTP requires positive result.

    def pick_transformation(self):
        '''Picks a move to play, based on MCTS readout statistics.
        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max.'''
        if self.root.state.n >= self.temp_threshold:
            fcoord = np.argmax(self.root.child_N)
        else:
            cdf = self.root.children_as_pi(squash=True).cumsum()
            cdf /= cdf[-2]  # Prevents passing via softpick.
            selection = random.random()
            fcoord = cdf.searchsorted(selection)
            assert self.root.child_N[fcoord] != 0
        return fcoord

    def tree_search(self, parallel_readouts=None):
        if parallel_readouts is None:
            parallel_readouts = hp.parallel_readouts
        leaves = []
        failsafe = 0
        while len(leaves) < parallel_readouts and failsafe < parallel_readouts * 2:
            failsafe += 1
            leaf = self.root.select_leaf()
            if self.verbosity >= 4:
                print(self.show_path_to_root(leaf))
            # if game is over, override the value estimate with the true score
            if leaf.is_done():
                value = leaf.state.score()
                leaf.backup_value(value, up_to=self.root)
                continue
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        if leaves:
            move_probs, values = self.network.run_many(
                [leaf.state for leaf in leaves])
            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob)
        return leaves

    def show_path_to_root(self, node):
        pos = node.state
        diff = node.state.n - self.root.state.n
        if len(pos.recent) == 0:
            return

        path = " ".join(move for move in pos.recent[-diff:])
        if node.state.n >= hp.max_game_length:
            path += " (depth cutoff reached) %0.1f" % node.state.score()
        elif node.state.is_game_over():
            path += " (game over) %0.1f" % node.state.score()
        return path

    def is_done(self):
        return self.result != 0 or self.root.is_done()

    def get_num_readouts(self):
        return self.num_readouts

    def set_num_readouts(self, readouts):
        self.num_readouts = readouts

    def set_result(self, is_solved):
        self.result = is_solved
        self.result_string = "Solved" if is_solved else "Not Solved"
