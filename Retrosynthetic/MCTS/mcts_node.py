# -*- coding: utf-8 -*-
# @Time    : 18-8-23 下午3:29
# @Author  : HeJi
# @FileName: module.py
# @E-mail: hj@jimhe.cn


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monte Carlo Tree Search implementation.
All terminology here (Q, U, N, p_UCT) uses the same notation as in the
AlphaGo (AG) paper.
"""

import collections
import math
from reaction import single_step_synthesis

from absl import flags
import numpy as np
import copy
from rdkit import Chem
from sklearn.preprocessing import LabelEncoder
from hyperparams import Hyperparams as hp

# 722 moves for 19x19, 162 for 9x9
flags.DEFINE_integer('max_game_length', int(25),
                     'Move number at which game is forcibly terminated')

flags.DEFINE_float('c_puct', 0.96,
                   'Exploration constant balancing priors vs. value net output.')

flags.DEFINE_float('dirichlet_noise_alpha', 0.03 * 361 / (100 ** 2),
                   'Concentrated-ness of the noise being injected into priors.')
flags.register_validator('dirichlet_noise_alpha', lambda x: 0 <= x < 1)

flags.DEFINE_float('dirichlet_noise_weight', 0.25,
                   'How much to weight the priors vs. dirichlet noise when mixing')
flags.register_validator('dirichlet_noise_weight', lambda x: 0 <= x < 1)

flags.DEFINE_integer('num_transformations', int(8000),
                     'all transformation used in experiment')


FLAGS = flags.FLAGS

all_transformations = {}



class IllegalTransformation(Exception):
    pass

class DummyNode(object):
    """A fake node of a MCTS search tree.
    This node is intended to be a placeholder for the root node, which would
    otherwise have no parent node. If all nodes have parents, code becomes
    simpler."""

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)


class State(object):

    def __init__(self, mols, n = 0, recent = tuple()):
        self.mols = mols
        self.n = n
        self.solved = False
        self.recent = recent # a tuple of recent transformations

    def is_solved(self, build_block_mols):
        unsolved_mols = []
        solved = True
        for mol in self.mols:
            if mol not in build_block_mols:
                unsolved_mols.append(mol)
                solved = False
        self.solved = solved
        return self.solved

    def get_solved_ratio(self, build_block_mols):
        unsolved_mols = []
        for mol in self.mols:
            if mol not in build_block_mols:
                unsolved_mols.append(mol)
        solved_ratio = 1 - len(unsolved_mols) / len(self.mols)
        return solved_ratio



    def transform(self, transformation, index, mutate = False):
        the_state = self if mutate else copy.deepcopy(self)

        '''
        if not self.is_transformation_legal(transformation):
            raise IllegalTransformation("transformation at {} is illegal: \n{}".format(
                transformation, self))
        '''
        the_state.n += 1
        the_state.recent += (transformation,)
        target_mol = self.mols[index]

        reactants = single_step_synthesis(transformation, target_mol)

        #print("reactants", reactants)
        if reactants is not None:
            del the_state.mols[index]
            the_state.mols.extend(reactants)
            #print("the_state.mols", the_state.mols)
        else:
            pass#print("None Reactants")
        return the_state


    def score(self, build_block_mols):
        if self.is_solved(build_block_mols):
            z = 1
        elif self.n > hp.max_game_length:
            z = -1
        else:
            z = self.get_solved_ratio(build_block_mols)
        return z

    def get_unsolved_mols_and_indexes(self, build_block_mols):
        unsolved_mols = []
        unsolved_indexes = []
        for i, mol in enumerate(self.mols):
            if mol not in build_block_mols:
                unsolved_mols.append(mol)
                unsolved_indexes.append(i)
        return unsolved_mols, unsolved_indexes






class MCTSNode(object):
    """A node of a MCTS search tree.
    A node knows how to compute the action scores of all of its children,
    so that a decision can be made about which move to explore next. Upon
    selecting a move, the children dictionary is updated with a new node.
    state: A state instance, dict, contains molecules
    transformation: A transformation (coordinate) that led to this state, a flattened coord
            (raw number between 0-num_transformations, with None a pass)
    parent: A parent MCTSNode.
    """

    def __init__(self, state,  transformation=None, trans_id = None,  parent=None):
        """

        :param state:
        :param transformation: the transformation maps from it's patent's state to the current state
        :param parent:
        """
        if parent is None:
            parent = DummyNode()
        self.parent = parent
        self.trans_id = trans_id
        self.transformation = transformation  # transformation that led to this state,
        self.state = state
        self.is_expanded = False
        self.losses_applied = 0  # number of virtual losses on this node
        # using child_() allows vectorized computation of action score.
        self.child_N = np.zeros([hp.num_transformations], dtype=np.float32)
        self.child_W = np.zeros([hp.num_transformations], dtype=np.float32)
        # save a copy of the original prior before it gets mutated by d-noise.
        self.original_prior = np.zeros([hp.num_transformations], dtype=np.float32)
        self.child_prior = np.zeros([hp.num_transformations], dtype=np.float32)
        self.children = {}  # map of transformation to resulting MCTSNode

    def __repr__(self):
        return "<MCTSNode move=%s, N=%s" % (
            self.state.recent[-1:], self.N)

    @property
    def child_action_score(self):
        return self.child_Q + self.child_U

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        """

        :return:
        """
        return (hp.c_puct * math.sqrt(max(1, self.N-1)) *
                self.child_prior / (1 + self.child_N))

    @property
    def Q(self):
        return self.W / (1 + self.N)

    @property
    def N(self):
        return self.parent.child_N[self.trans_id]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.trans_id] = value

    @property
    def W(self):
        return self.parent.child_W[self.trans_id]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.trans_id] = value

    def select_leaf(self, mol_index, lbl1, lbl2):
        current = self
        while True:
            # if a node has never been evaluated, we have no basis to select a child.
            if not current.is_expanded:
                break
            # HACK: if last move was a pass, always investigate double-pass first
            # to avoid situations where we auto-lose by passing too early.

            best_move = np.argmax(current.child_action_score)
            best_transformation = lbl1.inverse_transform(lbl2.inverse_transform(best_move))
            current = current.maybe_add_child(best_transformation,best_move, mol_index)
        return current


    def maybe_add_child(self, transformation, trans_id,  mol_index = None):
        """ Adds child node for transformation if it doesn't already exist, and returns it. """
        if transformation not in self.children:
            assert (mol_index is not None)
            new_state = self.state.transform(transformation, mol_index)
            self.children[transformation] = MCTSNode(
                new_state, transformation=transformation, trans_id=trans_id, parent=self)
            #print(self.children)
        return self.children[transformation]


    def add_virtual_loss(self, up_to):
        """Propagate a virtual loss up to the root node.
        Args:
            up_to: The node to propagate until. (Keep track of this! You'll
                need it to reverse the virtual loss later.)
        """
        self.losses_applied += 1
        # This is a "win" for the current node; hence a loss for its parent node
        # who will be deciding whether to investigate this node again.
        loss = -1
        self.W += loss
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        self.losses_applied -= 1
        revert = 1
        self.W += revert
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def incorporate_results(self, transformation_probabilities, build_block_mols):
        assert transformation_probabilities.shape == (hp.num_transformations,)
        # A finished game should not be going through this code path - should
        # directly call backup_value() on the result of the game.
        assert not self.state.is_solved(build_block_mols=build_block_mols)

        # If a node was picked multiple times (despite vlosses), we shouldn't
        # expand it more than once.
        if self.is_expanded:
            return
        self.is_expanded = True

        # Zero out illegal moves.
        transformation_probs = transformation_probabilities
        scale = sum(transformation_probs)
        if scale > 0:
            # Re-normalize move_probabilities.
            transformation_probs *= 1 / scale

        self.original_prior = self.child_prior = transformation_probs
        # initialize child Q as current node's value, to prevent dynamics where
        # if B is winning, then B will only ever explore 1 move, because the Q
        # estimation will be so much larger than the 0 of the other moves.
        #
        # Conversely, if W is winning, then B will explore all 362 moves before
        # continuing to explore the most favorable move. This is a waste of search.
        #
        # The value seeded here acts as a prior, and gets averaged into Q calculations.
        self.child_W = np.ones([hp.num_transformations], dtype=np.float32)

    def backup_value(self, value, up_to):
        """Propagates a value estimation up to the root node.
        Args:
            value: the value to be propagated
            up_to: the node to propagate until.
        """
        self.N += 1
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self, build_block_mols):
        '''True if the last two moves were Pass or if the position is at a move
        greater than the max depth.
        '''
        return self.state.is_solved(build_block_mols) or self.state.n >= hp.max_game_length

    def inject_noise(self):
        epsilon = 1e-5
        legal_transformations = (1 - self.illegal_transformations) + epsilon
        a = legal_transformations * ([hp.dirichlet_noise_alpha] * hp.num_transformations)
        dirichlet = np.random.dirichlet(a)
        self.child_prior = (self.child_prior * (1 - hp.dirichlet_noise_weight) +
                            dirichlet * hp.dirichlet_noise_weight)

    def children_as_pi(self, squash=False):
        """Returns the child visit counts as a probability distribution, pi
        If squash is true, exponentiate the probabilities by a temperature
        slightly larger than unity to encourage diversity in early play and
        hopefully to move away from 3-3s
        """
        probs = self.child_N
        if squash:
            probs = probs ** .98
        sum_probs = np.sum(probs)
        if sum_probs == 0:
            return probs
        return probs / np.sum(probs)

    def most_visited_path_nodes(self):
        node = self
        output = []
        while node.children:
            next_kid = np.argmax(node.child_N)
            node = node.children.get(next_kid)
            assert node is not None
            output.append(node)
        return output

    def most_visited_path(self):
        output = []
        node = self
        for node in self.most_visited_path_nodes():
            output.append("%s (%d) ==> " % (node.transformation, node.N))

        output.append("Q: {:.5f}\n".format(node.Q))
        return ''.join(output)

    def describe(self):
        sort_order = list(range(hp.num_transformations))
        sort_order.sort(key=lambda i: (
            self.child_N[i], self.child_action_score[i]), reverse=True)
        soft_n = self.child_N / max(1, sum(self.child_N))
        prior = self.child_prior
        p_delta = soft_n - prior
        p_rel = np.divide(p_delta, prior, out=np.zeros_like(
            p_delta), where=prior != 0)
        # Dump out some statistics
        output = []
        output.append("{q:.4f}\n".format(q=self.Q))
        output.append(self.most_visited_path())
        output.append(
            "move : action    Q     U     P   P-Dir    N  soft-N  p-delta  p-rel")
        for key in sort_order[:15]:
            if self.child_N[key] == 0:
                break
            output.append("\n{!s:4} : {: .3f} {: .3f} {:.3f} {:.3f} {:.3f} {:5d} {:.4f} {: .5f} {: .2f}".format(
                key,
                self.child_action_score[key],
                self.child_Q[key],
                self.child_U[key],
                self.child_prior[key],
                self.original_prior[key],
                int(self.child_N[key]),
                soft_n[key],
                p_delta[key],
                p_rel[key]))
        return ''.join(output)
