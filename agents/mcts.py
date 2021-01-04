"""
Adapted from https://github.com/tensorflow/minigo/blob/master/mcts.py
Implementation of the Monte-Carlo tree search algorithm as detailed in the
AlphaGo Zero paper (https://www.nature.com/articles/nature24270).
"""
import math
import random
import collections
import numpy as np

# Exploration Constant
c_PUCT = 1.38
D_NOISE_ALPHA = 0.03  # Dirichlet noise alpha parameter.
TEMP_THRESHOLD = 5  # Number of steps into the episode to select the action with highest probability


class DummyNode:
    """
    Special node that is used as the node above the initial root node
    to prevent having to deal with special cases when traversing the trees
    """

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

    def revert_virtual_loss(self, up_to=None):
        pass

    def add_virtual_loss(self, up_to=None):
        pass

    def revert_visits(self, up_to=None):
        pass

    def backup_value(self, value, up_to=None):
        pass


class MCTSNode:
    """
    Node in the Monte Carlo Search Tree. Each node holds a single state.
    """

    def __init__(self, state, n_actions, env, action=None, parent=None):
        self.env = env
        if parent is None:
            self.depth = 0
            parent = DummyNode()
        else:
            self.depth = parent.depth + 1

        self.parent = parent
        self.action = action
        self.state = state
        self.n_actions = n_actions
        self.is_expanded = False
        self.n_vlosses = 0
        self.child_N = np.zeros([n_actions], dtype=np.float32)
        self.child_W = np.zeros([n_actions], dtype=np.float32)
        # Save a copy of the original prior before it gets mutated by dirichlet noise
        self.original_prior = np.zeros([n_actions], dtype=np.float32)
        self.child_prior = np.zeros([n_actions], dtype=np.float32)
        self.children = {}

    @property
    def N(self):
        """
        Returns the current visit count of the node.
        """
        return self.parent.child_N[self.action]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.action] = value

    @property
    def W(self):
        """
        Returns the current total value of the node.
        """
        return self.parent.child_W[self.action]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.action] = value

    @property
    def Q(self):
        """
        Returns the current action value of the node.
        """
        return self.W / (1 + self.N)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return (c_PUCT * math.sqrt(1 + self.N) * self.child_prior / (1 + self.child_N))

    @property
    def child_action_score(self):
        """
        Action_Score(s,a) = Q(s,a) + U(s,a) as in the paper.
        A high value means the node should be traversed.
        """
        return self.child_Q + self.child_U

    def select_leaf(self):
        """
        Traverses the MCT rooted in the current node until it finds a leaf.
        Nodes are selected according to child_action_score.
        It expands the leaf by adding a dedicated MCTSNode.
        :return: Expanded leaf MCTSNode.
        """
        current = self
        while True:
            current.N += 1
            if not current.is_expanded:
                break
            # Choose action with highest score.
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action):
        """
        Adds a child node for the given action if it does not yet exists, and returns it.
        :param action: Action to take in current state which leads to desired child node.
        :return: Child MCTSNode.
        """
        if action not in self.children:
            # Obtain state following given action.
            new_state = self.env.next_state(self.state, action)
            self.children[action] = MCTSNode(new_state, self.n_actions,
                                             self.env, action=action, parent=self)

        return self.children[action]

    def add_virtual_loss(self, up_to):
        """
        Propagate a virtual loss up to a given node.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses += 1
        self.W -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        """
        Undo adding virtual loss.
        :param up_to: The node to propagate until.
        """
        self.n_vlosses -= 1
        self.W += 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        """
        Revert visit increments.
        Sometimes, repeated calls to select_leaf return the same node.
        This is rare and we're okay with the wasted computation to evaluate
        the position multiple times by the dual_net. But select_leaf has the
        side effect of incrementing visit counts. Since we want the value to
        only count once for the repeatedly selected node, we also have to
        revert the incremented visit counts.
        :param up_to: The node to propagate until.
        """
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)

    def incorporate_estimates(self, action_probs, value, up_to):
        """
        Call if the node has just been expanded via `select_leaf` to
        incorporate the prior action probabilities and state value estimated
        by the neural network.
        :param action_probs: Action probabilities for the current node's state
        predicted by the neural network.
        :param value: Value of the current node's state predicted by the neural
        network.
        :param up_to: The node to propagate until.
        """
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_prior = self.child_prior = action_probs

        self.child_W = np.ones([self.n_actions], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        """
        Propagates a value estimation up to the root node.
        :param value: Value estimate to be propagated.
        :param up_to: The node to propagate until.
        """
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self):
        return self.env.is_done_state(self.state, self.depth)

    def inject_noise(self):
        dirch = np.random.dirichlet([D_NOISE_ALPHA] * self.n_actions)
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def visits_as_probs(self, squash=False):
        """
        Returns the child visit counts as a probability distribution.
        :param squash: If True, exponentiate the probabilities by a temperature
        slightly larger than 1 to encourage diversity in the early steps.
        :return: Numpy array of shape (n_actions).
        """
        probs = self.child_N
        if squash:
            probs = probs ** 0.95
        return probs / np.sum(probs)

    def print_tree(self, level=0):
        node_string = "\033[94m|" + "----" * level
        node_string += "Node: action={}\033[0m".format(self.action)
        node_string += "\n• state:\n{}".format(self.state)
        node_string += "\n• N={}".format(self.N)
        node_string += "\n• score:\n{}".format(self.child_action_score)
        node_string += "\n• Q:\n{}".format(self.child_Q)
        node_string += "\n• P:\n{}".format(self.child_prior)
        print(node_string)
        for _, child in sorted(self.children.items()):
            child.print_tree(level + 1)


class MCTS:
    """
    Represents a Monte-Carlo search tree
    """

    def __init__(self, agent_netw, env, seconds_per_move=None,
                 simulations_per_move=800, num_parallel=8):
        """
        :param agent_netw: Network for predicting action probabilities and
        state value estimate.
        :param env: Static class that defines the environment dynamics,
        :param seconds_per_move: Currently unused.
        :param simulations_per_move: Number of traversals through the tree before performing a step.
        :param num_parallel: Number of leaf nodes to collect before evaluating them in conjuction.
        """
        self.agent_netw = agent_netw
        self.env = env
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.num_parallel = num_parallel
        self.temp_threshold = None

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

        self.root = None

    def initialize_search(self, state=None):
        init_state = self.env.initial_state()
        n_actions = self.env.n_actions
        self.root = MCTSNode(init_state, n_actions, self.env)

        self.temp_threshold = TEMP_THRESHOLD

        self.qs = []
        self.rewards = []
        self.searches_pi = []
        self.obs = []

    def tree_search(self, num_parallel=None):
        """
        Performs multiple simulations in the tree (following trajectories)
        until a given amount of leaves to expand have been encountered.
        Then it expands and evalutes these leaf nodes.
        :param num_parallel: Number of leaf states which the agent network can
        evaluate at once. Limits the number of simulations.
        :return: The leaf nodes which were expanded.
        """
        if num_parallel is None:
            num_parallel = self.num_parallel

        leaves = []
        failsafe = 0
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            leaf = self.root.select_leaf()

            if leaf.is_done():
                value = self.env.get_return(leaf.state, leaf.depth)
                leaf.backup_value(value, up_to=self.root)
                continue

            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)

        # Evaluate the leaf-states all at once and backup the value estimates.
        if leaves:
            action_probs, values = self.agent_netw.step(
                self.env.get_obs_for_states([leaf.state for leaf in leaves]))
            for leaf, action_prob, value in zip(leaves, action_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_estimates(action_prob, value, up_to=self.root)
        return leaves

    def pick_action(self):
        """
        Selects an action for the root state based on the visit counts.
        """
        if self.root.depth > self.temp_threshold:
            action = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            action = cdf.searchsorted(selection)
            assert self.root.child_N[action] != 0
        return action

    def take_action(self, action):
        ob = self.env.get_obs_for_states([self.root.state])
        self.obs.append(ob)
        self.searches_pi.append(
            self.root.visits_as_probs())
        self.qs.append(self.root.Q)
        reward = (self.env.get_return(self.root.children[action].state,
                                      self.root.children[action].depth)
                  - sum(self.rewards))
        self.rewards.append(reward)

        self.root = self.root.maybe_add_child(action)
        del self.root.parent.children


def execute_episode(agent_netw, num_simulations, env):
    mcts = MCTS(agent_netw, env)

    mcts.initialize_search()

    # Run this once at the start, so that noise injection affects the first action
    first_node = mcts.root.select_leaf()
    probs, vals = agent_netw.step(
        env.get_obs_for_states([first_node.state]))
    first_node.incorporate_estimates(probs[0], vals[0], first_node)

    while True:
        mcts.root.inject_noise()
        current_simulations = mcts.root.N

        while mcts.root.N < current_simulations + num_simulations:
            mcts.tree_search()

        # mcts.root.print_tree()

        action = mcts.pick_action()
        mcts.take_action(action)

        if mcts.root.is_done():
            break

    ret = [env.get_return(mcts.root.state, mcts.root.depth) for _ in range(len(mcts.rewards))]

    total_rew = np.sum(mcts.rewards)

    obs = np.concatenate(mcts.obs)
    return obs, mcts.searches_pi, ret, total_rew, mcts.root.state