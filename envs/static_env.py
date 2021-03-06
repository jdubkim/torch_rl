class StaticEnv:
    """
    Abstract class for a static environment. A static environment follows the
    same dynamics as a normal, stateful environment but without saving any
    state inside. As a consequence, all prior information (e.g. the current
    state) has to be provided as a parameter.
    The MCTS algorithm uses static environments because during the tree search,
    it jumps from one state to another (not following the dynamics), such that
    an environment which stores a single state does not make sense.
    """

    def next_state(self, state, action):
        """
        Given the current state of the environment and the action that is
        performed in that state, returns the resulting state.
        :param state: Current state of the environment.
        :param action: Action that is performed in that state.
        :return: Resulting state.
        """
        raise NotImplementedError

    def is_done_state(self, state, step_idx):
        """
        Given the state and the index of the current step, returns whether
        that state is the end of an episode, i.e. a done state.
        :param state: Current state.
        :param step_idx: Index of the step at which the state occurred.
        :return: True, if the step is a done state, False otherwise.
        """
        raise NotImplementedError

    def initial_state(self):
        """
        Returns the initial state of the environment.
        """
        raise NotImplementedError

    def get_obs_for_states(self, states):
        """
        Some environments distinguish states and observations. An observation
        can be a subset (e.g. in Poker, state is all cards in game, observation
        is cards on player's hand) or superset of the state (i.e. observations
        add additional information).
        :param states: List of states.
        :return: Numpy array of observations.
        """
        raise NotImplementedError

    def get_return(self, state, step_idx):
        """
        Returns the return that the agent has achieved so far when he is in
        a given state after a given number of steps.
        :param state: Current state that the agent is in.
        :param step_idx: Index of the step at which the agent reached that
        state.
        :return: Return the agent has achieved so far.
        """
        raise NotImplementedError