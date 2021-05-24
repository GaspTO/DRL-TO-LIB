class Base_Exploration_Strategy:
    """ receives the agent as to get full control over
    the current information """

    def __init__(self,exploration=True):
        self.exploration = True

    def perturb_action_for_exploration_purposes(self, action_vector, mask, info=None):
        raise NotImplementedError

    def turn_off_exploration(self):
        self.exploration = False

    def turn_on_exploration(self):
        self.exploration = True

    def is_on_exploration_mode(self):
        return self.exploration

    def reset(self):
        """Resets the noise process"""
        raise NotImplementedError