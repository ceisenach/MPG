# basic navigation task

# IMPORTS
import numpy as np
import random
import gym
from gym import spaces
from gym.envs.registration import register

register(
    id='Platform2D-v1',
    entry_point='environment:Platform2D',
)

class Platform2D(gym.Env):
    """
    2D Platform Navigation Task.
    Action space: R^2, interpreted as a direction
    State space: R^2
    """
    def __init__(self):
        self.__version__ = "1.0.0"
        print("Platform2D - Version {}".format(self.__version__))
        self.action_shape = (2,)
        self.state_shape = (2,)

        # Action Space
        high = np.array([np.finfo(np.float32).max],dtype = np.float32)
        self.action_space = spaces.Box(-high,high,dtype=np.float32)

        # environment config
        self.MARGIN = np.array([1.0,1.0],dtype = np.float32)
        self.START_POS = np.array([-1.0,-1.0],dtype = np.float32)
        self.END_POS = np.array([1.0,1.0],dtype = np.float32)
        self.BOUND_LOW = self.START_POS - self.MARGIN
        self.BOUND_HIGH = self.END_POS + self.MARGIN
        self.EPS = 0.01
        self.SPEED = 0.005
        self.LOG_INTERVAL = 50
        self.MAX_DIST = 2.15

        # State Space
        self.observation_space = spaces.Box(self.BOUND_LOW, self.BOUND_HIGH, dtype=np.float32)

        # initial state config
        self._reset()


    def _step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        action = np.array(action) # make sure it is array
        if np.linalg.norm(self._agent_location - self.END_POS) < self.EPS:
            raise RuntimeError("Episode is done")

        # get old state
        s_t = self.state

        # update state
        self.time += 1
        self._update_location(action)

        # return (S,R,T,I)
        s_tp1 = self.state
        oob = self._out_of_bounds(s_tp1) # check if in bounds
        r_tp1 = self._reward(s_t,s_tp1)
        terminal = (np.linalg.norm(self._agent_location - self.END_POS) < self.EPS) or oob

        return s_tp1, r_tp1, terminal, {}

    def _reward(self,old_state,new_state):
        # difference of potential function
        old_distance = np.linalg.norm(self.END_POS - old_state)
        new_distance = np.linalg.norm(self.END_POS - new_state)
        return old_distance - new_distance

    def _out_of_bounds(self,state):
        proj_location = np.maximum(self.BOUND_LOW,np.minimum(state,self.BOUND_HIGH))
        if np.linalg.norm(state - proj_location) != 0.0:
            return True
        return False

    def _update_location(self,action):
        norm_action = action / np.linalg.norm(action)
        new_location = self._agent_location + self.SPEED*norm_action
        self._agent_location = new_location

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.time = 0
        self._agent_location = self.START_POS

    @property
    def state(self):
        return self._agent_location
    

    def _render(self, mode='human', close=False):
        return

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed()