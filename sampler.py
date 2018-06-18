import gym
import torch
import logging
logger = logging.getLogger(__name__)
import environment
from utils import MultiRingBuffer


def _torch_from_float(val):
    tval = torch.FloatTensor(1)
    tval[0] = float(val)
    return tval

class Sampler(object):
    """
    Samples batches using policy for an environment specified by kwargs
    """ 
    def __init__(self,policy,**kwargs):
        env = gym.make(kwargs['env'])
        self._base_env = env
        self._policy = policy
        self._N = kwargs['N']
        self._gamma = kwargs['gamma']
        self._cr = 0.0
        self._terminal = False
        self._experience_buffer = MultiRingBuffer([self._base_env.state_shape,self._base_env.action_shape,(1,)],self._N+1)

    def sample(self):
        """
        Get a batch of samples
        """
        if self._terminal:
            self.reset()

        s_t = torch.from_numpy(self._base_env.state)
        while (self._experience_buffer.length < self._N) and  not self._terminal:
            # step environment and append to experience buffer
            a_t = self._policy.action(s_t)
            s_tp1,r_tp1_f,self._terminal,_ = self._base_env.step(a_t)
            s_tp1,r_tp1 = torch.from_numpy(s_tp1),_torch_from_float(r_tp1_f)
            self._experience_buffer.append(s_t,a_t,r_tp1)
            self._cr += self._gamma*r_tp1_f
            s_t = s_tp1

        if self._terminal:
            logger.info('FINSHED -- Final Location (%.3f,%.3f), Cum. Reward: %.3f' % tuple(list(s_t.numpy()) + [self._cr]))

        # check if longer than 1
        if self._experience_buffer.length > 1:
            # s_tp1 used to bootstrap, a and r ignored
            self._experience_buffer.append(s_tp1,a_t,r_tp1)             
            batch = self._experience_buffer.get_data()
            self._experience_buffer.reset()
            return batch,self._terminal

        return None,self._terminal

    def reset(self):
        """
        Resets underlying environment object
        """
        self._base_env.reset()
        self._cr = 0.0
        self._terminal = False

    @property
    def cumulative_reward(self):
        return self._cr


class BatchSampler(object):
    """
    Sample from multiple independent copies of the same environment.
    """
    def __init__(self,policy,**kwargs):
        self._samplers = []
        for i in range(kwargs['num_env']):
            smp = Sampler(policy,**kwargs)
            self._samplers.append(smp)

    def sample(self):
        """
        Get multiple minibatches of samples
        """
        bl,tl = [],[]
        for smp in self._samplers:
            b,t = smp.sample()
            bl.append(b)
            tl.append(t)
        return bl,tl

    def reset(self):
        """
        Resets underlying environment objects
        """
        for smp in self._samplers:
            smp.reset()

    @property
    def cumulative_reward(self):
        return [smp.cumulative_reward for smp in self._samplers]