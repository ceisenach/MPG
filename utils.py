import numpy as np
import torch
import os
import argparse
import time

def make_directory(dirpath):
    os.makedirs(dirpath,exist_ok=True)

def experiment_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument("-o", "--odir", type=str, default=None, help="output directory")
    parser.add_argument("-p",'--policy' ,type=str, default='angular', help="policy type to use")
    parser.add_argument("-u",'--num_updates', type=float, default=1e4, help="number of gradient updates")
    parser.add_argument("-g","--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--sigma", type=float, default=0.1, help="policy variance")
    parser.add_argument("-N", type=int, default=10, help="Use N-Step bootstrapped target")
    parser.add_argument("-s",'--save_interval', type=float, default=1e3, help="Model Save Interval")
    parser.add_argument("-l",'--log_interval', type=int, default=50, help="Log Interval")
    parser.add_argument("-V",'--num_env', type=int,default=1, help="number of simultaneous environments to use")
    parser.add_argument("-E",'--env', type=str, default='Platform2D-v1', help="Environment to use")
    parser.add_argument("-co", "--console", action="store_true", help="log to console")
    return parser

def train_config_from_args(args):
    experiment_config = {'sigma' : args.sigma,
                         'gamma' : args.gamma,
                         'lr' : args.lr,
                         'policy' : args.policy,
                         'num_updates' : int(args.num_updates),
                         'N' : args.N,
                         'num_env': args.num_env,
                         'env' : args.env,
                         'console' : args.console,
                         'odir' : args.odir if args.odir is not None else 'out/experiment_%s' % time.strftime("%Y.%m.%d_%H.%M.%S"),
                         'save_interval' : int(args.save_interval)}

    return experiment_config


class MultiRingBuffer(object):
    """
    Ring buffer that supports multiple data of different widths
    """
    def __init__(self, experience_shapes, max_len):
        assert isinstance(experience_shapes, list)
        assert len(experience_shapes) > 0
        assert isinstance(experience_shapes[0], tuple)
        self.maxlen = max_len
        self.start = 0
        self.length = 0
        self.dataList = [torch.FloatTensor(max_len, *shape).zero_() for shape in experience_shapes]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return [data[(self.start + idx) % self.maxlen] for data in self.dataList]

    def append(self, *vs):
        """
        Append to buffer
        """
        vs = list(vs)
        for i, v in enumerate(vs):
            if isinstance(v, np.ndarray):
                vs[i] = torch.from_numpy(v)
        if self.length < self.maxlen:
            # Space Available
            self.length += 1
        elif self.length == self.maxlen:
            # No Space, remove the first item
            self.start = (self.start + 1) % self.maxlen
        else:
            # Should not happen
            raise RuntimeError()
        for data, v in zip(self.dataList, vs):
            data[(self.start + self.length - 1) % self.maxlen] = v.squeeze()

    def reset(self):
        """
        Clear replay buffer
        """
        self.start = 0
        self.length = 0

    def get_data(self):
        """
        Get all data in the buffer
        """
        if self.length < self.maxlen:
            return [data[0:self.length] for data in self.dataList]
        return self.dataList