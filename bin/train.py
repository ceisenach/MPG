# Train model on Platform 2D environment
import sys
sys.path.append('./') # allows it to be run from parent dir
import os
import numpy as np
import logging
logger = logging.getLogger(__name__)
import sampler
from model import FFNet
from a2c import A2C
import utils
import policy

if __name__ == '__main__':
    #############################
    # SETUP
    parser = utils.experiment_argparser()
    args = parser.parse_args()
    train_config = utils.train_config_from_args(args)
    cumulative_reward_path = os.path.join(train_config['odir'],'%s_cr.npy'%train_config['policy'])

    utils.make_directory(train_config['odir'])
    if not train_config['console']:     
        logging.basicConfig(filename=os.path.join(train_config['odir'],'%s_log_main.log'%train_config['policy']),
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',)
    else:
        logging.basicConfig(level=logging.INFO)

    ###############################
    # MAKE NET AND POLICY
    critic_net = FFNet(in_size = 2,out_size = 1)
    actor_net = FFNet(in_size = 2,out_size = 2)
    plc = None
    if train_config['policy'] == 'angular':
        plc = policy.AngularPolicy(actor_net,train_config['sigma'])
    elif train_config['policy'] == 'gauss':
        plc = policy.GaussianPolicy(actor_net,train_config['sigma'])
    else:
        raise RuntimeError('Not a valid policy: %s'%train_config['policy'])

    ###############################
    # CREATE ENVIRONMENT AND RUN
    algo = A2C(plc,critic_net,train_config['lr'],train_config['gamma'])

    sampler = sampler.BatchSampler(plc,**train_config)
    cumulative_rewards = np.array([]).reshape((0,3))
    cur_update = 0
    finished_episodes = 0
    sampler.reset()

    while cur_update < train_config['num_updates']:
        batch,terminal = sampler.sample()
        algo.update(batch,terminal)
        cr = sampler.cumulative_reward

        # save cumulative rewards
        for i,t in enumerate(terminal):
            if t:
                finished_episodes += 1
                cumulative_rewards = np.concatenate((cumulative_rewards,np.array([cur_update,finished_episodes,cr[i]],ndmin=2)),axis=0)
                np.save(cumulative_reward_path, cumulative_rewards)
                logger.info('Finished Episode: %04d, Update Number: %06d, Cumulative Reward: %.3f'% (finished_episodes,cur_update,cr[i]))

        # checkpoint
        if cur_update % train_config['save_interval'] == 0:
            plc.save_model(os.path.join(train_config['odir'],'%s_model_update_%06d.pt'%(train_config['policy'],cur_update)))
        cur_update += 1