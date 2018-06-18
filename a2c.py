# Imports
import torch
from torch.autograd import Variable

class A2C(object):
    """
    A synchronous version of the A3C algorithm
    """
    def __init__(self,policy,critic,lr,gamma):
        self._gamma = gamma
        self._policy = policy
        self._actor = policy.net
        self._critic = critic
        self._optimizer =  torch.optim.SGD(self._actor.parameters(), lr=lr)
        self._updates = 0

    def _batch_prepare(self,batch,terminal):
        """
        compute advantages for each batch
        """
        # Compute advantages
        self._actor.eval(),self._critic.eval()
        S,A,R = batch
        s_T = Variable(S[-1]).unsqueeze(0)
        r_tp1 = R[:-1].view(-1,1)

        U = torch.zeros((r_tp1.size()[0]+1,1))
        if not terminal:
            qT = self._critic(s_T)
            U[-1,:] = qT.data

        length_episode = U.size()[0]
        for i in range(length_episode-1):
            U[length_episode-i-2,:] = r_tp1[length_episode-i-2,:] + self._gamma * U[length_episode-i-1,:]
        U = U[:-1]
        return S[:-1],A[:-1],U

    def _batch_merge(self,batch_list,terminal_list):
        """
        merge independent batches together
        """
        if not isinstance(terminal_list,list):
            return self._batch_prepare(batch_list,terminal_list)
        assert(len(batch_list) == len(terminal_list))
        S,A,U = [],[],[]
        for b,t in filter(lambda bt: bt[0] is not None, zip(batch_list, terminal_list)):
            s,a,u = self._batch_prepare(b,t)
            S.append(s)
            A.append(a)
            U.append(u)
        if len(S) == 0:
            return None
        return torch.cat(S,dim=0),torch.cat(A,dim=0),torch.cat(U,dim=0)

    def update(self,batch_list,terminal_list):
        """
        Update the actor and critic net from sampled minibatches
        """
        batch = self._batch_merge(batch_list,terminal_list)
        if batch is None:
            return

        self._actor.train(),self._critic.train()
        S,A,U = batch
        s_t = Variable(S)
        a_t_hat = Variable(A)

        a_t,v_t = self._actor(s_t),self._critic(s_t)
        U_1 = Variable(U - v_t.data)
        U_2 = Variable(U)

        # actor loss
        lf_actor = torch.mean(U_1.view(-1) * self._policy.nll(a_t_hat,a_t = a_t))

        # critic loss
        lf_critic = torch.mean((U_2 - v_t)**2) * 2.

        # update
        lf = lf_actor + lf_critic
        self._optimizer.zero_grad()
        lf.backward()
        self._updates += 1
        self._optimizer.step()