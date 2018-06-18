# Imports
import torch
from torch.autograd import Variable
import math

EPS = 10**(-6)

class BasePolicy(object):
    """
    Base policy class, supports loading and unloading actor network
    """
    def __init__(self,acnet):
        self._net = acnet

    def save_model(self,path):
        sga_sd = self._net.state_dict()
        torch.save(sga_sd,path)

    def load_model(self,path):
        sd = torch.load(path)
        self._net.load_state_dict(sd)

    def parameters(self):
        return self._net.parameters()

    @property
    def net(self):
        return self._net


class GaussianPolicy(BasePolicy):
    """
    Gaussian Policy
    """
    def __init__(self,acnet,sigma):
        super(GaussianPolicy,self).__init__(acnet)
        self._sigma = sigma

    def nll(self,a_t_hat,s_t=None,a_t = None):
        assert(s_t is not None or a_t is not None)
        if a_t is None:
            a_t = self._net(s_t)
        a_t_hat = a_t_hat.unsqueeze(0) if len(a_t_hat.size()) == 1 else a_t_hat
        a_t = a_t.unsqueeze(0) if len(a_t.size()) == 1 else a_t

        d = a_t_hat.size()[1]
        log_prob = - d * math.log(self._sigma) - 0.5*torch.sum((a_t_hat - a_t) ** 2, dim=1) / (self._sigma ** 2)

        assert log_prob.dim() == 1
        return -log_prob

    def action(self,s_t,sample=True):
        if not isinstance(s_t,Variable):
            s_t = Variable(s_t)
        if sample:
            self._net.eval()
        action_mean = self._net(s_t)
        if sample:
            noise = torch.FloatTensor(action_mean.size()).normal_(0, self._sigma)
            action = action_mean.data + noise
            return action
        return action_mean


class AngularPolicy(GaussianPolicy):
    """
    Angular Gaussian Policy -- negative log-likelihood is with respect to the angular Gaussian density.
    """

    def __init__(self,acnet,sigma):
        super(AngularPolicy,self).__init__(acnet,sigma)

    # gets M_{d-1} and M_{d-2}
    def _m_function(self,d, alpha):
        assert(d > 1)
        if type(alpha) == Variable:
            # detach alpha, handle the differential by ourselves
            alpha = Variable(alpha.data)
        def normal01_pdf(x):
            return torch.exp(-(x ** 2) / 2.0) / math.sqrt(2 * math.pi)
        def normal01_cdf(x):
            return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

        m0 = normal01_cdf(alpha)
        m1 = alpha * m0 + normal01_pdf(alpha)
        m = m1

        for i in range(d-1):
            m = alpha * m1 + (d + 1.0) * m0
            m_grad = (d + 1.0) * m1
            m0, m1 = m1, m

        return m1,m0

    def nll(self,a_t_hat,s_t=None,a_t = None):
        assert(s_t is not None or a_t is not None)
        if a_t is None:
            a_t = self._net(s_t)

        a_t_hat = a_t_hat.unsqueeze(0) if len(a_t_hat.size()) == 1 else a_t_hat
        a_t = a_t.unsqueeze(0) if len(a_t.size()) == 1 else a_t

        # basic expressions in angular gaussian log prob
        d = a_t_hat.size()[1]
        xTmu = torch.sum(a_t_hat * a_t + EPS, dim=1)
        xTx = torch.sum(a_t_hat ** 2, dim=1)
        muTmu = torch.sum(a_t ** 2, dim=1)
        xTx_sqrt = torch.sqrt(xTx)
        alpha = (1./self._sigma) * (1./xTx_sqrt) * xTmu
        alpha_sq = alpha ** 2

        # \cM_{d-1}(\alpha)
        m_dm1, m_dm2 = self._m_function(d, alpha)

        # higher level expressions
        term_3 = 0.5 * (alpha_sq - (muTmu / (self._sigma**2)))
        term_4 = (d * m_dm2 / m_dm1) * alpha
        log_prob = term_3 + term_4

        assert log_prob.dim() == 1
        return -log_prob