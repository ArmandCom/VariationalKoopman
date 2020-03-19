from __future__ import absolute_import, division, print_function

import warnings

import torch
import torch.nn as nn

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.util import MultiFrameTensor, get_iarange_stacks, is_validation_enabled, torch_item
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, torch_isnan


def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_iarange_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r

def _get_Gram(M):
    '''
    M: [*, T, f] --  T = n_frames_input
    G: [*, T/2 -1, T/2 -1] -- T/2 -1 = autoregressor_size
    '''
    size_G = M.size(1) // 2 + M.size(1) % 2
    G = torch.zeros(M.size(0), size_G, size_G).cuda()
    # meanK = get_K(M).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
    for i in range(M.size(1) - size_G + 1):
        a = M[:, i:i + size_G]
        G += torch.bmm(a, a.permute(0, 2, 1))  # - meanK # With unbias
    return G

def _get_Kernel(M):
    return torch.bmm(M, M.permute(0, 2, 1))

def _get_matrix_trace(M, flag='k'):
    '''
    :param M: [*, T, f] --  T = n_frames_input
    :return: scalar
    '''
    if flag == 'k':
        vecM = M.contiguous().view(M.size(0), 1, -1)
        tr = torch.bmm(vecM, vecM.permute(0, 2, 1))

        # With unbias
        # meanK = get_K(M).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        # tr = (vecM**2 - meanK).sum(dim=2, keepdim=True)
    elif flag == 'g':
        tr = 0
        size_G = M.size(1) // 2 + M.size(1) % 2
        # meanK = get_K(M).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        for i in range(M.size(1) - size_G + 1):
            a = M[:, i:i + size_G]
            veca = a.contiguous().view(a.size(0), 1, -1)
            tr += torch.bmm(veca, veca.permute(0, 2, 1))

            # With unbias
            # tr += (veca ** 2 - meanK).sum(dim=2, keepdim=True)
    else:
        warnings.warn('Wrong flag: choose either k or g.')
        raise NotImplementedError
    return tr

class Loss(ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide. The gradient estimator includes
    partial Rao-Blackwellization for reducing the variance of the estimator when
    non-reparameterizable random variables are present. The Rao-Blackwellization is
    partial in that it only uses conditional independence information that is marked
    by :class:`~pyro.iarange` contexts. For more fine-grained Rao-Blackwellization,
    see :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`.

    References

    [1] Automated Variational Inference in Probabilistic Programming,
        David Wingate, Theo Weber

    [2] Black Box Variational Inference,
        Rajesh Ranganath, Sean Gerrish, David M. Blei
    """
    def __init__(self, lam, gam, eps, delta):
        super(Loss, self).__init__()
        self.lam = lam # Weight of dynamics loss
        self.gam = gam # Weight of dimensionality loss
        self.eps = eps # Slack variable for local geometry
        self.delta = delta # To ensure logdet stability

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator
        """
        for i in range(self.num_particles):
            guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
            if is_validation_enabled():
                check_model_guide_match(model_trace, guide_trace)
                enumerated_sites = [name for name, site in guide_trace.nodes.items()
                                    if site["type"] == "sample" and site["infer"].get("enumerate")]
                if enumerated_sites:
                    warnings.warn('\n'.join([
                        'Trace_ELBO found sample sites configured for enumeration:'
                        ', '.join(enumerated_sites),
                        'If you want to enumerate sites, you need to use TraceEnum_ELBO instead.']))
            guide_trace = prune_subsample_sites(guide_trace)
            model_trace = prune_subsample_sites(model_trace)

            model_trace.compute_log_prob()
            guide_trace.compute_score_parts()

            if is_validation_enabled():
                for site in model_trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_iarange_nesting)
                for site in guide_trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_iarange_nesting)

            yield model_trace, guide_trace

    def _get_logdet_loss(self, M, delta=1e-5):
        G = _get_Gram(M)
        return torch.logdet(G + delta * torch.eye(G.size(-1)).repeat(G.size(0), 1, 1)).mean() #.cuda()

    def _get_traceK_loss(self, M):
        return -_get_matrix_trace(M, 'k').mean()

    # Neighboring loss is currently obtained by maximizing the likelihood
    # def _get_neigh_loss(self, M, neigh, ori_dist):
    #     loss_l1 = nn.L1Loss(reduction='none')
    #     ori_dist = ori_dist.cuda()
    #     loss_val = loss_l1(_get_neigh_dist(M, neigh),(ori_dist**2))
    #     norm_term = ori_dist + self.eps
    #     return (loss_val/norm_term).mean()

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(guide_trace.log_prob_sum())
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss


    def loss_and_grads(self, model, guide, *args, **kwargs):
        # TODO: add argument lambda --> assigns weights to losses
        # TODO: Normalize loss elbo value if not done
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """

        elbo = 0.0
        dyn_loss = 0.0
        dim_loss = 0.0

        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0
            surrogate_elbo_particle = 0
            log_r = None

            ys = []
            # compute elbo and surrogate elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    elbo_particle = elbo_particle + torch_item(site["log_prob_sum"])
                    surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob_sum"]

                    if site["name"] == "obs":
                        man = site["value"]
                        mean_man = man.mean(dim=1, keepdims=True)
                        man = man - mean_man
                        dyn_loss += self._get_logdet_loss(man, delta=self.delta)  # TODO: Normalize
                        dim_loss += self._get_traceK_loss(man)

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]

                    elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])

                    if not is_identically_zero(entropy_term):
                        surrogate_elbo_particle = surrogate_elbo_particle - entropy_term.sum()

                    if not is_identically_zero(score_function_term):
                        if log_r is None:
                            log_r = _compute_log_r(model_trace, guide_trace)
                        site = log_r.sum_to(site["cond_indep_stack"])
                        surrogate_elbo_particle = surrogate_elbo_particle + (site * score_function_term).sum()

            elbo += elbo_particle / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = any(site["type"] == "param"
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values())

            if trainable_params and getattr(surrogate_elbo_particle, 'requires_grad', False):
                surrogate_loss_particle = -surrogate_elbo_particle / self.num_particles \
                                          +self.lam * dyn_loss \
                                          +self.gam * dim_loss
                surrogate_loss_particle.backward()

        loss = -elbo
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss, dyn_loss, dim_loss, man