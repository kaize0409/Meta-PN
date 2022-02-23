import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical


@torch.no_grad()
def virtual_step_update(params, grads, eta, opt):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt)
    else:
        return update_params_adam(params, grads, opt)


def update_params_sgd(params, grads, eta, opt):
    # supports SGD-like optimizers
    params_updated = []

    wdecay = opt.defaults.get('weight_decay', 0.)
    momentum = opt.defaults.get('momentum', 0.)
    # eta = opt.defaults["lr"]
    for i, param in enumerate(params):
        if grads[i] is None:
            params_updated.append(param)
            continue
        momentum = opt.state[param].get('momentum_buffer', 0.) * momentum
        params_updated.append(param - (grads[i] + param * wdecay + momentum) * eta)  # eta is the learning tate

    return params_updated


def update_params_adam(params, grads, opt):
    ans = []
    group = opt.param_groups[0]
    assert len(opt.param_groups) == 1
    for p, grad in zip(params, grads):
        if grad is None:
            ans.append(p)
            continue
        amsgrad = group['amsgrad']
        state = opt.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        step_size = group['lr'] / bias_correction1

        # ans.append(torch.addcdiv(p, -step_size, exp_avg, denom))
        ans.append(p.addcdiv_(exp_avg, denom, value=-step_size))

    return ans


def _dot(grad_a, grad_b):
    return sum([torch.dot(gv[0].view(-1), gv[1].view(-1)) for gv in zip(grad_a, grad_b) if gv[0] is not None and gv[1] is not None])


def meta_loss_unrolled_backward(main_net, main_opt, meta_net, x_pseudo, batch_idx, x_gold, y_gold, lr_main):
    """
    theta: parameters of main_net
    phi: parameters of meta_net
    """
    # given current meta net, get corrected labels

    y_pseudo = meta_net(batch_idx).detach()

    # copy main parameters for recovery
    theta = [param.data.clone() for param in [i for i in main_net.parameters() if i.requires_grad]]

    # compute one-step gradient w'
    logit_pseudo = main_net(x_pseudo)
    loss_pseudo = main_net.soft_cross_entropy(logit_pseudo, y_pseudo)
    grads_theta = torch.autograd.grad(loss_pseudo, main_net.parameters())
    # test = torch.autograd.grad(loss_pseudo, meta_net.parameters())
    theta_prime = virtual_step_update(main_net.parameters(), grads_theta, lr_main, main_opt)

    # update w as w'
    for i, param in enumerate([param for param in main_net.parameters() if param.requires_grad]):
        param.data = theta_prime[i].data

    # compute unrolled loss with updated main_net
    logit_gold = main_net(x_gold)
    loss_gold = main_net.soft_cross_entropy(logit_gold, y_gold)

    # compute  d_w' L_{val}(w', α)
    grads_theta_prime = torch.autograd.grad(loss_gold, main_net.parameters())

    # revert from w' to w for main net
    for i, param in enumerate([param for param in main_net.parameters() if param.requires_grad]):
        param.data = theta[i]

    # compute hessian
    hessian = compute_hessian(meta_net, main_net, grads_theta_prime, x_pseudo, batch_idx)

    # update final gradient of meta-net: - xi*hessian
    with torch.no_grad():
        for param, h in zip(meta_net.parameters(), hessian):
            param.grad = - lr_main * h


def compute_hessian(meta_net, main_net, dw, x_pseudo, batch_idx):
    """
    grads_main_new: dw = dw` { L_val(w`, alpha) }
    w+ = w + eps * dw
    w- = w - eps * dw
    hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
    eps = 0.01 / ||dw||
    """
    norm = torch.cat([w.view(-1) for w in dw]).norm()
    eps = 0.1 / norm


    # w+ = w + eps*dw`
    with torch.no_grad():
        for p, d in zip(main_net.parameters(), dw):
            p += eps * d
    y_pseudo = meta_net(batch_idx)
    logits = main_net(x_pseudo)
    loss_p = main_net.soft_cross_entropy(logits, y_pseudo)
    meta_gradient_pos = torch.autograd.grad(loss_p, meta_net.parameters())  # dalpha { L_trn(w+) }

    # w- = w - eps*dw`
    with torch.no_grad():
        for p, d in zip(main_net.parameters(), dw):
            p -= 2. * eps * d
    y_pseudo = meta_net(batch_idx)
    logits = main_net(x_pseudo)
    loss_n = main_net.soft_cross_entropy(logits, y_pseudo)
    meta_gradient_neg = torch.autograd.grad(loss_n, meta_net.parameters())  # dalpha { L_trn(w-) }

    # recover w
    with torch.no_grad():
        for p, d in zip(main_net.parameters(), dw):
            p += eps * d

    hessian = [(p - n) / 2. * eps for p, n in zip(meta_gradient_pos, meta_gradient_neg)]
    return hessian
