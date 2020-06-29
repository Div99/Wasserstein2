import torch
import utils
import losses
import networks
import itertools
from collections import OrderedDict
from base_model import Base

import wandb

class W2(Base):
    """Wasserstein-2 based model W2GAN"""

    def get_data(self, config):
        """override z with gz"""
        z = utils.to_var(next(self.z_generator))
        gz = self.g(z)
        r = utils.to_var(next(self.r_generator))
        if gz.size() != r.size():
            z = utils.to_var(next(self.z_generator))
            gz = self.g(z)
            r = utils.to_var(next(self.r_generator))
        return r, gz

    # def get_gen_data(self, config, phi):
    #     """override z with gz"""
    #     z = utils.to_var(next(self.z_generator))
    #     gz = GEN(self.g(z))
    #     self.get_grad(gz, config.alpha, reverse=False)
    #     return gz, z

    def get_grad(self, x, alpha=0.01, reverse=False):
        x = Variable(x.data, requires_grad=True)
        if reverse:
            ux = self.psi0(x)
        else:
            ux = self.phi0(x)
        dux = torch.autograd.grad(outputs=ux, inputs=x,
                                  grad_outputs=utils.get_ones(ux.size()),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        Tx = x - alpha * dux
        return Tx

    def define_d(self, config):
        self.phi, self.eps = networks.get_d(config), networks.get_d(config)
        self.d_optimizer = networks.get_optim(itertools.chain(list(self.phi.parameters()),
                                                              list(self.eps.parameters())),
                                                              config.d_lr, config)

    def define_g(self, config):
        self.g = networks.get_g(config, self.phi)
        # self.phi0, self.eps0 = networks.get_g(config, None), networks.get_g(config, None)
        self.g_optimizer = networks.get_optim(self.g.parameters(), config.g_lr, config)

    def psi(self, y):
        return -self.phi(y) + self.eps(y)

    def psi0(self, y):
        return -self.phi0(y) + self.eps0(y)

    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        d_loss = -torch.mean(ux + vy)
        if config.ineq:
            d_loss += losses.ineq_loss(x, y, ux, vy, self.cost, config.lambda_ineq)
        if config.ineq_interp:
            d_loss += losses.calc_interp_ineq(x, y, self.phi, self.psi, self.cost, config.lambda_ineq, losses.ineq_loss)
        if config.eq_phi:
            d_loss += losses.calc_eq(x, tx, self.phi, self.psi, self.cost, config.lambda_eq)
        if config.eq_psi:
            d_loss += losses.calc_eq(ty, y, self.phi, self.psi, self.cost, config.lambda_eq)
        if config.lambda_eps > 0.0:
            d_loss += config.lambda_eps * torch.mean((torch.clamp(self.psi(y), min=0))**2)
        return d_loss

    # def calc_gloss(self, x, y, tx, ty, ux, vy, config):
    #     g_loss = -torch.mean(ux + vy)
    #     if config.ineq:
    #         g_loss += losses.ineq_loss(x, y, ux, vy, self.cost, config.lambda_ineq)
    #     if config.ineq_interp:
    #         g_loss += losses.calc_interp_ineq(x, y, self.phi0, self.psi0, self.cost, config.lambda_ineq, losses.ineq_loss)
    #     if config.eq_phi:
    #         g_loss += losses.calc_eq(x, tx, self.phi0, self.psi0, self.cost, config.lambda_eq)
    #     if config.eq_psi:
    #         g_loss += losses.calc_eq(ty, y, self.phi0, self.psi0, self.cost, config.lambda_eq)
    #     if config.lambda_eps > 0.0:
    #         g_loss += config.lambda_eps * torch.mean((torch.clamp(self.psi0(y), min=0))**2)
    #     return g_loss

    def calc_gloss(self, x, y, ux, vy, config):
        return torch.mean(vy)

    def get_stats(self,  config):
        """print outs"""
        stats = OrderedDict()
        stats['loss/disc'] = self.d_loss
        stats['loss/gen'] = self.g_loss
        stats['loss/total'] = self.d_loss + self.g_loss
        return stats

    def get_networks(self):
        nets = OrderedDict([('phi', self.phi),
                            ('eps', self.eps)])
        nets['gen'] = self.g
        return nets
