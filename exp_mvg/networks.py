import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import utils

def get_optim(parameters, lr, config):
    return optim.Adam(parameters, lr, [config.beta1, config.beta2])

def get_d(config):
    net = DUAL(config.n_hidden, config.d_n_layers, config.g_norm, config.activation)
    net.apply(weights_init_d)
    if torch.cuda.is_available():
        net.cuda()
    return net

def get_g(config, phi):
    residual = (config.solver != 'bary_ot')
    if not config.share_weights:
        net = GEN_network(config.n_hidden, config.g_n_layers, config.g_norm,
                  config.activation, residual, config.nabla)
        if residual:
            net.apply(weights_init_g)
        if torch.cuda.is_available():
            net.cuda()
        if config.nabla:
            generator = GEN(net)
        else:
            generator = net
    else:
        generator = GEN(phi)
    return generator

class DUAL(nn.Module):
    def __init__(self, n_hidden, n_layers, norm, activation):
        super(DUAL, self).__init__()
        in_h = 28*28
        out_h = n_hidden
        modules = []
        for i in range(n_layers):
            m = nn.Linear(in_h, out_h)
            norm_ms = apply_normalization(norm, out_h, m)
            for nm in norm_ms:
                modules.append(nm)
            modules.append(get_activation(activation))
            in_h = out_h
            out_h = n_hidden
        modules.append(nn.Linear(n_hidden, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        output = self.model(input)
        return output

class GEN(nn.Module):
    def __init__(self, network):
        super(GEN, self).__init__()
        self.network = network
        in_h = 28*28
        self.velocity = torch.zeros_like(in_h)
        
    def forward(self, input):
        inp = input.view(input.size(0), -1)
        inp = Variable(inp.data, requires_grad=True)
        nabla_phi = torch.autograd.grad(self.network(inp), inp,
                            grad_outputs=utils.get_ones(inp.size()),
                            create_graph=True, retain_graph=True,
                            only_inputs=True)[0]
        output = inp - nabla_phi
        output = output.view(*input.size())
        #output = torch.clamp(output, min=-1, max=1)
        return output

class GEN_network(nn.Module):
    def __init__(self, n_hidden, n_layers, norm, activation, residual, nabla):
        super(GEN_network, self).__init__()
        self.residual = residual
        self.nabla = nabla
        in_h = 28*28
        out_h = n_hidden
        modules = []
        for i in range(n_layers):
            m = nn.Linear(in_h, out_h)
            norm_ms = apply_normalization(norm, out_h, m)
            for nm in norm_ms:
                modules.append(nm)
            modules.append(get_activation(activation))
            in_h = out_h
            out_h = n_hidden
        if self.nabla:
            modules.append(nn.Linear(n_hidden, 1))
        else:
            modules.append(nn.Linear(n_hidden, 28*28))
            modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        output = self.model(input.view(input.size(0), -1))
        if self.nabla:
            return output
        output = output.view(*input.size())
        return 2*output + torch.clamp(input, min=-1, max=1) if self.residual else output
        
def get_activation(ac):
    if ac == 'relu':
        return nn.ReLU()
    elif ac == 'elu':
        return nn.ELU()
    elif ac == 'leakyrelu':
        return nn.LeakyReLU(0.2)
    elif ac == 'tanh':
        return nn.Tanh()
    elif ac == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError('activation [%s] is not found' % ac)

def apply_normalization(norm, dim, module):
    """
    Applies normalization `norm` to `module`.
    Optionally uses `dim`
    Returns a list of modules.
    """
    if norm == 'none':
        return [module]
    elif norm == 'batch':
        return [module, nn.BatchNorm1d(dim)]
    elif norm == 'layer':
        return [module, nn.GroupNorm(1, dim)]
    elif norm == 'spectral':
        return [torch.nn.utils.spectral_norm(module, name='weight')]
    else:
        raise NotImplementedError('normalization [%s] is not found' % norm)

def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)
