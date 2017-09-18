import theano
import theano.tensor as tensor
from theano.updates import OrderedUpdates
import numpy

from utils import itemlist
from utils import dup_shared_var_list, is_dup_params

profile = False


# todo: Add return grad norm value.


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8, **kwargs):
    g2 = kwargs.pop('g2', None)
    given_imm_data = kwargs.pop('given_imm_data', None)
    dump_imm = kwargs.pop('dump_imm', False)
    stochastic_updates = kwargs.pop('stochastic_updates', OrderedUpdates())

    if g2 is None:
        outputs = cost
    else:
        outputs = [cost, g2]

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, outputs, updates=gsup + stochastic_updates, profile=profile)

    updates = []

    ms = []
    vs = []

    if given_imm_data is not None:
        t_prev = theano.shared(given_imm_data[0])
    else:
        t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2 ** t) / (1. - beta1 ** t)

    for i, (p, g) in enumerate(zip(tparams.values(), gshared)):
        if given_imm_data is not None:
            m = theano.shared(given_imm_data[1][i], p.name + '_mean')
            v = theano.shared(given_imm_data[2][i], p.name + '_variance')
        else:
            m = theano.shared(p.get_value() * 0., p.name + '_mean')
            v = theano.shared(p.get_value() * 0., p.name + '_variance')

        ms.append(m)
        vs.append(v)

        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g ** 2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    if dump_imm:
        return f_grad_shared, f_update, gshared, [t_prev, ms, vs]
    return f_grad_shared, f_update, gshared, None


def adadelta(lr, tparams, grads, inp, cost, **kwargs):
    g2 = kwargs.pop('g2', None)
    given_imm_data = kwargs.pop('given_imm_data', None)
    dump_imm = kwargs.pop('dump_imm', False)
    stochastic_updates = kwargs.pop('stochastic_updates', OrderedUpdates())
    extra_costs = kwargs.pop('extra_costs', None)
    if g2 is None:
        if extra_costs is None:
            outputs = cost
        else:
            outputs = [cost, extra_costs]
    else:
        if extra_costs is None:
            outputs = [cost, g2]
        else:
            outputs = [cost, g2, extra_costs]

    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]

    if given_imm_data is not None:
        running_up2 = [theano.shared(value, name='%s_rup2' % k)
                       for k, value in zip(tparams.iterkeys(), given_imm_data[0])]
        running_grads2 = [theano.shared(value, '%s_rgrad2' % k)
                          for k, value in zip(tparams.iterkeys(), given_imm_data[1])]
    else:
        running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2' % k)
                       for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2' % k)
                          for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]

    f_grad_shared = theano.function(inp, outputs, updates=zgup + stochastic_updates,
                                    profile=profile)

    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, zipped_grads)]

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + lr * ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=rg2up + ru2up + param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update, zipped_grads, [running_up2, running_grads2]


def rmsprop(lr, tparams, grads, inp, cost, **kwargs):
    g2 = kwargs.pop('g2', None)
    if g2 is None:
        outputs = cost
    else:
        outputs = [cost, g2]

    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, outputs, updates=zgup + rgup + rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update, zipped_grads, None


def sgd(lr, tparams, grads, inp, cost, **kwargs):
    g2 = kwargs.pop('g2', None)
    stochastic_updates = kwargs.pop('stochastic_updates', OrderedUpdates())
    if g2 is None:
        outputs = cost
    else:
        outputs = [cost, g2]

    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, outputs, updates=gsup + stochastic_updates,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update, gshared, None


Optimizers = {
    'adadelta': adadelta,
    'adam': adam,
    'rmsprop': rmsprop,
    'sgd': sgd,
}


__all__ = [
    'adadelta',
    'adam',
    'rmsprop',
    'sgd',
    'Optimizers',
]
