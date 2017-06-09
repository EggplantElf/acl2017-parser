import lasagne
import lasagne.layers as L
from lasagne.init import *
from lasagne.nonlinearities import *
import theano
from theano import tensor as T
import numpy as np
import os
from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams

class Model(object):
    def __init__(self, args):
        self.srng = RandomStreams(seed=234)
        self.args = args
        self.actor = None
        self.layers = []

    def save_params(self, folder):
        print 'saving model to %s' % folder
        if self.actor:
            actor_file = os.path.join(folder, 'actor.npz')
            np.savez(actor_file, *L.get_all_param_values(self.actor_avg))

    def load_params(self, folder):
        print 'reading model from %s' % folder
        with np.load(os.path.join(folder, 'actor.npz')) as f:
            param_values = [f['arr_%d' %i] for i in range(len(f.files))]
            L.set_all_param_values(self.actor_avg, param_values)


    def log_params(self, logger = None):
        print
        msg = ''
        for p, v in zip(L.get_all_params(self.actor), L.get_all_param_values(self.actor)):
            msg += '%s: shape = %s, mean = %.4f, std = %.4f, norm = %.2f\n' % (p.name, v.shape, v.mean(), v.std(), (v**2).sum())
        if logger:
            logger.log(msg)
        else:
            print msg

    def get_char2word(self, ic, avg = False):
        suf = '_avg' if avg else ''
        ec = L.EmbeddingLayer(ic, self.args.vc, self.args.nc, name = 'ec'+suf, 
                                W=HeNormal() if not avg else Constant()) # (100, 24, 32, 16)
        ec.params[ec.W].remove('regularizable')

        if self.args.char_model == 'CNN':
            lds = L.dimshuffle(ec, (0, 3, 1, 2)) # (100, 16, 24, 32)
            ls = []
            for n in self.args.ngrams:
                lconv = L.Conv2DLayer(lds, self.args.nf, (1, n), untie_biases=True,
                                        W=HeNormal('relu') if not avg else Constant(),
                                        name = 'conv_%d'%n+suf) # (100, 64/4, 24, 32-n+1)
                lpool = L.MaxPool2DLayer(lconv, (1, self.args.max_len-n+1)) # (100, 64, 24, 1)
                lpool = L.flatten(lpool, outdim=3) # (100, 16, 24)
                lpool = L.dimshuffle(lpool, (0, 2, 1)) # (100, 24, 16)
                ls.append(lpool)
            xc = L.concat(ls, axis=2) # (100, 24, 64)
            return xc

        elif self.args.char_model == 'LSTM':
            ml = L.ExpressionLayer(ic, lambda x: T.neq(x, 0)) # mask layer (100, 24, 32)
            ml = L.reshape(ml, (-1, self.args.max_len)) # (2400, 32)

            gate_params = L.recurrent.Gate(W_in=Orthogonal(), W_hid=Orthogonal())
            cell_params = L.recurrent.Gate(W_in=Orthogonal(), W_hid=Orthogonal(),
                                            W_cell=None, nonlinearity=tanh)

            lstm_in = L.reshape(ec, (-1, self.args.max_len, self.args.nc)) # (2400, 32, 16)
            lstm_f = L.LSTMLayer(lstm_in, self.args.nw/2, mask_input=ml, grad_clipping=10., learn_init=True,
                                 peepholes=False, precompute_input=True,
                                 ingate=gate_params,forgetgate=gate_params,cell=cell_params,outgate=gate_params,
                                 # unroll_scan=True,
                                 only_return_final=True, name='forward'+suf) # (2400, 64)
            lstm_b = L.LSTMLayer(lstm_in, self.args.nw/2, mask_input=ml, grad_clipping=10., learn_init=True, 
                                 peepholes=False, precompute_input=True,
                                 ingate=gate_params,forgetgate=gate_params,cell=cell_params,outgate=gate_params,
                                 # unroll_scan=True,
                                 only_return_final=True, backwards=True, name='backward'+suf) # (2400, 64)
            remove_reg(lstm_f)
            remove_reg(lstm_b)
            if avg:
                set_zero(lstm_f)
                set_zero(lstm_b)
            xc = L.concat([lstm_f, lstm_b], axis=1) # (2400, 128)
            xc = L.reshape(xc, (-1, self.args.sw, self.args.nw)) # (100, 24, 256)
            return xc


    def get_actor(self, avg = False):
        suf = '_avg' if avg else ''
        iw = L.InputLayer(shape=(None, self.args.sw)) # (100, 24)
        ew = L.EmbeddingLayer(iw, self.args.vw, self.args.nw, name = 'ew'+suf, 
                                W=HeNormal() if not avg else Constant()) # (100, 24, 256)
        ew.params[ew.W].remove('regularizable') 
        if 'w' in self.args.freeze:
            ew.params[ew.W].remove('trainable')
        # for access from outside
        if not avg:
            self.Ew = ew.W 

        # char embedding with CNN/LSTM
        ic = L.InputLayer(shape=(None, self.args.sw, self.args.max_len)) # (100, 24, 32)
        ec = self.get_char2word(ic, avg) # (100, 24, 256)

        it = L.InputLayer(shape=(None, self.args.st))
        et = L.EmbeddingLayer(it, self.args.vt, self.args.nt, name = 'et'+suf, 
                                W=HeNormal() if not avg else Constant())
        et.params[et.W].remove('regularizable')

        il = L.InputLayer(shape=(None, self.args.sl))
        el = L.EmbeddingLayer(il, self.args.vl, self.args.nl, name = 'el'+suf, 
                                W=HeNormal() if not avg else Constant())
        el.params[el.W].remove('regularizable')

        to_concat = []
        if self.args.type == 'word':
            to_concat.append(ew)
        elif self.args.type == 'char':
            to_concat.append(ec)
        elif self.args.type == 'both':
            to_concat += [ew, ec]
        elif self.args.type == 'mix':
            to_concat.append(L.ElemwiseSumLayer([ew, ec]))

        if not self.args.untagged:
            to_concat.append(et)
        if not self.args.unlabeled:
            to_concat.append(el)

        x = L.concat(to_concat, axis=2) # (100, 24, 64+16+16)

        # additional: 
        # get the more compact representation of each token by its word, tag and label,
        # before putting into the hidden layer
        if self.args.squeeze:
            x = L.DenseLayer(x, num_units=self.args.squeeze, name='h0'+suf, num_leading_axes=2,
                            W=HeNormal('relu') if not avg else Constant()) # (100, 24, 64)

        h1 = L.DenseLayer(x, num_units=self.args.nh1, name = 'h1'+suf,
                            W=HeNormal('relu') if not avg else Constant()) # (100, 512)
        h1 = L.dropout(h1, self.args.p1)
        h2 = L.DenseLayer(h1, num_units=self.args.nh2, name = 'h2'+suf,
                            W=HeNormal('relu') if not avg else Constant()) # (100, 256)
        h2 = L.dropout(h2, self.args.p2)
        h3 = L.DenseLayer(h2, num_units=self.args.nh3, name = 'h3'+suf,
                            W=HeNormal() if not avg else Constant(),
                            nonlinearity=softmax) # (100, 125) num of actions

        return iw, ic, it, il, h3


    def build_graph(self):
        # theano variables
        iw_b = T.lmatrix('iw_b') 
        ic_b = T.ltensor3('ic_b')
        it_b = T.lmatrix('it_b')
        il_b = T.lmatrix('il_b')
        v_b = T.lmatrix('v_b')  # valid action mask
        y_b = T.lvector('y_b')  # index of the correct action from oracle


        steps = T.lscalar('steps') # num_of steps
        lr = self.args.learn_rate * self.args.decay ** T.cast(T.floor(steps / 2000.), 'float32')

        iw, ic, it, il, self.actor = self.get_actor(False)
        iw_avg, ic_avg, it_avg, il_avg, self.actor_avg = self.get_actor(True)

        actor_prob = L.get_output(self.actor_avg, {iw_avg:iw_b, ic_avg:ic_b, it_avg:it_b, il_avg:il_b}, deterministic=True)
        actor_rest = actor_prob * T.cast(v_b, theano.config.floatX) # mask the probabilities of invalid actions to 0
        actor_pred = T.argmax(actor_rest, 1)
        self.actor_predict = theano.function([v_b, iw_b, ic_b, it_b, il_b], actor_pred, on_unused_input='ignore')

        y_hat = L.get_output(self.actor, {iw:iw_b, ic:ic_b, it:it_b, il:il_b}, deterministic=False)
        xent = T.mean(lasagne.objectives.categorical_crossentropy(y_hat, y_b))
        reg = lasagne.regularization.regularize_network_params(L.get_all_layers(self.actor),
                                                             lasagne.regularization.l2)
        cost = xent + self.args.reg_rate * reg
        correct = T.eq(T.argmax(y_hat, 1), y_b).sum()

        params = L.get_all_params(self.actor) 
        avg_params = L.get_all_params(self.actor_avg)
        grads = T.grad(cost, params)
        if self.args.grad_norm:
            grads, norm = lasagne.updates.total_norm_constraint(grads, self.args.grad_norm, return_norm=True)

        updates = lasagne.updates.momentum(grads, params, lr, self.args.momentum)
        updates = apply_moving_average(params, avg_params, updates, steps, 0.9999)

        inputs = [steps, y_b, v_b, iw_b, ic_b, it_b, il_b]
        self.train_actor_supervised = theano.function(inputs, 
                                                      [correct, cost],
                                                      updates=updates, 
                                                      on_unused_input='ignore')



#######################################
# Helpers
def apply_moving_average(params, avg_params, updates, steps, decay):
    # assert params and avg_params are aligned
    weight = T.min([decay, steps / (steps + 1.)]).astype(theano.config.floatX)
    avg_updates = []
    for p, a in zip(params, avg_params):
        avg_updates.append((a, a - (1. - weight) * (a - p)))
    return updates.items() + avg_updates


def set_zero(layer):
    for p in layer.get_params():
        v = p.get_value()
        p.set_value(np.zeros_like(v))

def remove_reg(layer):
    for p in layer.params:
        if 'regularizable' in layer.params[p]:
            layer.params[p].remove('regularizable')

