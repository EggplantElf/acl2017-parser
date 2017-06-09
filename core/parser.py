import os, sys
from data import *
from model import *
from util import *
from collections import defaultdict, OrderedDict, Counter
from itertools import izip
import gzip, cPickle
import numpy as np


class Parser(object):
    def __init__(self, manager):
        self.args = manager.args
        self.maps = manager.maps
        self.system = manager.system

        if not self.args.log:
            self.args.log = self.args.model_to + '.log'
        self.logger = Logger(self.args.log)
        if not self.args.seed:
            self.args.seed = np.random.randint(10000)
        self.args.rng = np.random.RandomState(self.args.seed)

        self._add_model_args()
        self.model = Model(self.args)
        self.model.build_graph()
        if self.args.embw:
            self.load_emb()

        self.extractor = Extractor(self)
        self.extract = self.extractor.extract
        self.log_args()

    def _add_model_args(self):
        self.args.vc = len(self.maps.charmap)
        self.args.vw = len(self.maps.wordmap)
        self.args.vt = len(self.maps.tagmap)
        self.args.vl = len(self.maps.labelmap)
        self.args.sw = 24
        self.args.sc = 24
        self.args.st = 24
        self.args.sl = 24
        hidden_sizes = [int(s) for s in self.args.hidden_layer_sizes.split(',')]
        self.args.nh1 = hidden_sizes[0]
        self.args.nh2 = hidden_sizes[1]
        self.args.nh3 = self.system.num

        # for char
        self.args.ngrams = [int(n) for n in self.args.ngrams.split(',')] 
        self.args.nf = self.args.nw / len(self.args.ngrams)
        items = self.args.dropout.split(',')
        self.args.p1 = float(items[0])
        self.args.p2 = float(items[1])

    # change the embeddings from the value of the model directly
    def load_emb(self):
        print 'Loading word embedding...'
        Ew = self.model.Ew.get_value()    
        emb = {}
        for line in open(self.args.embw):
            items = line.split()
            word = items[0]
            if word in self.maps.wordmap:
                value = np.array([float(n) for n in items[1:]])
                Ew[self.maps.wordmap[word]] = self._normalize_vector(value)
        self.model.Ew.set_value(Ew)

    def _normalize_vector(self, v):
        if self.args.rescale:
            return ((v - v.mean()) / v.std()) * self.args.rescale
        elif self.args.l2norm:
            return v / (v ** 2).sum() * self.args.l2norm
        else:
            return v

    def log_args(self):
        msg = ''
        if self.args.msg:
            msg += self.args.msg + '\n'
        msg += 'args:\n'
        msg += '  -system: %s\n' % self.args.system
        msg += '  -train: %s\n' % self.args.train
        msg += '  -dev: %s\n' % self.args.dev
        msg += '  -test: %s\n' % self.args.test
        msg += '  -model_from: %s\n' % self.args.model_from
        msg += '  -model_to: %s\n' % self.args.model_to
        msg += '  -embw: %s\n' % self.args.embw
        msg += '  -log: %s\n' % self.args.log
        msg += '  -num_steps: %d\n' % self.args.num_steps
        msg += '  -first: %d\n' % self.args.first
        msg += '  -min_word_freq: %d\n' % self.args.min_word_freq
        msg += '  -min_char_freq: %d\n' % self.args.min_char_freq
        msg += '  -min_tag_freq: %d\n' % self.args.min_tag_freq
        msg += '  -min_label_freq: %d\n' % self.args.min_label_freq
        msg += '  -batch_size: %d\n' % self.args.batch_size
        msg += '  -reg_rate: %s\n' % self.args.reg_rate
        msg += '  -learn_rate: %s\n' % self.args.learn_rate
        msg += '  -decay: %.2f\n' % self.args.decay
        msg += '  -momentum: %.2f\n' % self.args.momentum
        msg += '  -seed: %s\n' % self.args.seed
        msg += '  -emb_sizes: w=%d, c=%d, t=%d, l=%d\n' % (self.args.nw, self.args.nc, self.args.nt, self.args.nl)
        msg += '  -hidden_layer_sizes: %s\n' % self.args.hidden_layer_sizes
        msg += '  -dropout: %s\n' % self.args.dropout
        msg += '  -type: %s\n' % self.args.type
        if self.args.type != 'word':
            msg += '  -char_model: %s\n' % self.args.char_model
            msg += '  -ngrams: %s\n' % self.args.ngrams
        msg += '  -untagged: %s\n' % self.args.untagged
        msg += '  -squeeze: %d\n' % self.args.squeeze
        msg += '  -grad_norm: %d\n' % self.args.grad_norm
        msg += '  -l2norm: %s\n' % self.args.l2norm
        msg += '  -rescale: %s\n' % self.args.rescale
        msg += '  -freeze: %s\n' % self.args.freeze

        self.logger.log(msg)

    ####################################################
    # batched parsing

    def parse(self, sents, batch_size = 1000):

        all_states = OrderedDict((sent, State(sent)) for sent in sents)
        batch_stream = BatchStream(all_states.values(), batch_size)


        for state_b in batch_stream:
            new_state_b = self.parse_one_step_in_batch(state_b)
            for new_state in new_state_b:
                all_states[new_state.sent] = new_state
                if not new_state.finished():
                    batch_stream.add(new_state)

        evaluator = Evaluator()
        for state in all_states.values():
            evaluator.evaluate(state)

        if self.args.out:
            with open(self.args.out, 'w') as f:
                for state in all_states.values():
                    f.write(state.to_str())

        self.logger.log(evaluator.result('TEST'))
        return evaluator.uas(), evaluator.las()

    def parse_one_step_in_batch(self, state_b, stochastic=False):
        new_state_b = []

        feats = zip(*(self.extract(state) for state in state_b))
        valid_b = np.array([self.system.valid_mask(state) for state in state_b], dtype='int64')
        index_b = self.model.actor_predict(valid_b, *feats) 

        for state, index in izip(state_b, index_b):
            action, label = self.system.get_action_label(index)
            new_state = state.perform(action, label, index)
            new_state_b.append(new_state)
        return new_state_b


class BatchStream(object):
    def __init__(self, init_data, batch_size = 5):
        self.queue = deque(init_data)
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        while self.queue:
            batch.append(self.queue.popleft())
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            elif not self.queue and batch:
                yield batch
                batch = []

    def add(self, data):
        self.queue.appendleft(data)


class Maps:
    def __init__(self):
        self.charmap = {'<PAD>': 0, '<UNK>': 1, '<MUL>':2, '<SOW>':3, '<EOW>':4}
        self.wordmap = {'<PAD>': 0, '<UNK>': 1}
        self.tagmap = {'<PAD>': 0, '<UNK>': 1}
        self.labelmap = {'<PAD>': 0, '<UNK>': 1}
        self.name_map = {'word': self.wordmap, 'char': self.charmap, 'tag':self.tagmap, 'label': self.labelmap}


    def save(self, filename):
        stream = gzip.open(filename,'wb')
        cPickle.dump(self.charmap,stream,-1)
        cPickle.dump(self.wordmap,stream,-1)
        cPickle.dump(self.tagmap,stream,-1)
        cPickle.dump(self.labelmap,stream,-1)
        stream.close()

    def load(self, filename):
        stream = gzip.open(filename,'rb')
        self.charmap = cPickle.load(stream)
        self.wordmap = cPickle.load(stream)
        self.tagmap = cPickle.load(stream)
        self.labelmap = cPickle.load(stream)
        self.name_map = {'word': self.wordmap, 'char': self.charmap, 'tag':self.tagmap, 'label': self.labelmap}
        stream.close()


    def add(self, name, item):
        if item not in self.name_map[name]:
            self.name_map[name][item] = len(self.name_map[name])

    def get(self, name, item):
        return self.name_map[name].get(item, 1)


class DataManager(object):
    def __init__(self, args):
        self.args = args
        if self.args.model_from:
            self.load()
        else:
            self.init()

    def load(self):
        self.maps = Maps()
        self.maps.load(os.path.join(self.args.model_from, 'maps.gz'))
        self.system = TransitionSystem(self.args.system)
        self.system.register_labels(self.maps.labelmap)
        self.test_sents = list(self.read_sentences(self.args.test, self.args.conll06))\
                         if self.args.test else []
        self.lookup_idx(self.test_sents)
                         
    def init(self):
        if not os.access(self.args.model_to, os.R_OK):
            os.makedirs(self.args.model_to)
        # data preprocessing: read sentences, get maps, lookup indices
        self.train_sents = list(self.read_sentences(self.args.train, self.args.conll06, self.args.first))\
                         if self.args.train else []
        self.dev_sents = list(self.read_sentences(self.args.dev, self.args.conll06))\
                         if self.args.dev else []
        self.test_sents = list(self.read_sentences(self.args.test, self.args.conll06))\
                         if self.args.test else []
        self.create_maps()
        self.lookup_idx(self.train_sents)
        self.lookup_idx(self.dev_sents)
        self.lookup_idx(self.test_sents)
        self.system = TransitionSystem(self.args.system)
        self.system.register_labels(self.maps.labelmap)
        self.maps.save(os.path.join(self.args.model_to, 'maps.gz'))


    def read_sentences(self, filename, conll06 = True, first = None):
        i = 0
        sentence = Sentence()
        with open(filename) as f:
            for line in f:
                line = line.rstrip()
                if line and not line.startswith('#'):
                    if '-' not in line.split()[0]:
                        sentence.add_token(Token(line, conll06))
                elif len(sentence.tokens) > 1:
                    yield sentence.complete()
                    sentence = Sentence()
                    i += 1
                    if first and i >= first:
                        break


    def create_maps(self):
        self.maps = Maps()

        words = Counter()
        chars = Counter()
        tags = Counter()
        labels = Counter()

        for sent in self.train_sents:
            for token in sent.tokens:
                word = token.word
                words[word] += 1
                tags[token.ptag] += 1                
                labels[token.label] += 1
                for char in token.word.decode('utf8'):
                    chars[char] += 1

        # extend the vocab for pre-trained embeddings only
        self.all_vocab = set(words.keys())
        for sent in self.dev_sents + self.test_sents:
            for token in sent.tokens:
                word = token.word
                if word not in self.all_vocab:
                    self.all_vocab.add(word)

        for word in sorted(words):
            if words[word] > self.args.min_word_freq:
                self.maps.add('word', word)

        for char in sorted(chars):
            if chars[char] > self.args.min_char_freq:
                self.maps.add('char', char)

        if not self.args.untagged:
            for tag in sorted(tags):
                if tags[tag] > self.args.min_tag_freq:
                    self.maps.add('tag', tag)

        if not self.args.unlabeled:
            for label in sorted(labels):
                if labels[label] > self.args.min_label_freq:
                    self.maps.add('label', label)


    # useful redundancy, look up both word and char
    def lookup_idx(self, sents):
        for sent in sents:
            for token in sent.tokens:
                token.idxw = self.maps.get('word', token.word)
                token.idxc = self.get_idxc(token.word.decode('utf-8'))
                token.idxt = self.maps.get('tag', token.ptag)


    # get vector of character indices of a word, not index of a word
    def get_idxc(self, word):
        # exclude start and end symbol
        max_len = self.args.max_len - 2
        if len(word) <= max_len:
            # PAD on both sides
            return np.array([0]*((max_len-len(word)) / 2) \
                          + [3] \
                          + [self.maps.get('char', c) for c in word] \
                          + [4]\
                          + [0]*((max_len-len(word)+1) / 2))
        # for longer words, take the first and last N/2 chars and collapse the rest to 2 in the middle
        # the intuition is that prefix and suffix are more important than the middle part
        else:
            return np.array([3] \
                          + [self.maps.get('char', c) for c in word[:(max_len-1)/2]] \
                          + [2] \
                          + [self.maps.get('char', c) for c in word[-(max_len/2):]]\
                          + [4])


