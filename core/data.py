import numpy as np
import random
from collections import deque
import gzip, cPickle

class Token:
    __slots__ = ['tid', 'word', 'lemma', 'tag', 'mor', 'head', 'label', \
                 'glemma', 'gtag', 'gmor', 'phead', 'plabel', 'idxw', 'idxc', 'idxt', 'lang']

    def __init__(self, line, conll06):
        self.conll06 = conll06
        entries = line.split()
        self.tid = int(entries[0].split('_')[-1])
        self.word = entries[1]
        if conll06:
            self.lemma = entries[2] # gold
            self.plemma = entries[2] # gold
            self.tag = entries[3] # gold
            if entries[4] == '_':
                self.ptag = entries[3] # pred
            else:
                self.ptag = entries[4] # pred
            self.mor = entries[5] # gold
            self.pmor = entries[5] # gold
            self.head = int(entries[6]) # gold
            self.label = entries[7] # gold
            self.phead = '_' if entries[8] == '_' else int(entries[8])
            self.plabel = entries[9]
            self.lang = '_' 
        else:
            self.lemma = entries[2] # gold
            self.plemma = entries[3] # pred
            self.tag = entries[4] # gold
            self.ptag = entries[5] # pred
            self.mor = entries[6] # gold
            self.pmor = entries[7] # pred
            self.head = int(entries[8]) # gold
            self.label = entries[10] # gold
            self.phead = '_'
            self.plabel = '_'
            self.lang = '_'

    def to_str(self, blind=True):
        if blind:
            if self.conll06:
                return '%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s' %\
                        (self.tid, self.word, self.lemma, self.tag, self.ptag, self.mor, self.phead, self.plabel, '_', '_')
            else:
                return '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t_\t_' %\
                        (self.tid, self.word, self.lemma, self.plemma, self.tag, self.ptag, \
                            self.mor, self.pmor, self.phead, '_', self.plabel, '_')
        else:
            if self.conll06:
                return '%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s' %\
                        (self.tid, self.word, self.lemma, self.tag, self.ptag, self.mor, self.head, self.label, self.phead, self.plabel)
            else:
                return '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t_\t_' %\
                        (self.tid, self.word, self.lemma, self.plemma, self.tag, self.ptag, \
                            self.mor, self.pmor, self.head, self.phead, self.label, self.plabel)


    def to_str_converted(self, pred = False):
        if self.conll06:
            return '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t_\t%s' %\
                    (self.tid, self.word, self.lemma, self.lemma, self.tag, self.tag, \
                        self.mor, self.mor, self.head, self.phead, self.label, self.plabel, self.lang)
        elif pred:
            return '%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s' %\
                    (self.tid, self.word, self.plemma, self.ptag, self.ptag, self.pmor, self.head, self.label, self.phead, self.plabel)
        else:
            return '%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s' %\
                    (self.tid, self.word, self.lemma, self.tag, self.tag, self.mor, self.head, self.label, self.phead, self.plabel)

class Root(Token):
    def __init__(self):
        self.tid = 0
        self.word = '<ROOT>'
        self.lemma = '<ROOT>'
        self.plemma = '<ROOT>'
        self.tag = '<ROOT>'
        self.ptag = '<ROOT>'
        self.mor = '<ROOT>'  
        self.pmor = '<ROOT>'
        self.label = '<ROOT>'
        self.lang = '_'


class Sentence(object):
    def __init__(self):
        self.tokens = [Root()]

    def add_token(self, token):
        self.tokens.append(token)

    def complete(self):
        self.gold_heads = {d.tid: d.head for d in self.tokens[1:]}
        self.lang = self.tokens[-1].lang
        return self

    def to_str(self):
        return '\n'.join(t.to_str() for t in self.tokens[1:]) + '\n\n'

    def to_str_converted(self, pred):
        return '\n'.join(t.to_str_converted(pred) for t in self.tokens[1:]) + '\n\n'



# TODO: use idx of label instead of string to reduce lookup
class State(object):
    __slots__ = ['sent', 'prev_state', 'prev_action', 'prev_label', 'prev_idx', 'arcs', 'attached',\
                 'flag', 'stk', 'bfr', 'iw', 'ic', 'it', 'il', 'ft']

    def __init__(self, sent, prev_state = None, prev_action = None, prev_label = None, prev_idx = None,
                arcs = (), attached = set(), stk = None, bfr = None, flag = False):
        self.sent = sent
        self.prev_state = prev_state
        self.prev_action = prev_action
        self.prev_label = prev_label
        self.prev_idx = prev_idx
        self.arcs = arcs
        self.attached = attached
        self.flag = flag
        if stk == None:
            self.stk = (0, )
            self.bfr = tuple(range(1, len(sent.tokens)))
        else:
            self.stk = stk
            self.bfr = bfr

    def show(self):
        return self.stk, self.bfr

    # caution, really changes stuff in sentence
    def to_str(self):
        for d, h, l in sorted(self.arcs):
            self.sent.tokens[d].phead = h
            self.sent.tokens[d].plabel = l
        return self.sent.to_str()

    def attach(self, iw, ic, it, il):
        self.iw = iw
        self.ic = ic
        self.it = it
        self.il = il

    ##################################
    # helper functions to unify the behaviours of different state classes
    # could be merged into one function for efficiency
    def left_children(self, idx, num = 2):
        lefts = sorted([d for d, h, l in self.arcs if h == idx and d < h])
        if len(lefts) == 0:
            return (None, None)
        elif len(lefts) == 1:
            return (lefts[0], None)
        else:
            return (lefts[0], lefts[1])
        # return tuple(lefts + [None, None])[:2]

    def right_children(self, idx, num = 2):
        rights = sorted([d for d, h, l in self.arcs if h == idx and d > h], reverse = True)
        if len(rights) == 0:
            return (None, None)
        elif len(rights) == 1:
            return (rights[0], None)
        else:
            return (rights[0], rights[1])
        # return tuple(rights + [None, None])[:2]

    def label(self, idx):
        for d, h, l in self.arcs:
            if d == idx:
                return l
        return None

    def head(self, idx):
        for d, h, l in self.arcs:
            if d == idx:
                return h
        return None

    def head_label(self, idx):
        for d, h, l in self.arcs:
            if d == idx:
                return h, l
        return None, None

    ##################################


    def finished(self):
        return len(self.stk) == 1 and len(self.bfr) == 0

    def valid(self, action, label = None):
        if action == 'SH':
            return len(self.bfr) > 0
        elif action == 'LA':
            return len(self.stk) > 2
        elif action == 'RA':
            return len(self.stk) > 1 and (self.stk[-2] != 0 or len(self.bfr) == 0) # extra
        elif action == 'LA2':
            return len(self.stk) > 3
        elif action == 'RA2':
            return len(self.stk) > 2 and (self.stk[-3] != 0 or len(self.bfr) == 0) # extra
        elif action == 'LA3':
            return len(self.stk) > 4
        elif action == 'RA3':
            return len(self.stk) > 3 and (self.stk[-4] != 0 or len(self.bfr) == 0) # extra
        else:
            return False


    def perform(self, action, label, idx):
        if action == 'SH':
            stk = self.stk + self.bfr[:1]
            bfr = self.bfr[1:]
            arcs = self.arcs
            attached = self.attached
            flag = self.flag
        elif action == 'LA':
            stk = self.stk[:-2] + self.stk[-1:]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-2], self.stk[-1], label),)
            attached = self.attached.union({self.stk[-2]})
            flag = True
        elif action == 'RA':
            stk = self.stk[:-1]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-1], self.stk[-2], label),)
            attached = self.attached.union({self.stk[-1]})
            flag = True
        elif action == 'LA2':
            stk = self.stk[:-3] + self.stk[-2:]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-3], self.stk[-1], label),)
            attached = self.attached.union({self.stk[-3]})
            flag = True
        elif action == 'RA2':
            stk = self.stk[:-1]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-1], self.stk[-3], label),)
            attached = self.attached.union({self.stk[-1]})
            flag = True
        elif action == 'LA3':
            stk = self.stk[:-4] + self.stk[-3:]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-4], self.stk[-1], label),)
            attached = self.attached.union({self.stk[-4]})
            flag = True
        elif action == 'RA3':
            stk = self.stk[:-1]
            bfr = self.bfr
            arcs = self.arcs + ((self.stk[-1], self.stk[-4], label),)
            attached = self.attached.union({self.stk[-1]})
            flag = True
        else:
            raise Exception('No such move')
            
        return State(self.sent, self, action, label, idx, arcs, attached, stk, bfr, flag)



class Extractor:
    def __init__(self, parser):
        self.maps = parser.maps
        self.max_len = parser.args.max_len

    # s0, s1, s2, s3, b0, b1, b2, b3, s0L0, s0L1, s0R0, s0R1, s1L0, s1L1, s1R0, s1R1, s0L0L0, s0R0R0, s1L0L0, s1R0R0
    # s0L0, s0L1, s0R0, s0R1, s1L0, s1L1, s1R0, s1R1, s0L0L0, s0R0R0, s1L0L0, s1R0R0
    # additional for attardi (and all): s2L0(20), s2L1(21), s2R0(22), s2R1(23)
    def extract(self, state):
        if not hasattr(state, 'iw'):
            idxc = np.zeros((24, self.max_len), dtype='int64')
            idxw = [0] * 24
            idxt = [0] * 24
            idxl = [0] * 24
            stk = state.stk
            bfr = state.bfr
            sent = state.sent.tokens


            if len(bfr) > 0:
                b0 = bfr[0]
                idxw[4] = sent[b0].idxw
                idxc[4] = sent[b0].idxc
                idxt[4] = sent[b0].idxt
                if len(bfr) > 1:
                    b1 = bfr[1]
                    idxw[5] = sent[b1].idxw
                    idxc[5] = sent[b1].idxc
                    idxt[5] = sent[b1].idxt
                    if len(bfr) > 2:
                        b2 = bfr[2]
                        idxw[6] = sent[b2].idxw
                        idxc[6] = sent[b2].idxc
                        idxt[6] = sent[b2].idxt
                        if len(bfr) > 3:
                            b3 = bfr[3]
                            idxw[7] = sent[b3].idxw
                            idxc[7] = sent[b3].idxc
                            idxt[7] = sent[b3].idxt


            if len(stk) > 0:
                s0 = stk[-1]
                idxw[0] = sent[s0].idxw
                idxc[0] = sent[s0].idxc
                idxt[0] = sent[s0].idxt
                s0L0, s0L1 = state.left_children(s0) 
                s0R0, s0R1 = state.right_children(s0) 
                if s0L0 is not None:
                    idxw[8] = sent[s0L0].idxw
                    idxc[8] = sent[s0L0].idxc
                    idxt[8] = sent[s0L0].idxt
                    idxl[8] = self.maps.get('label', state.label(s0L0))
                    s0L0L0, _ = state.left_children(s0L0)
                    if s0L0L0 is not None:
                        idxw[16] = sent[s0L0L0].idxw
                        idxc[16] = sent[s0L0L0].idxc
                        idxt[16] = sent[s0L0L0].idxt
                        idxl[16] = self.maps.get('label', state.label(s0L0L0))
                if s0L1 is not None:
                    idxw[9] = sent[s0L1].idxw
                    idxc[9] = sent[s0L1].idxc
                    idxt[9] = sent[s0L1].idxt
                    idxl[9] = self.maps.get('label', state.label(s0L1))

                if s0R0 is not None:
                    idxw[10] = sent[s0R0].idxw
                    idxc[10] = sent[s0R0].idxc
                    idxt[10] = sent[s0R0].idxt
                    idxl[10] = self.maps.get('label', state.label(s0R0))
                    s0R0R0, _ = state.right_children(s0R0)
                    if s0R0R0 is not None:
                        idxw[17] = sent[s0R0R0].idxw
                        idxc[17] = sent[s0R0R0].idxc
                        idxt[17] = sent[s0R0R0].idxt
                        idxl[17] = self.maps.get('label', state.label(s0R0R0))
                if s0R1 is not None:
                    idxw[11] = sent[s0R1].idxw
                    idxc[11] = sent[s0R1].idxc
                    idxt[11] = sent[s0R1].idxt
                    idxl[11] = self.maps.get('label', state.label(s0R1))

                if len(stk) > 1:
                    s1 = stk[-2]
                    idxw[1] = sent[s1].idxw
                    idxc[1] = sent[s1].idxc
                    idxt[1] = sent[s1].idxt
                    s1L0, s1L1 = state.left_children(s1) 
                    s1R0, s1R1 = state.right_children(s1) 
                    if s1L0 is not None:
                        idxw[12] = sent[s1L0].idxw
                        idxc[12] = sent[s1L0].idxc
                        idxt[12] = sent[s1L0].idxt
                        idxl[12] = self.maps.get('label', state.label(s1L0))
                        s1L0L0, _ = state.left_children(s1L0)
                        if s1L0L0 is not None:
                            idxw[18] = sent[s1L0L0].idxw
                            idxc[18] = sent[s1L0L0].idxc
                            idxt[18] = sent[s1L0L0].idxt
                            idxl[18] = self.maps.get('label', state.label(s1L0L0))
                    if s1L1 is not None:
                        idxw[13] = sent[s1L1].idxw
                        idxc[13] = sent[s1L1].idxc
                        idxt[13] = sent[s1L1].idxt
                        idxl[13] = self.maps.get('label', state.label(s1L1))
                    if s1R0 is not None:
                        idxw[14] = sent[s1R0].idxw
                        idxc[14] = sent[s1R0].idxc
                        idxt[14] = sent[s1R0].idxt
                        idxl[14] = self.maps.get('label', state.label(s1R0))
                        s1R0R0, _ = state.right_children(s1R0)
                        if s1R0R0 is not None:
                            idxw[19] = sent[s1R0R0].idxw
                            idxc[19] = sent[s1R0R0].idxc
                            idxt[19] = sent[s1R0R0].idxt
                            idxl[19] = self.maps.get('label', state.label(s1R0R0))
                    if s1R1 is not None:
                        idxw[15] = sent[s1R1].idxw
                        idxc[15] = sent[s1R1].idxc
                        idxt[15] = sent[s1R1].idxt
                        idxl[15] = self.maps.get('label', state.label(s1R1))

                    if len(stk) > 2:
                        s2 = stk[-3]
                        idxw[2] = sent[s2].idxw
                        idxc[2] = sent[s2].idxc
                        idxt[2] = sent[s2].idxt
                        s2L0, s2L1 = state.left_children(s2) 
                        s2R0, s2R1 = state.right_children(s2) 

                        if s2L0 is not None:
                            idxw[20] = sent[s2L0].idxw
                            idxc[20] = sent[s2L0].idxc
                            idxt[20] = sent[s2L0].idxt
                            idxl[20] = self.maps.get('label', state.label(s2L0))
                        if s2L1 is not None:
                            idxw[21] = sent[s2L1].idxw
                            idxc[21] = sent[s2L1].idxc
                            idxt[21] = sent[s2L1].idxt
                            idxl[21] = self.maps.get('label', state.label(s2L1))
                        if s2R0 is not None:
                            idxw[22] = sent[s2R0].idxw
                            idxc[22] = sent[s2R0].idxc
                            idxt[22] = sent[s2R0].idxt
                            idxl[22] = self.maps.get('label', state.label(s2R0))
                        if s2R1 is not None:
                            idxw[23] = sent[s2R1].idxw
                            idxc[23] = sent[s2R1].idxc
                            idxt[23] = sent[s2R1].idxt
                            idxl[23] = self.maps.get('label', state.label(s2R1))

                        if len(stk) > 3:
                            s3 = stk[-4]
                            idxw[3] = sent[s3].idxw
                            idxc[3] = sent[s3].idxc
                            idxt[3] = sent[s3].idxt

            state.attach(idxw, idxc, idxt, idxl)
        return state.iw, state.ic, state.it, state.il


class TransitionSystem:
    def __init__(self, name):
        self.name = name

        if name == 'ArcStandard':
            self.actions = ['LA', 'RA', 'SH'] # always the order prefered by the oracle
            self.trans2idx = {('SH', None): 0}
            self.idx2trans = {0: ('SH', None)}
            self.idx_group = {'SH':[0], 'LA':[], 'RA':[]}
        elif name == 'Attardi':
            self.actions = ['LA', 'LA2', 'RA', 'RA2', 'SH']
            self.trans2idx = {('SH', None): 0}
            self.idx2trans = {0: ('SH', None)}
            self.idx_group = {'SH':[0], 'LA':[], 'RA':[], 'LA2':[], 'RA2':[]}
        elif name == 'Attardi2':
            self.actions = ['LA', 'LA2', 'LA3', 'RA', 'RA2', 'RA3', 'SH']
            self.trans2idx = {('SH', None): 0}
            self.idx2trans = {0: ('SH', None)}
            self.idx_group = {'SH':[0], 'LA':[], 'RA':[], 'LA2':[], 'RA2':[], 'LA3':[], 'RA3':[]}
        elif name == 'Swap':
            self.actions = ['LA', 'RA', 'SW', 'SH'] # check the order
            self.trans2idx = {('SH', None): 0, ('SW', None): 1}
            self.idx2trans = {0: ('SH', None), 1: ('SW', None)}
            self.idx_group = {'SH':[0], 'LA':[], 'RA':[], 'SW':[1]}
        else:
            raise Exception('No such system')



    # both register labels from scratch and register additional labels from another treebank
    def register_labels(self, labels):
        for action in self.actions:
            if action not in ['SH', 'SW']:
                for label in sorted(labels):
                    if label != '<PAD>' and (action, label) not in self.trans2idx:
                        idx = len(self.trans2idx)
                        self.trans2idx[(action, label)] = idx
                        self.idx2trans[idx] = (action, label)
                        self.idx_group[action].append(idx)
        self.num = len(self.trans2idx)

    # just for compatible
    def get_action(self, idx):
        return self.idx2trans[idx][0]

    def get_action_label(self, idx):
        return self.idx2trans[idx]

    def get_index(self, action, label):
        if (action, label) in self.trans2idx:
            return self.trans2idx[(action, label)]
        else:
            return self.trans2idx[(action, '<UNK>')]

    def valid_idx(self, state):
        '''return the group of indices that are valid given the state'''
        return sum([self.idx_group[a] for a in self.actions if state.valid(a)], [])

    def valid_mask(self, state):
        '''return the mask of valid actions, with 1 for valid and 0 for invalid'''
        mask = np.zeros(self.num, dtype='int64')
        mask[self.valid_idx(state)] = 1
        return mask

    def invalid_actions(self, state):
        return [a for a in self.actions if not state.valid(a)]

    def invalid_idx(self, state):
        return sum([self.idx_group[action] for action in self.invalid_actions(state)], [])
