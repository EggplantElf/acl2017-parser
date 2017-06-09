from data import *
from parser import *
from util import *
import random
import numpy as np
from collections import deque, OrderedDict
from itertools import izip

class Oracle:
    def __init__(self, system):
        self.system = system

    def tell(self, state):
        for action in self.system.actions:
            if self.can_do(state, action):
                return action, self.get_label(state, action)
        return 'NO', None

    def get_label(self, state, action):
        if action in ['RA', 'RA2', 'RA3']:
            return state.sent.tokens[state.stk[-1]].label
        elif action == 'LA':
            return state.sent.tokens[state.stk[-2]].label
        elif action == 'LA2':
            return state.sent.tokens[state.stk[-3]].label
        elif action == 'LA3':
            return state.sent.tokens[state.stk[-4]].label
        else:
            return None


    def can_do(self, state, action):
        # it has to be at least valid
        if not state.valid(action):
            return False

        # constraints by the oracle
        if action == 'LA':
            return state.sent.gold_heads[state.stk[-2]] == state.stk[-1] \
                    and self.has_all_deps(state, state.stk[-2])
        elif action == 'RA':
            return state.sent.gold_heads[state.stk[-1]] == state.stk[-2] \
                    and self.has_all_deps(state, state.stk[-1])
        elif action == 'LA2':
            return state.sent.gold_heads[state.stk[-3]] == state.stk[-1] \
                    and self.has_all_deps(state, state.stk[-3])
        elif action == 'RA2':
            return state.sent.gold_heads[state.stk[-1]] == state.stk[-3] \
                    and self.has_all_deps(state, state.stk[-1])
        elif action == 'LA3':
            return state.sent.gold_heads[state.stk[-4]] == state.stk[-1] \
                    and self.has_all_deps(state, state.stk[-4])
        elif action == 'RA3':
            return state.sent.gold_heads[state.stk[-1]] == state.stk[-4] \
                    and self.has_all_deps(state, state.stk[-1])
        elif action == 'SH':
            return True
        else:
            return False

    def has_all_deps(self, state, head):
        return sum(1 for d in range(1, len(state.sent.tokens)) if state.sent.gold_heads[d] == head) \
            == sum(1 for d in range(1, len(state.sent.tokens)) if state.head(d) == head)


    def filter_trainable(self, sents):
        trainable_sents = []
        good, total = 0, 0
        for sent in sents:
            if total % 1000 == 0:
                print 'Reading sentences %d \r' % total,
                sys.stdout.flush()
            total += 1
            trainable = True
            state = State(sent)
            while not state.finished():
                action, label = self.tell(state)
                if action == 'NO':
                    trainable = False
                    break
                else:
                    idx = self.system.get_index(action, label)
                    state = state.perform(action, label, idx)
            if trainable:
                good += 1
                trainable_sents.append(sent)
        print
        print 'trainable sentences: %d / %d = %.2f%%' % (good, total, 100. * good / total)
        return trainable_sents


class ReplayBuffer(object):

    def __init__(self, buffer_size = 0):
        '''infinite buffer size if set to 0 (default)'''
        self.deque = deque([], buffer_size) if buffer_size else deque()


    def add(self, *experience):
        ''' experience is tuple of instance, e.g. (feature, label) or (s0, a0, r1, s1)'''
        self.deque.append(experience)

    def extend(self, experiences):
        self.deque.extend(experiences)

    def clear(self):
        self.deque.clear()

    def sample_batch(self, batch_size):
        '''     
        returns the zipped batches of experiences, 
        each element in the tuple is separated into a batch
        '''
        batch = random.sample(self.deque, min(batch_size, len(self.deque)))
        return tuple(np.array(b) for b in izip(*batch)) 



class Trainer(object):
    def __init__(self, manager):
        self.args = manager.args
        self.logger = Logger(self.args.log)
        self.replay = ReplayBuffer()
        self.manager = manager

    def add_instances(self, parser):
        good, total = 0, 0
        for i, sent in enumerate(self.manager.train_sents):
            total += 1
            if i % 100 == 0:
                print 'reading sentences... %d\r' % i,
                sys.stdout.flush()
            state = State(sent)

            experiences = []
            while not state.finished():
                feats = parser.extract(state)
                valid = parser.system.valid_mask(state)
                action, label = self.oracle.tell(state)
                if action == 'NO':
                    experiences = []
                    break
                y = parser.system.get_index(action, label)
                experiences.append((y, valid) + feats)
                state = state.perform(action, label, y)

            if experiences:
                self.replay.extend(experiences)

    def train(self, parser, num_steps = 20000):
        self.replay.clear()
        every = min(max(1000, num_steps / 10), 2000)

        self.oracle = Oracle(parser.system)
        self.add_instances(parser)

        print
        self.logger.log('# instances = %d\n' % len(self.replay.deque))

        best_uas, best_las, best_step = 0, 0, 0
        no_improve = 0
        run_acc = 0.0  # running average accuracy of the most recent 10 batches

        parser.model.log_params(self.logger)

        for step in xrange(1, num_steps):
            batch = self.replay.sample_batch(self.args.batch_size)
            correct, loss = parser.model.train_actor_supervised(step, *batch)

            run_acc += 0.1 * (1.0*correct/self.args.batch_size - run_acc)
            print 'step: %d acc: %.2f%% loss: %.6f \r' % (step, run_acc*100, loss),
            sys.stdout.flush()

            if step % every == 0:
                print
                uas, las = parser.parse(self.manager.dev_sents)
                if (self.args.unlabeled and uas > best_uas) or las > best_las: 
                    parser.model.save_params(self.args.model_to)
                    best_uas, best_las, best_step = uas, las, step
                    no_improve = 0
                else:
                    no_improve += 1
                if self.args.stop_after and no_improve >= self.args.stop_after:
                    self.logger.log('EARLY STOP FOR NO IMPROVEMENTS')
                    break

                parser.model.log_params(self.logger)

        self.logger.log('BEST MODEL: STEP = %d, UAS = %.2f%%, LAS = %.2f%%' % (best_step, best_uas, best_las))
        
        if self.args.test:
            print 'Testing...'
            parser.model.load_params(self.args.model_to)
            uas, las = parser.parse(self.manager.test_sents)
            self.logger.log('FINAL TEST: UAS = %.2f%%, LAS = %.2f%%' % (uas, las))

