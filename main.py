#! /usr/bin/python

from core.util import *
from core.trainer import *
from core.parser import *
import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Greedy Parser')
    args.add_argument('-msg',default='', help='simple message of the experiment')
    args.add_argument('-type',default='word',choices=['word','char','both', 'mix'],help='type of parser model, word or char')
    args.add_argument('-char_model',default='CNN',choices=['CNN','LSTM'],help='char model, CNN or LSTM ')

    args.add_argument('-unlabeled', action='store_true', help='unlabeled parsing')
    args.add_argument('-untagged', action='store_true', help='untagged parsing')
    args.add_argument('-squeeze', default=256,type=int, help='squeeze the representation of each word (word, char, tag, label...), 0 for not using')

    # common arguments for both word and char 
    args.add_argument('-train',default=None, help='training file')
    args.add_argument('-dev',default=None, help='dev file')
    args.add_argument('-test',default=None, help='test file')

    args.add_argument('-out',default=None, help='output prediction file')
    args.add_argument('-log',default=None, help='log file')
    args.add_argument('-num_steps',default=100000,type=int,help='number of training steps (batches)')

    args.add_argument('-model_to',default='tmp', help='folder to save model')
    args.add_argument('-model_from',default=None, help='continue training from model')
    args.add_argument('-system',default='Attardi', help='transition system')
    args.add_argument('-batch_size', default=100, type=int, help='batch size')
    args.add_argument('-learn_rate', default=0.1, type=float, help='learning rate')
    args.add_argument('-reg_rate', default=1e-4, type=float, help='regularization rate')
    args.add_argument('-decay', default=0.95, type=float, help='decay of learning rate per 2000 steps')
    args.add_argument('-momentum', default=0.9, type=float, help='momentum of learning rate')
    args.add_argument('-dropout', default='0.1,0.1', help='comma separated dropout rate for each hidden layer, 0 for no dropout')
    args.add_argument('-first', default=999999, type=int, help='read only the first N sentences, for easier debug')
    args.add_argument('-stop_after', default=3, type=int, help='stop training if no improvement in N epochs on dev set, 0 for not early stop')
    args.add_argument('-seed', default=None, type=int, help='random state seed')
    args.add_argument('-conll06', action='store_true', help='conll06 data')
    args.add_argument('-hidden_layer_sizes', default='512,256', help='comma separated hidden layer sizes')
    args.add_argument('-nw', default=256, type=int, help='dimension of word embeddings (must be divisible by the number of n-grams)')
    args.add_argument('-nt', default=32, type=int, help='dimension of tag embeddings')
    args.add_argument('-nl', default=32, type=int, help='dimension of label embeddings')

    args.add_argument('-min_word_freq', default=1, type=int, help='cut-off threshold of word occurrences in the training data')
    args.add_argument('-min_char_freq', default=5, type=int, help='cut-off threshold of char occurrences in the training data')
    args.add_argument('-min_tag_freq', default=0, type=int, help='cut-off threshold of tag occurrences in the training data')
    args.add_argument('-min_label_freq', default=0, type=int, help='cut-off threshold of label occurrences in the training data')
    args.add_argument('-freeze', default='', help='freeze the embedding, can be cominations of [cwtly]')
    args.add_argument('-grad_norm', default=10, type=int, help='gradient clipping')

    # arguments for word only
    args.add_argument('-embw',default=None, help='pretrained word embedding file')
    args.add_argument('-save_embw', action='store_true', help='save word embedding to txt file')
    args.add_argument('-avg_oov', action='store_true', help='set the vector of oov words as average of rare words')
    args.add_argument('-rescale', default=None, type=float, help='rescale the std of pretrained embedding')
    args.add_argument('-l2norm', default=None, type=float, help='l2norm the std of pretrained embedding')

    # arguments for char only
    args.add_argument('-max_len', default=32, type=int, help='maximum length of a word in character embedding')
    args.add_argument('-ngrams', default='3,5,7,9', help='comma separated number for N-grams for CNN character filter')
    args.add_argument('-nc', default=32, type=int, help='dimension of character embeddings')

    args = args.parse_args()


    manager = DataManager(args)
    parser = Parser(manager)


    if args.model_from:
        print 'Test Only'
        parser.model.load_params(args.model_from)
        uas, las = parser.parse(manager.test_sents)
        parser.logger.log('FINAL TEST: UAS = %.2f%%, LAS = %.2f%%' % (uas, las))
    else:
        trainer = Trainer(manager)
        trainer.train(parser, args.num_steps)
