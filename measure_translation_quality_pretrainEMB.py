from __future__ import division
from collections import Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import sys
import logging
import numpy as np

use_cuda = torch.cuda.is_available()

if use_cuda:
    print "use cuda"
else:
    print "use CPU"
#logger = logging.getLogger(__name__)

class EncoderCNNRNN(nn.Module):
	def __init__(self, vocab_size, emb_size, feature_size, window_size, hidden_size, dropout, n_layers=1, pretrained_embs = None):
		super(EncoderCNNRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.dropout = nn.Dropout(dropout)
		self.embedding = nn.Embedding(vocab_size, emb_size)
		if pretrained_embs is not None:
			self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embs))
		self.conv = nn.ModuleList([ nn.Conv1d(emb_size, feature_size, 2*sz+1, padding = sz) for sz in window_size])
		self.gru = nn.GRU(feature_size*len(window_size), hidden_size, n_layers, batch_first = True)

	def forward(self, input, batch_size):
		embedded = self.embedding(input).permute(0, 2, 1) # batch_size x seq_len x emb_size =>  batch_size x emb_size x seq_len
		feature = [ self.dropout( F.relu(conv(embedded))) for conv in self.conv] # batch_size x emb_size x seq_len => batch_size x feature_size x seq_len

		output = torch.cat(feature, 1).permute(0, 2, 1) # batch_size x feature_size x seq_len => batch_size x seq_len x feature_size
		hidden = self.initHidden(batch_size)
		output, hidden = self.gru(output, hidden)

		return output, torch.mean(output, 1) # batch_size x seq_len x hidden_size,  batch_size x hidden_size

	def initHidden(self, batch_size):	
		result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result

class AttentionRegression(nn.Module):
	def __init__(self, vocab, emb_size, feature_size, window_size, dropout, hidden_size, n_layers, attention_size):
		super(AttentionRegression, self).__init__()
		self.encoder_s = EncoderCNNRNN(vocab.source_vocab_size, emb_size, feature_size, window_size, hidden_size, dropout, n_layers, vocab.get_pretrained_src())
		self.encoder_t = EncoderCNNRNN(vocab.target_vocab_size, emb_size, feature_size, window_size, hidden_size, dropout, n_layers, vocab.get_pretrained_tgt())
		self.s2att_s = nn.Linear(hidden_size, attention_size)
		self.t2att_s = nn.Linear(hidden_size, attention_size ,bias = False)
		self.attw_s = nn.Linear(attention_size, 1)
		
		self.t2att_t = nn.Linear(hidden_size, attention_size)
		self.s2att_t = nn.Linear(hidden_size, attention_size ,bias = False)
		self.attw_t = nn.Linear(attention_size, 1)
		self.dropout = nn.Dropout(dropout)
		self.regression = nn.ModuleList([nn.Linear(2*hidden_size, 1) for i in range(4)])


	def forward(self, source, target, batch_size):
		output_s, repr_s = self.encoder_s(source, batch_size)
		output_t, repr_t = self.encoder_t(target, batch_size)
		
		weight_s = self.attw_s(F.relu(self.s2att_s(output_s) + self.t2att_s(repr_t).view(batch_size,1,-1)))#batch_size x seq_len
		weight_t = self.attw_t(F.relu(self.t2att_t(output_t) + self.s2att_s(repr_s).view(batch_size,1,-1)))#batch_size x seq_len

		repr_s = torch.sum(weight_s * output_s, 1)
		repr_t = torch.sum(weight_t * output_t, 1)

		repr_st = self.dropout(torch.cat((repr_s, repr_t), 1))
		score = [ regression(repr_st) for regression in self.regression]

		return score

def train_minibatch(input, target, score, model, optimizer, criterion):
	batch_size = len(input)
	optimizer.zero_grad()
	input = Variable(torch.LongTensor(input))
	target = Variable(torch.LongTensor(target))
	golden = Variable(torch.FloatTensor(score))

	if use_cuda:
		input = input.cuda()
		target = target.cuda()
		golden = golden.cuda()

	preds = model(input, target, batch_size)
	loss = 0.
	for i, (pred, max_score) in enumerate(zip(preds, [35, 25, 25 ,15])):
		loss += criterion(pred, golden[:,i]/ max_score)
	loss.backward()
	nn.utils.clip_grad_norm(model.parameters(), 5.)
	optimizer.step()

	return loss.data[0]


def train(model, vocab, args):
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	criterion = nn.MSELoss()
	data_loader = DataLoader(vocab, args.train_file)

	lowest_loss = 987654321.0
	for epoch in range(args.epochs):
                model.train()
		for input, target, score in data_loader.get_batches(args.batch_size):
			loss = train_minibatch(input, target, score, model, optimizer, criterion)
			sys.stdout.write('train loss: %.3f\r\r'%loss)
			sys.stdout.flush()
                model.eval()
		result = test(model, vocab, args)
		print ('Epoch %d finished'%(epoch+1))
		print (result)
		if sum(result) < lowest_loss:
			lowest_loss = sum(result)
			torch.save(model, args.save_path)

def test(model, vocab, args):
	data_loader = DataLoader(vocab, args.test_file)
	criterion = nn.MSELoss()
	loss = [0., 0., 0. ,0.]
	tot_size = 0
	for input, target, score in data_loader.get_batches(args.batch_size):
		if use_cuda:
			input = input.cuda()
			target = target.cuda()
			golden = golden.cuda()

		batch_size = len(input)
		input = Variable(torch.LongTensor(input))
		target = Variable(torch.LongTensor(target))
		golden = Variable(torch.FloatTensor(score))
		preds = model(input, target, batch_size)
		for i, (pred, max_score) in enumerate(zip(preds, [35, 25, 25, 15])):
                        #print pred - golden[:,i]
			loss[i] += batch_size * criterion(pred*max_score, golden[:, i]).data[0]
			tot_size += batch_size
        result = [ l/tot_size for l in loss ]
	return result


class Vocab(object):
	def __init__(self, train_file, pretrained_src = None, pretrained_tgt = None):
		src_cnt = Counter()
		tgt_cnt = Counter()
		with open(train_file) as f:
			for line in f.readlines():
				info  = line.strip().split('\t')
				assert len(info) == 6, (line,info )
				src_cnt.update(info[0].split())
				tgt_cnt.update(info[1].split())
		self._id2src = ['UNK', '</s>']
		self._id2tgt = ['UNK', '</s>']
		for w in src_cnt:
			if src_cnt[w]>=2:
				self._id2src.append(w)

		for w in tgt_cnt:
			if tgt_cnt[w]>=2:
				self._id2tgt.append(w)
		self.source_vocab_size  =  len(self._id2src)
		self.target_vocab_size  =  len(self._id2tgt)
		#print self.source_vocab_size, self.target_vocab_size

		self._src2id = dict(zip( self._id2src, range(self.source_vocab_size)))
		self._tgt2id = dict(zip( self._id2tgt, range(self.target_vocab_size)))
		self.pretrained_src = pretrained_src
		self.pretrained_tgt = pretrained_tgt

	def src2id(self, x):
		if type(x) is list:
			return [self._src2id.get( t, 0) for t in x]
		return self._src2id.get(x, 0)

	def id2src(self, x):
		if type(x) is list:
			return [self._id2src[t] for t in x]
		return self._id2src[x]	


	def tgt2id(self, x):
		if type(x) is list:
			return [self._tgt2id.get( t, 0) for t in x]
		return self._tgt2id.get(x, 0)

	def id2tgt(self, x):
		if type(x) is list:
			return [self._id2tgt[t] for t in x]
		return self._id2tgt[x]

	def get_pretrained_src(self):
		if self.pretrained_src is None:
			return None
		embs = [[]]*len(self._src2id)
		with open(self.pretrained_src) as f:
			for line in f.readlines():
				info = line.strip().split()
				word, data = info[0], info[1:]
				if word in self._src2id:
					embs[self.src2id(word)] = data

		emb_size = len(data)
		for idx, emb in enumerate(embs):
			if not emb:
				embs[idx] = np.zeros(emb_size)
		return np.array(embs, dtype=np.float32)

	def get_pretrained_tgt(self):
		if self.pretrained_tgt is None:
			return None
		embs = [[]]*len(self._tgt2id)
		with open(self.pretrained_tgt) as f:
			for line in f.readlines():
				info = line.strip().split()
				word, data = info[0], info[1:]
				if word in self._tgt2id:
					embs[self.tgt2id(word)] = data

		emb_size = len(data)
		for idx, emb in enumerate(embs):
			if not emb:
				embs[idx] = np.zeros(emb_size)
		return np.array(embs, dtype=np.float32)

class DataLoader(object):
	def __init__(self, vocab, fname):
		self.src_data = []
		self.tgt_data = []
		self.scores =[]
		self.vocab = vocab
		with open(fname) as f:
			for line in f.readlines():
				info  = line.strip().split('\t')
				assert len(info) == 6, line
				src = info[0].split()
				tgt = info[1].split()
				scores = [ float(x) for x in info[2:]]
				self.src_data.append(src)
				self.tgt_data.append(tgt)
				self.scores.append(scores)

	def get_batches(self, batch_size, shuffle = True):
		idx = list(range(len(self.src_data)))
		if shuffle:
			np.random.shuffle(idx)
		cur_size = 0
		input, target, score = [], [], []
		for _id in sorted(idx, key = lambda x: len(self.src_data[x])):
			cur_size += len(self.src_data[_id])
			input.append(self.src_data[_id])
			target.append(self.tgt_data[_id])
			score.append(self.scores[_id])
			if cur_size  >= batch_size:
				cur_size  = 0
				seq_len = max(len(t) for t in input)
				input = [ self.vocab.src2id(t) + [0]*(seq_len - len(t)) for t in input ]
				seq_len = max(len(t) for t in target)
				target = [ self.vocab.tgt2id(t) + [0]*(seq_len - len(t)) for t in target ]
				yield input, target, score


def main(args):
	vocab = Vocab(args.train_file, args.src_emb, args.tgt_emb)
	model = AttentionRegression(vocab, args.emb_size, args.feature_size, args.window_size, args.dropout, args.hidden_size, args.n_layers, args.attention_size)
	if use_cuda:
		model.cuda()
	model.train()
	train(model, vocab, args)

	model.eval()
	model.test()
	test(model,vocab, args)

import argparse
if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--emb_size', type = int, default= 200)
	argparser.add_argument('--feature_size', type = int, default= 100)
	argparser.add_argument('--window_size', nargs = '+', type = int, default= [1,2])
	argparser.add_argument('--dropout', type=float, default=0.2)
	argparser.add_argument('--hidden_size', type = int, default= 200)
	argparser.add_argument('--n_layers', type = int, default= 1)
	argparser.add_argument('--attention_size', type = int, default= 100)
	argparser.add_argument('--train_file', default= 'sent_training_data.txt')
	argparser.add_argument('--test_file', default= 'sent_testing_data.txt')
	argparser.add_argument('--learning_rate', type = float, default= 0.001)
	argparser.add_argument('--epochs', type = int, default= 10)
	argparser.add_argument('--batch_size', type = int, default= 512)
	argparser.add_argument('--src_emb', type=str, default= '/home/yuyuan/quality_estimation/glove.6B.200d.txt')
	argparser.add_argument('--tgt_emb', type=str, default= '/home/yuyuan/quality_estimation/wikizhword.emb')
        argparser.add_argument('--save_path', type=str, default= './translation_quality.pt')
	args, extra_args = argparser.parse_known_args()
	main(args)
