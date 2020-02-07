# -*- coding: utf-8 -*
import codecs
import sys
sys.path.append('..')
import numpy as np
import model.config.conf as conf
import copy
import random
import math
import json
class DataSet(object):
	def __init__(self,embedding_path,alphabet_path,train_path,test_path,valid_path,local_train_path=None):
		self.train_path = train_path
		self.test_path = test_path
		self.valid_path = valid_path
		self.local_train_path = local_train_path
		self.local_weight = conf.local_weight
		self.entity_dict = json.load(codecs.open(conf.entity_dict,'r','utf-8'))
		word_idf_dict_ = json.load(codecs.open(conf.word_idf_dict,'r','utf-8'))
		self.word2id,self.word_embeddings = self.load_word_embedding(embedding_path)
		self.id2word = {self.word2id[k]:k for k in self.word2id}
		print ('word embedding length',len(self.word2id))
		self.word_idf_dict = {}
		self.word_idf_dict['_sentence_num_'] = word_idf_dict_['_sentence_num_']
		word_idf_dict_['UNK'] = 10
		word_idf_dict_['BLANK'] = 10

		for w in self.word2id:
			self.word_idf_dict[self.word2id[w]] =  word_idf_dict_[w] if w in word_idf_dict_ else 10
		print('word embedding loaded')
		self.alphabet2id = self.load_alphabet_dict(alphabet_path)
		print('alphabet loaded')
		self.entity_nil = {}
		self.train_data_,self.test_data_,self.valid_data_,\
		self.num_train,self.num_test,self.num_valid,\
		self.entity_train,self.entity_test,self.entity_valid = self.load_dataset(loc_=True)

		self.train_data = self.sample_batch_iter(self.train_data_,conf.batch_size,conf.num_epochs,conf.sample_epochs,True)
		print('dataset loaded',self.num_train,self.num_test,self.num_valid)
		
		self.nofuzzy_train_data_,self.num_nofuzzy_train,self.entity_nofuzzy_train = self.load_bucket_data(conf.nofuzzy_train_path)
		self.nofuzzy_train_data = self.bucket_batch_iter(self.nofuzzy_train_data_, conf.nofuzzy_batch_size, conf.nofuzzy_num_epochs,True)
		print('nofuzzy dataset loaded',self.num_nofuzzy_train)
		if local_train_path and conf.use_local:
			if conf.local_window:
				self.local_train_data_, self.num_local_train = self.load_local_win_dataset()
			else:
				self.local_train_data_, self.num_local_train, self.entity_local_train = self.load_local_dataset()
			self.local_train_data = self.bucket_batch_iter(self.local_train_data_,conf.local_batch_size,conf.local_num_epochs,True)
			print('local dataset loaded', self.num_local_train)
			if conf.weighted_local_loss:
				max_type = max(self.local_weight)
				self.local_weight[conf.type2id[u'B-Person']] = max_type/self.local_weight[conf.type2id[u'B-Person']]
				self.local_weight[conf.type2id[u'B-Location']] = max_type/self.local_weight[conf.type2id[u'B-Location']]
				self.local_weight[conf.type2id[u'B-Organisation']] = max_type/self.local_weight[conf.type2id[u'B-Organisation']]
				self.local_weight[conf.type2id[u'I-Person']] = self.local_weight[conf.type2id[u'B-Person']]
				self.local_weight[conf.type2id[u'I-Location']] = self.local_weight[conf.type2id[u'B-Location']]
				self.local_weight[conf.type2id[u'I-Organisation']] = self.local_weight[conf.type2id[u'B-Organisation']]
				self.local_weight[conf.type2id[u'O']] = 1.
				print('local_weight',self.local_weight)
	def load_alphabet_dict(self,alphabet_path):
		f_alphabet = codecs.open(alphabet_path,'r','utf-8')
		lines = f_alphabet.readlines()
		alphabet2id = {}
		for line in lines:
			character = line.strip().split(u' ')[0]
			alphabet2id[character] = len(alphabet2id)
		alphabet2id[u'NUM'] = len(alphabet2id)
		alphabet2id[u'UNK'] = len(alphabet2id)
		alphabet2id[u'BLANK'] = len(alphabet2id)
		f_alphabet.close()
		return alphabet2id

	def load_word_embedding(self,embedding_path):
		f_embedding = codecs.open(embedding_path,'r','utf-8')
		line = f_embedding.readline()
		line = f_embedding.readline()
		word2id = {}
		word_embeddings = []
		while line:
			line = line.strip('\n').split(u' ')
			temp = []
			for dim in line[1:]:
				temp.append(float(dim))
			if line[0] in word2id:
				ex = word_embeddings[word2id[line[0]]]
				if ex != temp:
					print('duplicate word embedding',line[0])
				line = f_embedding.readline()
				continue
			if (len(word_embeddings)>0) and (len(temp) != len(word_embeddings[0])):
				print('wrong dim',line)
				line = f_embedding.readline()
				continue
			word2id[line[0]] = len(word2id)
			word_embeddings.append(temp)
			line = f_embedding.readline()
		f_embedding.close()
		word2id[u'UNK'] = len(word2id)
		word2id[u'BLANK'] = len(word2id)
		lists = [0.0 for tempi in range(len(word_embeddings[0]))]
		word_embeddings.append(lists)
		word_embeddings.append(lists)
		return word2id,np.array(word_embeddings).astype(float)
	
	def load_data(self,path,shuffle = True,loc = False):
		f_in = codecs.open(path,'r','utf-8')
		line = f_in.readline()
		half_max_word_len = conf.max_word_len//2
		x_w = []
		x_c = []
		y = []
		length = []
		locations = []
		while line:
			parts = line.strip().split(u'\t')
			content = parts[0]
			annotations = parts[1]
			if loc:
				location = parts[2].split(u' ')
				for l in range(len(location)):
					location[l] = int(location[l])
				locations.append(location)
			words = content.split(u' ')
			annotations = annotations.split(u' ')
			
			plos = [] 
			for tempi in annotations:
				if tempi in conf.type2id:
					plos.append(conf.type2id[tempi])
				else:
					plos.append(conf.type2id[u'O'])
			if len(plos) != len(words):
				print('words and plos length unmatch!',line)
				line = f_in.readline()
				continue

			wordids = []
			characterids = []
			for word in words:
				if word in self.word2id:
					wordids.append(self.word2id[word])
				else:
					wordids.append(self.word2id[u'UNK'])
				characters = [self.alphabet2id[u'BLANK'] for i in  range(conf.max_word_len)]
				if len(word) > conf.max_word_len:
					word = word[:half_max_word_len]+word[-(conf.max_word_len-half_max_word_len):]
				for c in range(len(word)):
					if word[c] in self.alphabet2id:
						characters[c] = self.alphabet2id[word[c]]
					elif word[c].isdigit():
						characters[c] = self.alphabet2id[u'NUM']
					else:
						characters[c] = self.alphabet2id[u'UNK']
				characterids.append(characters)
			x_w.append(wordids)
			x_c.append(characterids)
			y.append(plos)
			length.append(len(wordids))
			line = f_in.readline()
		f_in.close()
		x_w = np.array(x_w)
		x_c = np.array(x_c)
		y = np.array(y)
		length = np.array(length)
		if loc:
			return list(zip(x_w,x_c,y,length,np.array(locations))),len(x_w)
		else:
			return list(zip(x_w,x_c,y,length)),len(x_w)
	def load_bucket_data(self,path,shuffle = True,loc = False,sample = None,ratio = 0.5):
		entity_dict = {}
		f_in = codecs.open(path,'r','utf-8')
		line = f_in.readline()
		half_max_word_len = conf.max_word_len//2
		x_w = []
		x_c = []
		y = []
		length = []
		locations = []
		bucket = {}
		while line:
			parts = line.strip('\n').split(u'\t')
			content = parts[0]
			annotations = parts[1]
			location = []

			words = content.split(u' ')
			annotations = annotations.split(u' ')
			
			plos = [] 
			for tempi in annotations:
				if tempi in conf.type2id:
					plos.append(conf.type2id[tempi])
				elif (tempi in conf.fuzzy2id) and conf.use_nil:
					plos.append(conf.fuzzy2id[tempi])
				else:
					plos.append(conf.type2id[u'O'])
			if len(plos) != len(words):
				print('words and plos length unmatch!',line)
				line = f_in.readline()
				continue

			wordids = []
			characterids = []
			for word in words:
				if word in self.word2id:
					wordids.append(self.word2id[word])
				else:
					wordids.append(self.word2id[u'UNK'])
				characters = [self.alphabet2id[u'BLANK'] for i in  range(conf.max_word_len)]
				if len(word) > conf.max_word_len:
					word = word[:half_max_word_len]+word[-(conf.max_word_len-half_max_word_len):]
				for c in range(len(word)):
					if word[c] in self.alphabet2id:
						characters[c] = self.alphabet2id[word[c]]
					elif word[c].isdigit():
						characters[c] = self.alphabet2id[u'NUM']
					else:
						characters[c] = self.alphabet2id[u'UNK']
				characterids.append(characters)
			if sample and np.random.random() > sample:
				line = f_in.readline()
				continue
			len_ = len(wordids)
			if len_ not in bucket:
				bucket[len_] = {'x_w':[],'x_c':[],'y':[],'length':[],'location':[]}
			bucket[len_]['x_w'].append(wordids)
			bucket[len_]['x_c'].append(characterids)
			bucket[len_]['y'].append(plos)
			
			bucket[len_]['length'].append(len(wordids))
			if loc:
				location = self.locate_sentence2([wordids],[plos],ratio)[0]
				bucket[len_]['location'].append(location)
			for temp in range(len_):
				if plos[temp] in conf.begin:
					entity = [str(wordids[temp])]
					if loc:
						self.local_weight[plos[temp]] += 1.
					for temp2 in range(temp+1,len_):
						if plos[temp2] in conf.inside:
							entity.append(str(wordids[temp2]))
						else:
							break
					entity_dict['_'.join(entity)] = 1
				elif conf.use_nil and plos[temp] == conf.fuzzy2id['B-Nil']:
					entity = [str(wordids[temp])]
					for temp2 in range(temp+1,len_):
						if plos[temp2] == conf.fuzzy2id['I-Nil']:
							entity.append(str(wordids[temp2]))
						else:
							break
					self.entity_nil['_'.join(entity)] = 1

			line = f_in.readline()
		f_in.close()
		all_len_ = list(bucket)
		if 1 in all_len_:
			print('length = 1:',len(bucket[1]['x_c']),path)
			all_len_.remove(1)
		all_len_.sort(reverse=True)
		print ('max_length',all_len_[0])
		for len_ in all_len_:
			x_w.extend(bucket[len_]['x_w'])
			x_c.extend(bucket[len_]['x_c'])
			y.extend(bucket[len_]['y'])
			length.extend(bucket[len_]['length'])
			if loc:
				locations.extend(bucket[len_]['location']) 
		bucket = {}
		x_w = np.array(x_w)
		x_c = np.array(x_c)
		y = np.array(y)
		length = np.array(length)

		if loc:
			return list(zip(x_w,x_c,y,length,np.array(locations))),len(x_w),entity_dict
		else:
			return list(zip(x_w,x_c,y,length)),len(x_w),entity_dict
	def load_local_win_data(self,path):
		f_in = codecs.open(path,'r','utf-8')
		line = f_in.readline()
		half_max_word_len = conf.max_word_len//2
		x_w = []
		x_c = []
		y = []
		mask = []
		length = []
		while line:
			parts = line.strip().split(u'\t')
			content = parts[0]
			annotations = parts[1].split(u' ')
			locations = parts[2].split(u' ')
			for tempi in range(len(locations)):
				locations[tempi] = int(locations[tempi])
			words = content.split(u' ')
			plos = [] 
			for tempi in annotations:
				if tempi in conf.type2id:
					plos.append(conf.type2id[tempi])
				else:
					plos.append(conf.type2id[u'O'])
			if len(plos) != len(words):
				print('words and plos length unmatch!',line)
				line = f_in.readline()
				continue
			wordids = []
			characterids = []
			for word in words:
				if word in self.word2id:
					wordids.append(self.word2id[word])
				else:
					wordids.append(self.word2id[u'UNK'])
				characters = [self.alphabet2id[u'BLANK'] for i in  range(conf.max_word_len)]
				if len(word) > conf.max_word_len:
					word = word[:half_max_word_len]+word[-(conf.max_word_len-half_max_word_len):]
				for c in range(len(word)):
					if word[c] in self.alphabet2id:
						characters[c] = self.alphabet2id[word[c]]
					elif word[c].isdigit():
						characters[c] = self.alphabet2id[u'NUM']
					else:
						characters[c] = self.alphabet2id[u'UNK']
				characterids.append(characters)
			local_x_ws,local_x_cs,local_ys,local_masks,span_nums=self.get_local(\
				[wordids],[characterids],[plos],[len(wordids)],[locations])
			x_w.extend(local_x_ws)
			x_c.extend(local_x_cs)
			y.extend(local_ys)
			mask.extend(local_masks)
			line = f_in.readline()
		f_in.close()
		x_w = np.array(x_w)
		x_c = np.array(x_c)
		y = np.array(y)
		mask = np.array(mask)
		return list(zip(x_w,x_c,y,mask)),len(x_w)

	def load_dataset(self,loc_=False):
		train_data,num_train,entity_train = self.load_bucket_data(self.train_path,sample = conf.train_sample,loc=loc_,ratio=conf.sample_ratio)
		test_data,num_test,entity_test = self.load_bucket_data(self.test_path,shuffle=False)
		valid_data,num_valid,entity_valid = self.load_bucket_data(self.valid_path,shuffle=False)
		return train_data,test_data,valid_data,num_train,num_test,num_valid,entity_train,entity_test,entity_valid
	def load_local_win_dataset(self):
		train_local_data,num_local_train = self.load_local_win_data(self.local_train_path)
		return train_local_data,num_local_train
	def load_local_dataset(self):
		train_local_data,num_local_train,entity_local_train = self.load_bucket_data(self.local_train_path, loc = True, ratio=conf.local_sample_ratio)
		return train_local_data,num_local_train,entity_local_train
	def pad_sentence(self,x_w_batch,x_c_batch,y_batch,length_batch):
		max_length = max(length_batch)
		padded_x_w_batch = []
		padded_x_c_batch = []
		padded_y_batch = []
		characters = [self.alphabet2id[u'BLANK'] for i in  range(conf.max_word_len)]
		mask_batch = []
		for i in range(len(x_w_batch)):
			padded_x_w = copy.deepcopy(x_w_batch[i])
			padded_x_c = copy.deepcopy(x_c_batch[i])
			padded_y = copy.deepcopy(y_batch[i])
			mask = [1 for j in range(length_batch[i])]
			for j in range(max_length-len(x_w_batch[i])):
				padded_x_w.append(self.word2id[u'BLANK'])
				padded_x_c.append(characters)
				padded_y.append(conf.type2id[u'O'])
				mask.append(0)
			padded_x_w_batch.append(padded_x_w)
			padded_x_c_batch.append(padded_x_c)
			padded_y_batch.append(padded_y)
			mask_batch.append(mask)
		return padded_x_w_batch,padded_x_c_batch,padded_y_batch,mask_batch
	def locate_sentence(self,y_batch):
		locations_batch = []
		for i in range(len(y_batch)):
			locations = []
			bound = []
			notbound = []
			for j in range(len(y_batch[i])):
				if y_batch[i][j] != conf.type2id['O']:
					locations.append(j)
					if (j-1 >= 0) and (y_batch[i][j-1] == conf.type2id['O']):
						if j-1 not in bound:
							bound.append(j-1)
					if (j+1 < len(y_batch[i])) and (y_batch[i][j+1] == conf.type2id['O']):
						if j+1 not in bound:
							bound.append(j+1)
			for j in range(len(y_batch[i])):
				if (j not in locations) and (j not in bound):
					notbound.append(j)
			notO_ratio = float(len(locations))/len(y_batch[i])
			bound_ratio = min(1.0, 0.7+notO_ratio)
			notbound_ratio = min(1.0, 0.05+notO_ratio*3)
			br = int(math.ceil(len(bound)*bound_ratio))
			notbr = int(math.ceil(len(notbound)*notbound_ratio))
			if br > 0:
				locations.extend(np.random.choice(bound,size=br,replace=False).tolist())
			if notbr > 0:
				locations.extend(np.random.choice(notbound,size=notbr,replace=False).tolist())
			locations.sort()
		locations_batch.append(locations)
		return locations_batch
	def locate_sentence2(self,x_batch,y_batch,ratio):
		locations_batch = []
		weight = np.array(conf.sample_feature_weight)
		def softmax(x):
			e_x = np.exp(x - np.max(x))
			return e_x / e_x.sum(axis=0) 
		for i in range(len(y_batch)):
			locations = []
			tosample = []
			sample_feature = []
			wf = {}
			for w in x_batch[i]:
				if w not in wf:
					wf[w] = 1.
				else:
					wf[w] += 1.

			for j in range(len(y_batch[i])):
				if y_batch[i][j] != conf.type2id['O']:
					locations.append(j)
				else:
					tosample.append(j)
					feature = [0.,0.,0.]
					if (j > 1 and y_batch[i][j-1] != conf.type2id['O']) or (j+1<len(y_batch[i]) and y_batch[i][j+1] != conf.type2id['O']):
						feature[0] = 1.
					if y_batch[i][j] in self.entity_dict:
						feature[1] = 1.
					feature[2] = np.log(self.word_idf_dict[x_batch[i][j]]) - np.log(self.word_idf_dict['_sentence_num_'])\
					+ np.log(wf[x_batch[i][j]]) - np.log(len(x_batch[i]))
					sample_feature.append(feature)
			if len(tosample) > 0:
				sample_feature = np.array(sample_feature)
				sample_score = np.matmul(sample_feature, weight.T)
				sample_prob = softmax(sample_score)
				sample_num = int(math.floor(len(tosample)*ratio))
				locations.extend(np.random.choice(tosample, size = sample_num, replace=False, p=sample_prob).tolist())
			locations.sort()
			locations_batch.append(locations)
		return locations_batch
	def get_local(self,x_w,x_c,y,length,locations=None):
		local_x_ws = []
		characters = [self.alphabet2id[u'BLANK'] for tempi in  range(conf.max_word_len)]
		local_x_cs = []
		local_masks = []
		local_ys = []
		span_nums = []
		span = conf.span
		for s in range(len(x_w)):
			if locations:
				locs = locations[s]
			else:
				locs = range(len(x_w[s]))
			span_nums.append(len(locs))
			sentence = x_w[s]
			for loc in  locs:
				local_x_w = [self.word2id[u'BLANK'] for tempi in range(conf.span*2+1)]
				local_x_c = [characters for tempi in range(conf.span*2+1)]
				local_mask = np.array([0 for tempi in range(conf.span*2+1)])
				local_ys.append(y[s][loc])
				local_x_w[span] = sentence[loc]
				local_x_c[span] = x_c[s][loc]
				local_mask[span] = 1
				if loc < span:
					local_x_w[span-loc:span] = sentence[:loc]
					local_x_c[span-loc:span] = x_c[s][:loc]
					local_mask[span-loc:span] = 1
				else:
					local_x_w[:span] = sentence[loc-span:loc]
					local_x_c[:span] = x_c[s][loc-span:loc]
					local_mask[:span] = 1
				if loc + span > length[s]-1:
					local_x_w[span+1:span+length[s]-loc] = sentence[loc+1:]
					local_x_c[span+1:span+length[s]-loc] = x_c[s][loc+1:]
					local_mask[span+1:span+length[s]-loc] = 1
				else:
					local_x_w[span+1:] = sentence[loc+1:loc+1+span]
					local_x_c[span+1:] = x_c[s][loc+1:loc+1+span]
					local_mask[span+1:] = 1
				local_x_ws.append(local_x_w)
				local_x_cs.append(local_x_c)
				local_masks.append(local_mask.tolist())
		return local_x_ws,local_x_cs,local_ys,local_masks,span_nums
	def batch_iter(self,data, batch_size, num_epochs, shuffle=True):
		"""
		Generates a batch iterator for a dataset.
		"""
		data = np.array(data)
		data_size = len(data)
		num_batches_per_epoch = (int)(len(data)/batch_size)
		for epoch in range(num_epochs):
			# Shuffle the data at each epoch
			if shuffle:
				shuffle_indices = np.random.permutation(np.arange(data_size))
				shuffled_data = data[shuffle_indices]
			else:
				shuffled_data = data
			for batch_num in range(num_batches_per_epoch):
				start_index = batch_num * batch_size
				end_index = min((batch_num + 1) * batch_size, data_size)
				yield shuffled_data[start_index:end_index]
			start_index = num_batches_per_epoch * batch_size
			end_index = min(start_index + batch_size, data_size)
			if start_index < data_size:
				yield shuffled_data[start_index:end_index]
	def bucket_batch_iter(self,data,batch_size,num_epochs,shuffle=True):
		data = np.array(data)
		data_size = len(data)
		num_batches_per_epoch = int(len(data)/batch_size)+int(len(data)%batch_size > 0)
		for epoch in range(num_epochs):
			if shuffle:
				batch_seq = np.random.permutation(np.arange(num_batches_per_epoch))
			else:
				batch_seq = np.arange(num_batches_per_epoch)
			for batch_index in batch_seq:
				start_index = batch_index * batch_size
				end_index = min((batch_index+1)*batch_size,data_size)
				yield data[start_index:end_index]
	def sample_batch_iter(self,data,batch_size,num_epochs,sample_epochs,shuffle=True):
		data_ = np.array(data)
		data_size = len(data_)
		num_batches_per_epoch = int(len(data_)/batch_size)+int(len(data_)%batch_size > 0)
		for epoch in range(num_epochs):
			if (epoch > 0)  and (epoch % sample_epochs == 0):		
				x_w,x_c,y,length,locations = zip(*data)
				locations = self.locate_sentence2(x_w,y,conf.sample_ratio)
				locations = np.array(locations)
				data_ = list(zip(x_w,x_c,y,length,locations))
				data_ = np.array(data_)
			if shuffle:
				batch_seq = np.random.permutation(np.arange(num_batches_per_epoch))
			else:
				batch_seq = np.arange(num_batches_per_epoch)
			for batch_index in batch_seq:
				start_index = batch_index * batch_size
				end_index = min((batch_index+1)*batch_size,data_size)
				yield data_[start_index:end_index]
	def get_test_iter(self):
		return self.bucket_batch_iter(self.test_data_,conf.batch_size,1,False)
	def get_valid_iter(self):
		return self.bucket_batch_iter(self.valid_data_,conf.batch_size,1,False)