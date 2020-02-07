import numpy as np
import datetime
import sys
sys.path.append('..')
from model.config import conf as conf
class PrecRecallCounter(object):
	def __init__(self, class_nums, log_dir, name, number, entity_train=None):
		if type(class_nums) == int:
			self.class_nums = np.array([class_nums])
		elif type(class_nums) == list:
			self.class_nums = np.array(class_nums)
		else:
			print('input must be int or list')
			exit()
		self.log_dir = log_dir
		self.name = name
		self.number = number

		self.total = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.pred = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.correct = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.accuracy_class = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.recall_class = np.array([[0.0 for j in range(i)] for i in self.class_nums])
		self.accuracy_object = np.array([0.0 for i in self.class_nums])
		self.accuracy_total = 0.0
		self.macro_acc = np.array([0.0 for i in self.class_nums])
		self.macro_recall = np.array([0.0 for i in self.class_nums])

		self.entity_TP = self.entity_FP = self.entity_FN = self.entity_TN = 0.
		self.entity_TP_type = self.entity_FP_type = self.entity_FN_type = self.entity_TN_type = 0.
		self.TPs = np.array([0. for i in range(len(conf.begin))])
		self.FPs = np.array([0. for i in range(len(conf.begin))])
		self.FNs = np.array([0. for i in range(len(conf.begin))])
		self.entity_train = entity_train
		self.in_train_y = 0
		self.out_train_y = 0
		self.in_train_p = 0
		self.out_train_p = 0
	def count(self, y, correct, x=0):
		if (correct >= self.class_nums[x]):
			return
		self.total[x][correct] += 1
		self.pred[x][y] += 1
		if correct == y:
			self.correct[x][y] += 1
	def count_entity(self, prediction,y,length,x=0):
		def get_entities(sentence,l):
			current = 0
			entities = []
			for word in range(l):
				if sentence[word] in conf.begin:
					entity = [word]
					entity_type = conf.seven2three[sentence[word]]
					for i in range(word+1,l):
						if (sentence[i] not in conf.inside) or (conf.seven2three[sentence[i]] != conf.seven2three[sentence[word]]):
							break
						entity.append(i)
					entities.append([entity,entity_type])
			return entities
		batch_size = len(length)
		for b in range(batch_size):
			predict_entities = get_entities(prediction[b],length[b])
			y_entities = get_entities(y[b],length[b])
			tocompare = 0
			tp = tp_type = 0.
			ps = np.array([0. for i in range(len(conf.begin))])
			ys = np.array([0. for i in range(len(conf.begin))])
			tps = np.array([0. for i in range(len(conf.begin))])
			for e in y_entities:
				ys[e[1]] += 1
			for e in predict_entities:
				ps[e[1]] += 1
				for i in range(tocompare,len(y_entities)):
					if y_entities[i][0][0] > e[0][0]:
						break
					if (y_entities[i][0][0] == e[0][0]) and (y_entities[i][0][-1] == e[0][-1]):
						tp += 1
						if y_entities[i][1] == e[1]:
							tp_type += 1
							tps[e[1]] += 1
						tocompare = i+1
			self.entity_TP += tp
			self.entity_TP_type += tp_type
			self.entity_FP += (len(predict_entities)-tp)
			self.entity_FP_type += (len(predict_entities)-tp_type)
			self.entity_FN += (len(y_entities)-tp)
			self.entity_FN_type += (len(y_entities)-tp_type)

			self.TPs += tps
			self.FPs += (ps-tps)
			self.FNs += (ys-tps)
	def count_entity_overlap(self, prediction,y,length,x):
		def get_entities_x(sentence,l,x):
			current = 0
			entities = []
			entities_x = []
			for word in range(l):
				if sentence[word] in conf.begin:
					entity = [word]
					entity_x = [str(x[word])]
					entity_type = conf.seven2three[sentence[word]]
					for i in range(word+1,l):
						if (sentence[i] not in conf.inside) or (conf.seven2three[sentence[i]] != conf.seven2three[sentence[word]]):
							break
						entity.append(i)
						entity_x.append(str(x[i]))
					entities.append([entity,entity_type,'_'.join(entity_x)])
			return entities
		batch_size = len(length)
		for b in range(batch_size):
			predict_entities = get_entities_x(prediction[b],length[b],x[b])
			y_entities = get_entities_x(y[b],length[b],x[b])
			tocompare = 0
			tp = tp_type = 0.
			ps = np.array([0. for i in range(len(conf.begin))])
			ys = np.array([0. for i in range(len(conf.begin))])
			tps = np.array([0. for i in range(len(conf.begin))])
			for e in y_entities:
				ys[e[1]] += 1

			p_T = [False for temp in range(len(predict_entities))]
			y_T = [False for temp in range(len(y_entities))]
			for e in range(len(predict_entities)):
				ps[predict_entities[e][1]] += 1
				for i in range(tocompare,len(y_entities)):
					if y_entities[i][0][0] > predict_entities[e][0][0]:
						break
					if (y_entities[i][0][0] == predict_entities[e][0][0]) and (y_entities[i][0][-1] == predict_entities[e][0][-1]):
						tp += 1
						if y_entities[i][1] == predict_entities[e][1]:
							tp_type += 1
							tps[predict_entities[e][1]] += 1
							p_T[e] = True
							y_T[i] = True
						tocompare = i+1
						break
			for e in range(len(p_T)):
				if not p_T[e]:
					if predict_entities[e][2] in self.entity_train:
						self.out_train_p += 1
					else:
						self.in_train_p += 1
			for e in range(len(y_T)):
				if not y_T[e]:
					if y_entities[e][2] in self.entity_train:
						self.out_train_y += 1
					else:
						self.in_train_y += 1


			self.entity_TP += tp
			self.entity_TP_type += tp_type
			self.entity_FP += (len(predict_entities)-tp)
			self.entity_FP_type += (len(predict_entities)-tp_type)
			self.entity_FN += (len(y_entities)-tp)
			self.entity_FN_type += (len(y_entities)-tp_type)

			self.TPs += tps
			self.FPs += (ps-tps)
			self.FNs += (ys-tps)
	def count_sequence(self,prediction,y,length,x=0):
		batch_size = len(length)
		for b in range(batch_size):
			for i in range(length[b]):
				self.total[x][y[b][i]]+=1
				self.pred[x][prediction[b][i]]+=1
				if prediction[b][i] == y[b][i]:
					self.correct[x][y[b][i]] += 1

	def multicount(self, y, indexes, x = 0):
		if len(indexes) == 0:
			return
		for index in indexes:
			self.total[x][index] += 1
			if y == index:
				self.correct[x][y] += 1
		self.pred[x][y] += 1
	def compute(self):
		self.accuracy_class = self.correct/(self.pred+1e-3)
		self.recall_class = self.correct/(self.total+1e-3)
		self.accuracy_object = np.sum(self.correct, axis=1)/(np.sum(self.total, axis=1)+1e-3)
		self.accuracy_total = np.sum(self.correct)/(np.sum(self.total)+1e-3)
		self.macro_acc = np.sum(self.accuracy_class, axis=1)/self.class_nums
		self.macro_recall = np.sum(self.recall_class, axis=1)/self.class_nums
		self.f1_class = 2*self.accuracy_class*self.recall_class/(self.accuracy_class+self.recall_class+1e-10)
		self.macro_f1 = np.sum(self.f1_class, axis = 1)/self.class_nums
		self.macro_acc_noO = np.sum(self.accuracy_class[0][1:])/(self.class_nums-1)
		self.macro_rec_noO = np.sum(self.recall_class[0][1:])/(self.class_nums-1)
	def compute_entity(self):
		self.precision = self.entity_TP/(self.entity_TP+self.entity_FP+1e-3)
		self.precision_type = self.entity_TP_type/(self.entity_TP_type+self.entity_FP_type+1e-3)
		self.recall = self.entity_TP/(self.entity_TP+self.entity_FN+1e-3)
		self.recall_type = self.entity_TP_type/(self.entity_TP_type+self.entity_FN_type+1e-3)
		self.f1 = 2*self.precision*self.recall/(self.precision+self.recall+1e-3)
		self.f1_type = 2*self.precision_type*self.recall_type/(self.precision_type+self.recall_type+1e-3)

		self.precisions = self.TPs/(self.TPs+self.FPs+1e-3)
		self.recalls = self.TPs/(self.TPs+self.FNs+1e-3)
		self.f1s = 2*self.precisions*self.recalls/(self.precisions+self.recalls+1e-3)
	def output_entity(self):
		print(self.name)
		print(datetime.datetime.now().isoformat())
		print('precision:', np.round(self.precision,4), 'recall:', np.round(self.recall,4),
			'f1', np.round(self.f1,4))
		print('precision type:', np.round(self.precision_type,4), 'recall type:', np.round(self.recall_type,4),
			'f1 type', np.round(self.f1_type,4))
		if self.entity_train:
			print('in_train_p:',self.in_train_p,'out_train_p:',self.out_train_p,'in_train_y:',self.in_train_y,'out_train_y',self.out_train_y)
		f = open(self.log_dir + str(self.number), 'w')
		tn = datetime.datetime.now()
		f.write(tn.isoformat()+'\n')
		f.write('precision:'+str(np.round(self.precision,4)) + '\n')
		f.write('recall:' + str(np.round(self.recall,4)) + '\n')
		f.write('f1:' + str(np.round(self.f1,4)) + '\n')
		f.write('precision type:'+str(np.round(self.precision_type,4)) + '\n')
		f.write('recall type:' + str(np.round(self.recall_type,4)) + '\n')
		f.write('f1 type:' + str(np.round(self.f1_type,4)) + '\n')
		f.write('in_train_p:'+str(self.in_train_p)+' out_train_p:'+str(self.out_train_p)+' in_train_y:'+str(self.in_train_y)+' out_train_y:'+str(self.out_train_y)+'\n')
		
		f.write('precisions:'+'\n')
		towrite = ''
		for i in range(len(conf.id2type3)):
			towrite = towrite + str(conf.id2type3[i]) + ':' + str(round(self.precisions[i],4)) + ' '
		towrite = towrite + '\n'
		f.write(towrite)

		f.write('recalls:'+'\n')
		towrite = ''
		for i in range(len(conf.id2type3)):
			towrite = towrite + str(conf.id2type3[i]) + ':' + str(round(self.recalls[i],4)) + ' '
		towrite = towrite + '\n'
		f.write(towrite)

		f.write('f1s:'+'\n')
		towrite = ''
		for i in range(len(conf.id2type3)):
			towrite = towrite + str(conf.id2type3[i]) + ':' + str(round(self.f1s[i],4)) + ' '
		towrite = towrite + '\n'
		f.write(towrite)

		f.write('accuracy_class:'+'\n')
		for i in range(len(self.class_nums)):
			towrite = ''
			for j in range(self.class_nums[i]):
				towrite = towrite + str(conf.id2type[j]) + ':' + str(round(self.accuracy_class[i][j],4)) + ' '
			towrite = towrite + '\n'
			f.write(towrite)

		f.write('recall_class:'+'\n')
		for i in range(len(self.class_nums)):
			towrite = ''
			for j in range(self.class_nums[i]):
				towrite = towrite + str(conf.id2type[j]) + ':' + str(round(self.recall_class[i][j],4)) + ' '
			towrite = towrite + '\n'
			f.write(towrite)

		f.write('accuracy_object:'+'\n')
		towrite = ''
		for i in range(len(self.class_nums)):
			towrite = towrite + str(round(self.accuracy_object[i],4)) + ' '
		towrite = towrite + '\n'
		f.write(towrite)

		f.write('pred:'+'\n')
		for i in range(len(self.class_nums)):
			towrite = ''
			for j in range(self.class_nums[i]):
				towrite = towrite + str(conf.id2type[j]) + ':' + str(self.pred[i][j]) + ' '
			towrite = towrite + '\n'
			f.write(towrite)

		f.write('total:'+'\n')
		for i in range(len(self.class_nums)):
			towrite = ''
			for j in range(self.class_nums[i]):
				towrite = towrite + str(conf.id2type[j]) + ':' + str(self.total[i][j]) + ' '
			towrite = towrite + '\n'
			f.write(towrite)

		f.close()