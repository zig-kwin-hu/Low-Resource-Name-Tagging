# -*- coding: utf-8 -*
#symbol parameter
only_nil = True
fuzzy = True
NOT_CONFIRM_IDX = -3
START_TAG = "<START>"
STOP_TAG = "<STOP>"
type2id = {u'O':0,u'B-Person':1,u'I-Person':2,u'B-Location':3,u'I-Location':4,\
u'B-Organisation':5,u'I-Organisation':6}
id2type = {0:u'O',1:u'B-Person',2:u'I-Person',3:u'B-Location',4:u'I-Location',\
5:u'B-Organisation',6:u'I-Organisation'}
local_weight = [1. for temp in range(len(type2id))]
begin = {1:0,3:1,5:2}
inside = {2:0,4:1,6:2}
id2type3 = {0:'Person',1:'Location',2:'Organisation'}
seven2three = {0:-1,1:0,2:0,3:1,4:1,5:2,6:2}
sample_feature_weight = [0,-10,1]
fuzzy2id = {'not_conf':-3,'B-Nil':-2,'I-Nil':-1}

#input parameter
use_nil = False
use_trans = False
use_local = True
weighted_local_loss = False
local_window = False
early_stop = True
use_fuzzy = True
joint = True

#model parameter
filter_sizes = [3]
filter_nums = [30]
char_dim = 30
position_size = 20
nb_head = 8
size_per_head = 150
trans_output_size = 150
num_layers = 1
bi_flag = True
max_word_len = 15
span = 5
#train parameter
evaluate_times = 1
sample_epochs = 10
sample_ratio = 0.9
local_sample_ratio = 0.9
train_sample = 1
learning_rate = 1e-3
local_learning_rate = 1e-3
lr_decay = 0
local_lr_decay = 0
l2_rate = 1e-5
batch_size = 64
nofuzzy_batch_size = 32
local_batch_size = 64
num_epochs = 100
nofuzzy_num_epochs = 100
local_num_epochs = 20
dropout_rate = 0.5
max_patience = 100
local_max_patience = 10
batch_per_eva = 100
local_batch_per_eva = 120
