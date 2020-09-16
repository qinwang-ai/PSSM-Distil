from collections import OrderedDict

pretrain_path = './module/'
epochs = 300
gen_lr = 1e-3
lstm_hidden_dim = 400
lstm_hidden_dim_att = 256
lstm_num_layers = 2
embed_dim = 32
profile_width = 20
kernel_num = 200
kernel_sizes = [3, 3, 5]
dropout = 0.75
batch_size = 16

num_labels = 3
psm_real_data_path_prefix = './low_pssms/'
psm_fake_data_path_prefix = './bert_pssms/'
sequence_data_path_prefix = './sequences/'
label_data_path_prefix = './labels/'
# will be overwritten by passed parameters
train_scratch = True

# > 60, > 30, > 10
# 0 MSA means there only one item in .a3m file, hence, > 1
bins = [-1, 1]

IUPAC_VOCAB = OrderedDict(
    [('A', 0), ('R', 1), ('N', 2), ('D', 3),
     ('C', 4), ('Q', 5), ('E', 6), ('G', 7),
     ('H', 8), ('I', 9), ('L', 10),
     ('K', 11), ('M', 12), ('F', 13), ('P', 14),
     ('S', 15), ('T', 16), ('W', 17), ('Y', 18), ('V', 19), ('X', 20)])
