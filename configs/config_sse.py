from collections import OrderedDict
# from configs.DataPathRef90.cullpdb6125 import *
from configs.DataPathRef90.bc40 import *
# from configs.DataPathRef90.pfam import *
# from configs.DataPathRef90.casp10113 import *
# from configs.DataPathRef90.cb513 import *
# from configs.DataPathRef90.cameo import *
# from configs.DataPath.cameo import *

pretrain_path = '../module/'
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

# ss3 or ss8
# num_labels = 3
# label_data_path_prefix = label3_data_path_prefix
num_labels = 8
label_data_path_prefix = label8_data_path_prefix

# will be overwritten by passed parameters
train_scratch = True

# > 60, > 30, > 10
# 0 MSA means there only one item in .a3m file, hence, > 1
bins = [-1, 1]

#bins = [-1, 5]
# bins = [5, 10]
# bins = [10, 30]
# bins = [30, 50]
# bins = [50, 70]
# bins = [70, 90]
# bins = [90, 110]
# bins = [110, 200]
# bins = [200, 300]
# bins = [300, 400]
# bins = [400, 500]

high_quality_list = list(filter(lambda x: int(x.split(' ')[1]) > bins[1], open(num_file, 'r').readlines()))
low_quality_list = list(
    filter(lambda x: int(x.split(' ')[1]) > bins[0] and x not in high_quality_list, open(num_file, 'r').readlines()))

# only keep number without filename
low_quality_num = list(map(lambda x: int(x.split(' ')[1].strip()), low_quality_list))

# only keep filename without number
high_quality_list = list(map(lambda x: x.split(' ')[0].strip(), high_quality_list))
low_quality_list = list(map(lambda x: x.split(' ')[0].strip(), low_quality_list))


IUPAC_VOCAB = OrderedDict(
    [('A', 0), ('R', 1), ('N', 2), ('D', 3),
     ('C', 4), ('Q', 5), ('E', 6), ('G', 7),
     ('H', 8), ('I', 9), ('L', 10),
     ('K', 11), ('M', 12), ('F', 13), ('P', 14),
     ('S', 15), ('T', 16), ('W', 17), ('Y', 18), ('V', 19), ('X', 20)])



# divide by meff instead of count
# bins = [-1, 0.2]
# bins = [0.2, 1]
# bins = [1, 1.8]
# bins = [1.8, 2.5]
# bins = [2.5, 2.8]
# bins = [2.8, 3]
# bins = [3, 3.2]
# bins = [3.2, 3.3]
# bins = [3.3, 3.5]
# bins = [3.5, 4]
# bins = [4, 5.29]

# bins = [-1,3.55]
# bins = [-1,3.21]
# bins = [-1,2.71]

bins = [-1, 1.61]

meff_file = open(meff_file, 'r').readlines()

high_quality_list = []
low_quality_list =[]
for line in meff_file:
    filename = line.split(' ')[0].strip()
    meff = float(line.split(' ')[-1].strip())
    if bins[0]<meff<=bins[1]:
        low_quality_list.append(filename)
    elif meff > bins[1]:
        high_quality_list.append(filename)
