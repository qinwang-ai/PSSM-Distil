import numpy as np
from utils.profile2psm import get_bg_profile, get_profile, get_psm
from configs.DataPathRef90.bc40 import *
test_name = '5tceA'


dict_table = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
pssm_file = '/data/proli/raw_data/visualization/enhanced_pssm/%s.npy' % test_name
# pssm_file = '/data/proli/data/ref90_psm_bc40_real/valid/%s.npy'%test_name
# pssm_file = '/data/proli/raw_data/visualization/low_pssm/%s.npy'%test_name

save_msa_file_prefix = '/data/proli/raw_data/visualization/bc40_msa_from_pssm/'


if __name__ == '__main__':
    filename = pssm_file.split('/')[-1].split('.npy')[0]
    save_msa_file = save_msa_file_prefix + filename + '.a3m'
    original_msa_file = msa_data_path_prefix + filename + '.a3m'

    # inverse
    msa_strs = list(map(lambda x: x.strip(), open(original_msa_file, 'r').readlines()[1::2]))
    # bg_profile = get_bg_profile(msa_strs)
    pssm_array = np.load(pssm_file)
    pssm_array = np.exp(pssm_array)# * bg_profile
    pssm_array = pssm_array - np.expand_dims(pssm_array.min(axis=1), axis=1)
    profile = pssm_array / np.expand_dims(pssm_array.sum(axis=1), axis=1)

    f = open(save_msa_file, 'w')
    msa_list = []
    for i in range(2000):
        ans = ''
        for j in range(profile.shape[0]):
            char = np.random.choice(dict_table, p=profile[j,:])
            ans += char
        msa_list.append(ans)

    for msa in msa_list:
        f.write('>' + filename + '\n')
        f.write(msa+'\n')
    print(save_msa_file, 'saved')
    f.close()








