import numpy as np
import sys

dict_table = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

save_msa_file_prefix = 'a3ms/'

if __name__ == '__main__':
    pssm_file = sys.argv[1]
    filename = pssm_file.split('/')[-1].split('.npy')[0]
    kind = pssm_file.split('/')[-2]
    save_msa_file = save_msa_file_prefix + filename + '_' + kind + '.a3m'

    # inverse
    pssm_array = np.load(pssm_file)
    pssm_array = np.exp(pssm_array)  # * bg_profile
    pssm_array = pssm_array - np.expand_dims(pssm_array.min(axis=1), axis=1)
    profile = pssm_array / np.expand_dims(pssm_array.sum(axis=1), axis=1)

    f = open(save_msa_file, 'w')
    msa_list = []
    for i in range(2000):
        ans = ''
        for j in range(profile.shape[0]):
            char = np.random.choice(dict_table, p=profile[j, :])
            ans += char
        msa_list.append(ans)

    for msa in msa_list:
        f.write('>' + filename + '\n')
        f.write(msa + '\n')
    print(save_msa_file, 'saved')
    f.close()
