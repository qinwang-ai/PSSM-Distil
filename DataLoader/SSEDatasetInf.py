from torch.utils.data import Dataset
import numpy as np
import glob
import torch
from configs.config_sse import IUPAC_VOCAB


class SSEDataset(Dataset):
    def __init__(self, real_psm_data_path, fake_psm_data_path_prefix, sequence_data_path_prefix, label_data_path_prefix):
        self.psm_file_list = glob.glob(real_psm_data_path)
        self.sequence_data_path = sequence_data_path_prefix
        self.label_data_path = label_data_path_prefix
        self.tokenizer = Tokenizer()
        self.fake_psm_data_path_prefix = fake_psm_data_path_prefix
        super(SSEDataset, self).__init__()

    def __getitem__(self, index):

        psm_file_real = self.psm_file_list[index]
        filename = psm_file_real.split('/')[-1].split('.npy')[0]
        psm_file_fake = self.fake_psm_data_path_prefix + filename + '.npy'

        sequence_file = self.sequence_data_path + filename + '.fasta'
        label_file = self.label_data_path + filename + '.label'

        # load psm
        real_psm_array = np.load(psm_file_real)
        fake_psm_array = np.load(psm_file_fake)

        sequence_str = open(sequence_file, 'r').readlines()[1].strip()
        token_ids = self.tokenizer.encode(sequence_str)
        label = np.loadtxt(label_file)

        return filename, token_ids, real_psm_array, fake_psm_array, label

    # one epoch 6125 samples
    def __len__(self):
        return len(self.psm_file_list)

    def collate_fn(self, batch):
        # gap model
        filename, input_ids, real_psm_array, fake_psm_array, label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        label = torch.from_numpy(pad_sequences(label, -1))
        real_psm_array = torch.from_numpy(pad_sequences(real_psm_array))
        fake_psm_array = torch.from_numpy(pad_sequences(fake_psm_array))

        return {'filename': filename[0], 'sequence': input_ids, 'real_psm': real_psm_array,
                'label': label, 'fake_psm': fake_psm_array}


class Tokenizer(object):
    def __init__(self):
        super(Tokenizer, self).__init__()
        self.vocab = IUPAC_VOCAB

    def tokenize(self, text: str):
        return [x for x in text]

    def encode(self, text: str):
        tokens = self.tokenize(text)
        token_ids = [self.vocab[token] for token in tokens if token != '\n']
        return np.array(token_ids, np.int64)


def pad_sequences(sequences, constant_value=0, dtype=None):
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array
