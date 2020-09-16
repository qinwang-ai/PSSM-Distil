# coding=utf-8

import os
import torch
from torch.utils.data import DataLoader
from DataLoader.SSEDatasetInf import SSEDataset
from models.SSENet import SSENet
from models.Generator import Generator
from tqdm import tqdm
import configs.config_sse as config
import numpy as np
import torch.nn.functional as F

def try_get_pretrained(ssenet, scratch=False):
    ssenet_path = config.pretrain_path + 'ssenet_real_ref90_psm.pth'

    import torch.nn.init as init
    from models.weight_initializer import Initializer
    Initializer.initialize(model=ssenet, initialization=init.xavier_uniform, gain=init.calculate_gain('relu'))

    if not scratch:
        if os.path.exists(ssenet_path):
            ssenet.load_state_dict(torch.load(ssenet_path))

    return ssenet.cuda()


def parse_batch(batch):
    sequence = torch.tensor(batch['sequence'], dtype=torch.int64).cuda()
    label = torch.tensor(batch['label'], dtype=torch.int64).cuda()
    psm = torch.tensor(batch['real_psm'], dtype=torch.float32).cuda()
    filename = batch['filename']
    return sequence, psm, label, filename


def get_mse_loss(sequence, low_psm, real_psm):
    low_psm = low_psm[sequence != 0, :]
    real_psm = real_psm[sequence != 0, :]

    mse_loss = F.mse_loss(low_psm, real_psm)
    return mse_loss


def test_sse(val_loader, ssenet):
    ssenet.eval()
    summary = []
    feature_all = []
    for batch in tqdm(val_loader):
        sequence, psm, label, filename = parse_batch(batch)

        pred, feature = ssenet(sequence, psm)
        pred_no_pad = pred[sequence != 0, :]
        label_no_pad = label[sequence != 0]

        pred_label = torch.argmax(pred_no_pad, dim=-1)
        acc = (pred_label == label_no_pad).sum().float() / pred_label.shape[0]
        summary.append(acc.item())

        feature = feature.squeeze().cpu().detach().numpy()
        feature_all.append(feature)

    feature_all = np.concatenate(feature_all, axis=0)
    # np.save('./logs/ssenet_real.npy', feature_all)

    # statistic
    summary_np = np.array(summary).mean()
    print('[EVAL]', 'curr_acc: %0.3f' % summary_np)


if __name__ == '__main__':
    psm_files = './low_pssms/*.npy'
    sse_dataset = SSEDataset(psm_files,
                             config.psm_fake_data_path_prefix,
                             config.sequence_data_path_prefix,
                             config.label_data_path_prefix,)
    sse_loader = DataLoader(sse_dataset, batch_size=1, num_workers=config.batch_size,
                            collate_fn=sse_dataset.collate_fn, shuffle=False)

    ssenet = SSENet(input_dim=config.embed_dim + config.profile_width)
    generator = Generator(pure_bert=True)

    # try load pretrained model
    ssenet = try_get_pretrained(ssenet, scratch=False)

    test_sse(sse_loader, ssenet)

