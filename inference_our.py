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


def try_get_pretrained(teacher, student, generator, scratch=False):
    teacher_path = '../module/teacher_our_ref908.pth'
    student_path = '../module/student_our_ref908.pth'
    generator_path = '../module/generator_our_ref908.pth'

    student.init_weights()
    teacher.init_weights()
    generator.init_weights()

    if not scratch:
        if os.path.exists(teacher_path):
            teacher.load_state_dict(torch.load(teacher_path))
            print('load teacher')

        if os.path.exists(student_path):
            student.load_state_dict(torch.load(student_path))
            print('load student')

        if os.path.exists(generator_path):
            generator.load_state_dict(torch.load(generator_path))
            print('load generator')
    return teacher.cuda(), student.cuda(), generator.cuda()

def parse_batch(batch):
    sequence = torch.tensor(batch['sequence'], dtype=torch.int64).cuda()
    bert_psm = torch.tensor(batch['fake_psm'], dtype=torch.float32).cuda()
    label = torch.tensor(batch['label'], dtype=torch.int64).cuda()
    low_psm = torch.tensor(batch['real_psm'], dtype=torch.float32).cuda()
    filename = batch['filename']
    return filename, sequence, low_psm, bert_psm, label

def get_acc(sequence, profile, label, net):
    pred, feature = net(sequence, profile)  # 16 x 44 x 3
    pred_no_pad = pred[sequence != 0, :]
    label_no_pad = label[sequence != 0]
    pred_label = torch.argmax(pred_no_pad, dim=-1)
    acc = (pred_label == label_no_pad).sum().float() / pred_label.shape[0]
    return acc, feature

def get_mse_loss(sequence, low_psm, real_psm):
    low_psm = low_psm[sequence != 0, :]
    real_psm = real_psm[sequence != 0, :]

    mse_loss = F.mse_loss(low_psm, real_psm)
    return mse_loss

def save_pssm_file(filename, pssm):
    save_path = '/data/proli/raw_data/visualization/enhanced_pssm/'+ filename + '.npy'
    np.save(save_path, pssm)
    print(save_path, 'saved')


def inference(val_loader, generator, student):

    # validation
    student.eval()
    generator.eval()
    summary = []
    label_all = []
    feature_all = []
    for batch in tqdm(val_loader):
        filename, sequence, low_psm, bert_psm, label = parse_batch(batch)
        profile = torch.cat([bert_psm, low_psm], dim=2)

        enhanced = generator(sequence, profile)
        acc, feature = get_acc(sequence, enhanced, label, student)
        summary.append(acc.item())

        enhanced_np = enhanced.squeeze().cpu().detach().numpy()
        save_pssm_file(filename, enhanced_np)

        feature = feature.squeeze().cpu().detach().numpy()
        label = label.squeeze().cpu().detach().numpy()
        feature_all.append(feature)
        label_all.append(label)

    label_all = np.concatenate(label_all, axis=0)
    feature_all = np.concatenate(feature_all, axis=0)
    np.save('./logs/ssenet_our.npy', feature_all)
    np.save('./logs/label.npy', label_all)



    summary = np.array(summary).mean()
    print('[EVAL]', 'curr_acc: %0.3f' % summary)


if __name__ == '__main__':
    sse_dataset = SSEDataset('/data/proli/raw_data/visualization/low_pssm/*.npy',
                             config.psm_real_data_path_prefix.replace('real', 'fake'),
                             config.sequence_data_path_prefix.replace('train', 'valid'),
                             config.label_data_path_prefix.replace('train', 'valid'))
    sse_loader = DataLoader(sse_dataset, batch_size=1, num_workers=config.batch_size,
                            collate_fn=sse_dataset.collate_fn, shuffle=False)

    teacher = SSENet(input_dim=config.embed_dim + config.profile_width)
    student = SSENet(input_dim=config.embed_dim + config.profile_width)
    generator = Generator()

    # try load pretrained model
    teacher, student, generator = try_get_pretrained(teacher, student, generator, scratch=False)
    inference(sse_loader, generator, student)