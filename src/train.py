import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace
import numpy as np
from model_net import demo_net
from util import llprint
import dill
from sklearn.decomposition import PCA
from outer_models import ddi_rate_score, multi_label_metric
from model_net import demo_net
torch.manual_seed(1203)
import random



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def build_visit_targets(drug_labels, visit_num, num_classes, device):
    gt_container = torch.zeros(visit_num, num_classes, device=device)
    margin_target = torch.full((visit_num, num_classes), -1, dtype=torch.long, device=device)

    for v in range(min(visit_num, drug_labels.size(0))):
        valid_labels = drug_labels[v][drug_labels[v] > 0].long()
        if valid_labels.numel() == 0:
            continue

        gt_container[v, valid_labels] = 1.0
        label_count = min(valid_labels.numel(), num_classes)
        margin_target[v, :label_count] = valid_labels[:label_count]

    gt_container[:, 0] = 0
    return gt_container, margin_target


def main(temp_number=None, epochs=5, lambda_weight=0.02, tau=0.06, alpha=1.0, lambda_schedule='fixed'):
    """
    主函数，用于训练和评估模型
    参数:
        temp_number: 临时编号，默认为None
        epochs: 训练轮数，默认为5
        lambda_weight: 对比损失的权重，默认为0.02
        tau: 对比学习的温度参数，默认为0.06
        alpha: 时间门控的alpha参数，默认为1.0
        lambda_schedule: lambda权重调度策略，默认为'fixed'
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 加载词汇表数据，包含诊断、手术和药物的词汇表
    voc = dill.load(open(r'datas/mimic-iii/voc_final.pkl', 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word) + 1, len(pro_voc.idx2word) + 1, len(med_voc.idx2word) + 1)

    # 加载患者记录数据，并过滤掉长度小于2的记录
    data = dill.load(open(r'datas/mimic-iii/records_final.pkl', 'rb'))
    data = [x for x in data if len(x) >= 2]  # 过滤掉长度小于2的记录


    # 数据预处理：标准化每个访问记录的格式
    for patient_idx, patient in enumerate(data):
        for visit_idx, visit in enumerate(patient):
            # 处理不同长度的访问记录，确保格式统一
            if len(visit) == 4:
                timestamp = visit[3]
                visit[:] = [visit[0], visit[1], visit[2], 0, 0, timestamp]
            elif len(visit) == 5:
                visit.append(visit_idx)
            # 将索引从0开始转换为从1开始
            visit[0] = [i + 1 for i in visit[0]]  # 诊断索引
            visit[1] = [i + 1 for i in visit[1]]  # 手术索引
            visit[2] = [i + 1 for i in visit[2]]  # 药物索引

    # 数据集划分：训练集、测试集和验证集
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]  # 训练集占2/3
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]  # 测试集占1/3中的前半部分
    data_eval = data[split_point + eval_len:]  # 验证集占1/3中的后半部分

    # 加载药物相互作用矩阵和电子健康记录邻接矩阵
    ddi_matrix = dill.load(open(r'datas/mimic-iii/ddi_A_final.pkl', 'rb'))
    #ddi_4_matrix = dill.load(open(r'datas/mimic-iv/ddi_A_final_4.pkl', 'rb'))
    #ddi_matrix = ddi_4_matrix
    ehr_matrix = dill.load(open('datas/mimic-iii/ehr_adj_final.pkl', 'rb'))
    #ehr_4_matrix = dill.load(open(r'datas/mimic-iv/ehr_adj_final_4.pkl', 'rb'))
    #ehr_matrix = ehr_4_matrix

    ddi_matrix = torch.tensor(ddi_matrix, device=device)
    ddi_matrix = torch.concat([torch.zeros_like(ddi_matrix[0]).unsqueeze(dim=-1), ddi_matrix], dim=-1)
    ddi_matrix = torch.concat([torch.zeros_like(ddi_matrix[0]).unsqueeze(dim=0), ddi_matrix], dim=0)
    ehr_matrix = torch.tensor(ehr_matrix, device=device)
    ehr_matrix = torch.concat([torch.zeros_like(ehr_matrix[0]).unsqueeze(dim=-1), ehr_matrix], dim=-1)
    ehr_matrix = torch.concat([torch.zeros_like(ehr_matrix[0]).unsqueeze(dim=0), ehr_matrix], dim=0)

    train_loader = DataLoader(data_train, batch_size=1, collate_fn=pad_batch_v2_train, shuffle=True, pin_memory=False)
    eval_loader = DataLoader(data_eval, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=False)

    model = demo_net(emb_dim=64, voc_size=voc_size, device=device, ddi_graph=ddi_matrix, ehr_graph=ehr_matrix, contrastive_temperature=tau, temporal_gate_alpha=alpha).to(device)

    print('parameters', get_n_params(model))

    optimizer = Adam(model.parameters(), lr=0.0001)
    EPOCH = epochs
    demo_loss_1 = nn.BCELoss()
    demo_loss_2 = nn.MultiLabelMarginLoss()

    if lambda_schedule == 'linear':
        contrastive_weight_scheduler = lambda epoch: min(lambda_weight, max(0.0, (epoch - 5) * (lambda_weight / 10.0)))
    else:
        contrastive_weight_scheduler = lambda epoch: lambda_weight
    print(f'Hyperparameters: epochs={EPOCH}, lambda={lambda_weight}, tau={tau}, alpha={alpha}, lambda_schedule={lambda_schedule}')
    best_f1 = 0.0
    best_epoch = -1
    patience = 10
    epochs_without_improvement = 0
    best_model_path = os.path.join('state_dict', 'best_model.pt')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(EPOCH):
        ddi_rate = 0
        avg_precise = 0
        avg_recall = 0
        avg_f1 = 0
        count = 1e-6
        model.to(device)
        model.train()
        model_train = True
        all_his = True

        total_contrastive_loss = 0
        contrastive_loss_count = 0

        if model_train:
            for index, datas in enumerate(train_loader):
                datas = [i.to(device) for i in datas]
                optimizer.zero_grad(set_to_none=True)

                output = list(model(datas))

                visit_num = output[0].size(0)
                num_classes = output[0].size(-1)
                output_2d = output[0].reshape(visit_num, num_classes)

                contrastive_loss = output[4]
                temporal_prob = output[5]
                total_contrastive_loss += contrastive_loss.item()
                contrastive_loss_count += 1

                drug_labels = datas[2][0]
                gt_container, loss3_target = build_visit_targets(
                    drug_labels, visit_num, num_classes, device
                )

                loss_1 = demo_loss_1(output_2d, gt_container)
                loss_2 = demo_loss_2(output_2d, loss3_target)

                co = 0.02
                contrastive_weight = contrastive_weight_scheduler(epoch)
                total_loss = loss_1 + co * loss_2 + contrastive_weight * contrastive_loss

                total_loss.backward()
                optimizer.step()

                if index % 100 == 0:
                    llprint(f'\r|{index}/{len(train_loader)}| '
                            f'Loss: {total_loss.item():.4f} | '
                            f'BCE: {loss_1.item():.4f} | '
                            f'Margin: {loss_2.item():.4f} | '
                            f'Contrastive: {contrastive_loss.item():.4f}')
                else:
                    llprint('\r|' + \
                            '#' * int(50 * (index / len(train_loader))) + \
                            '-' * int(50 - 50 * (index / len(train_loader))) + \
                            '|{:.2f}%|train_step:{}/{}'.format(100 * (index / len(train_loader)), index,
                                                               len(train_loader))
                            )

            if contrastive_loss_count > 0:
                avg_contrastive_loss = total_contrastive_loss / contrastive_loss_count
                print(f'\nEpoch {epoch} Avg Contrastive Loss: {avg_contrastive_loss:.4f}')

        print()

        model.eval()

        prob_container = []
        gt_container = []
        labels_container = []
        ddi_cnt = 0
        ddi_all_cnt = 0
        avg_med = 0
        eval_contrastive_losses = []

        for index, datas in enumerate(eval_loader):
            datas = [i.to(device) for i in datas]

            with torch.no_grad():
                output = model(datas)

                if isinstance(output, tuple):
                    prob_output = output[0]
                    if len(output) > 4 and torch.is_tensor(output[4]):
                        eval_contrastive_losses.append(output[4].item())
                else:
                    prob_output = output

                visit_num = prob_output.size(0)
                num_classes = prob_output.size(-1)
                output_2d = prob_output.reshape(visit_num, num_classes)

            gt_data = datas[2][0]

            for idx in range(visit_num):
                if idx >= len(gt_data):
                    continue

                vst = output_2d[idx]
                gt_temp = torch.zeros_like(vst, device=device)
                valid_gt = gt_data[idx][gt_data[idx] > 0].long()
                if valid_gt.numel() == 0:
                    continue
                gt_temp[valid_gt] = 1
                gt_temp[0] = 0

                avg_med += vst.sum()
                out_labels = torch.where(vst > 0.35, 1.0, 0.0)
                ddi_temp_container = out_labels * out_labels.T.unsqueeze(dim=-1)
                labels_container.append(out_labels)
                prob_container.append(vst)
                gt_container.append(gt_temp)

                if gt_temp.sum() != 0:
                    precise = (out_labels * gt_temp).sum() / (out_labels.sum() + 1e-9)
                    recall = (out_labels * gt_temp).sum() / (gt_temp.sum() + 1e-9)
                else:
                    continue

                avg_precise += precise
                avg_recall += recall

                if (precise + recall) != 0:
                    f1 = (2.0 * precise * recall) / (precise + recall)
                    avg_f1 += f1

                ddi_cnt += ddi_rate_score(ddi_temp_container, ddi_matrix)[0]
                ddi_all_cnt += ddi_rate_score(ddi_temp_container, ddi_matrix)[1]

                count += 1

            llprint('\r|' + \
                    '@' * int(50 * (index / len(eval_loader))) + \
                    '-' * int(50 - 50 * (index / len(eval_loader))) + \
                    '|{:.2f}%|eval_step:{}/{}'.format(100 * (index / len(eval_loader)), index, len(eval_loader))
                    )

        avg_precise = avg_precise / count if count > 0 else 0
        avg_recall = avg_recall / count if count > 0 else 0
        avg_f1 = avg_f1 / count if count > 0 else 0

        jac, prauc, F_1 = multi_label_metric(gt_container, labels_container, prob_container, voc_size=voc_size)

        try:
            ddi_rate = ddi_cnt / ddi_all_cnt if ddi_all_cnt > 0 else 0
        except:
            ddi_rate = 0

        avg_eval_contrastive = np.mean(eval_contrastive_losses) if eval_contrastive_losses else 0

        print('\navg_prc = {:.4f}'.format(avg_precise),
              'avg_rec = {:.4f}'.format(avg_recall),
              'jac = {:.4f}'.format(jac),
              'prauc = {:.4f}'.format(prauc),
              'avg_f1 = {:.4f}'.format(avg_f1),
              'ddi_rate = {:.4f}'.format(ddi_rate),
              'avg_med = {:.4f}'.format(avg_med / count if count > 0 else 0),
              'eval_contrastive_loss = {:.4f}'.format(avg_eval_contrastive))

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_f1': avg_f1,
                'jac': jac,
                'prauc': prauc,
                'ddi_rate': ddi_rate,
            }, best_model_path)
            print(f'New best model saved: {best_model_path} (epoch={epoch}, avg_f1={avg_f1:.4f}, jac={jac:.4f})')
        else:
            epochs_without_improvement += 1
            print(f'No improvement for {epochs_without_improvement} epoch(s); best avg_f1={best_f1:.4f} at epoch {best_epoch}.')
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch}. Best epoch: {best_epoch}, best avg_f1: {best_f1:.4f}')
                break




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lambda_weight', type=float, default=0.02)
    parser.add_argument('--tau', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--lambda_schedule', choices=['fixed', 'linear'], default='fixed')
    args = parser.parse_args()
    main(temp_number=1, epochs=args.epochs, lambda_weight=args.lambda_weight, tau=args.tau, alpha=args.alpha, lambda_schedule=args.lambda_schedule)


