import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from outer_models import RetNetRelPos as RNP
from outer_models import GCN
import random
from outer_models import trd_encoder
torch.manual_seed(1203)
import dgl.function as fn
from torch.nn import Parameter
import math
from typing import List


# ========== 时间感知组件 ==========
class Time2VecTimeEncoder(nn.Module):
    def __init__(self, embedding_dimension: int):
        """
        Time2Vec 时间编码器。
        第 1 维学习线性时间趋势，其余维度学习周期性时间变化。
        """
        super(Time2VecTimeEncoder, self).__init__()
        if embedding_dimension < 2:
            raise ValueError("embedding_dimension must be at least 2 for Time2Vec.")
        self.embedding_dimension = embedding_dimension
        self.scale_factor = (1 / embedding_dimension) ** 0.5
        self.linear_w = nn.Parameter(torch.randn(1, 1))
        self.linear_b = nn.Parameter(torch.randn(1, 1))
        self.periodic_w = nn.Parameter(torch.randn(1, embedding_dimension - 1))
        self.periodic_b = nn.Parameter(torch.randn(1, embedding_dimension - 1))

    def forward(self, input_relative_time: torch.Tensor):
        linear_encoding = torch.matmul(input_relative_time, self.linear_w) + self.linear_b
        periodic_encoding = torch.sin(torch.matmul(input_relative_time, self.periodic_w) + self.periodic_b)
        return self.scale_factor * torch.cat([linear_encoding, periodic_encoding], dim=-1)



class GraphConvolution(torch.nn.Module):
    """
    图卷积层 (Graph Convolutional Layer) 实现
    实现简单的图卷积层，支持带有可选偏置项的线性变换，
    并使用邻接矩阵进行图卷积操作。
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        初始化图卷积层
        参数:
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            bias (bool): 是否使用偏置项，默认为 True
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features  # 保存输入特征维度
        self.out_features = out_features  # 保存输出特征维度

        assert isinstance(in_features, int), f"in_features must be int, got {type(in_features)}"
        assert isinstance(out_features, int), f"out_features must be int, got {type(out_features)}"

        # 创建可训练的权重参数矩阵
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            # 如果使用偏置，则创建偏置参数
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            # 如果不使用偏置，则注册一个 None 参数
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        """
        重置网络参数
        使用均匀分布初始化权重和偏置（如果存在）
        初始化范围为 [-1/√n, 1/√n]，其中 n 是输出特征的数量
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # 初始化权重
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  # 初始化偏置

    def forward(self, input, adj):
        """
        前向传播函数
        参数:
            input (Tensor): 输入特征矩阵，形状为 [num_nodes, in_features]
            adj (Tensor): 邻接矩阵，形状为 [num_nodes, num_nodes]
        返回:
            Tensor: 输出特征矩阵，形状为 [num_nodes, out_features]
        """
        # 计算输入特征与权重矩阵的乘积
        support = torch.mm(input, self.weight)
        # 使用邻接矩阵对特征进行图卷积操作
        output = torch.mm(adj, support)
        if self.bias is not None:
            # 如果存在偏置，则添加偏置项
            return output + self.bias
        else:
            # 如果不存在偏置，则直接返回输出
            return output


class GCN(torch.nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        """
        初始化图神经网络模型
        参数:
            voc_size: 词汇表大小，包含诊断、手术和药物的数量
            emb_dim: 嵌入维度
            ehr_adj: 电子健康记录的邻接矩阵
            ddi_adj: 药物相互作用的邻接矩阵
            device: 计算设备，默认为 CPU
        """
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        # 对邻接矩阵添加自环，再进行归一化处理
        # 训练入口传入 Tensor（可能位于 CUDA）；NumPy 归一化前显式移到 CPU。
        if torch.is_tensor(ehr_adj):
            ehr_adj = ehr_adj.detach().cpu().numpy()
        if torch.is_tensor(ddi_adj):
            ddi_adj = ddi_adj.detach().cpu().numpy()
        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        # 将邻接矩阵转换为张量并移动到指定设备
        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # 使用药物数量作为输入维度
        med_voc_size = voc_size[2]
        # 创建单位矩阵作为药物嵌入的初始输入
        self.x = torch.eye(med_voc_size).to(device)

        # 定义三个图卷积层
        self.gcn1 = GraphConvolution(med_voc_size, emb_dim)
        self.dropout = torch.nn.Dropout(p=0.3)  # 定义 dropout 层，防止过拟合
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        """
        前向传播过程
        返回:
            ehr_node_embedding: EHR 图中的节点嵌入
            ddi_node_embedding: DDI 图中的节点嵌入
        """
        # 第一层图卷积处理
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
        # 应用 ReLU 激活函数
        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        # 应用dropout
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        # 分别使用第二个和第三个图卷积层处理 EHR 图与 DDI 图
        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """
        对邻接矩阵进行归一化处理
        参数:
            mx: 需要归一化的邻接矩阵
        返回:
            归一化后的邻接矩阵
        """
        # 计算每行的和
        rowsum = np.array(mx.sum(1))
        # 计算行和的倒数，并处理无穷大的情况
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        # 创建对角矩阵
        r_mat_inv = np.diagflat(r_inv)
        # 应用归一化
        mx = r_mat_inv.dot(mx)
        return mx

class SimpleContrastiveLearningModule(nn.Module):
    """Contrastive learning module."""

    def __init__(self, input_dim, temperature=0.06):
        super().__init__()
        self.input_dim = input_dim #输入特征维度
        self.temperature = temperature # 温度参数，控制相似度的平滑程度

        # 改进的投影头：添加LayerNorm和更深的网络
        self.projection_head = nn.Sequential(
            # 第一层：扩展到两倍维度
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2), # 层归一化，稳定训练
            nn.ReLU(), # 非线性激活
            nn.Dropout(0.1), # 防止过拟合
            # 第二层：降回原始维度
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim), # 再次归一化
            nn.ReLU(),
            # 第三层：最终投影到对比空间
            nn.Linear(input_dim, input_dim)
        )

        # 负样本队列（简单实现）
        self.register_buffer("neg_queue", torch.randn(1024, input_dim))
        self.queue_ptr = 0 # 队列指针，用于循环写入

    @torch.no_grad()
    def update_queue(self, features):
        """Update negative sample queue."""
        batch_size = features.size(0)
        ptr = self.queue_ptr
        # 队列剩余空间足够
        if ptr + batch_size <= 1024:
            self.neg_queue[ptr:ptr + batch_size] = features.detach() #直接写入
            self.queue_ptr = (ptr + batch_size) % 1024 #更新指针
        else:
            # 循环覆盖
            self.neg_queue[ptr:] = features.detach()[:1024 - ptr]
            self.neg_queue[:batch_size - (1024 - ptr)] = features.detach()[1024 - ptr:]
            self.queue_ptr = batch_size - (1024 - ptr)

    def forward(self, anchor_emb, positive_emb, negative_emb=None, negative_mask=None):
        batch_size = anchor_emb.size(0)

        # 投影到对比空间
        anchor_proj = self.projection_head(anchor_emb)
        positive_proj = self.projection_head(positive_emb)

        # L2 归一化
        anchor_proj = F.normalize(anchor_proj, dim=-1)
        positive_proj = F.normalize(positive_proj, dim=-1)

        # 正样本相似度
        pos_sim = torch.sum(anchor_proj * positive_proj, dim=-1) / self.temperature

        if negative_emb is not None:
            # 使用提供的负样本
            if negative_emb.dim() == 3:
                negative_proj = self.projection_head(negative_emb.view(-1, self.input_dim))
                negative_proj = F.normalize(negative_proj, dim=-1)
                negative_proj = negative_proj.view(batch_size, -1, self.input_dim)
                neg_sim = torch.bmm(anchor_proj.unsqueeze(1),
                                    negative_proj.transpose(1, 2)).squeeze(1) / self.temperature
                if negative_mask is not None:
                    neg_sim = neg_sim.masked_fill(~negative_mask.bool(), -1e9)
            else:
                negative_proj = self.projection_head(negative_emb)
                negative_proj = F.normalize(negative_proj, dim=-1)
                neg_sim = torch.mm(anchor_proj, negative_proj.T) / self.temperature
                if negative_mask is not None:
                    neg_sim = neg_sim.masked_fill(~negative_mask.bool(), -1e9)
        else:
            # 使用队列中的负样本
            queue_norm = F.normalize(self.neg_queue, dim=-1)
            neg_sim = torch.mm(anchor_proj, queue_norm.T) / self.temperature

        # 组合logits
        if negative_emb is not None:
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        else:
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # 更新队列：保存历史用药检索表示，作为后续负样本
        self.update_queue(positive_proj)

        # 限制 loss 的最大值
        loss = torch.clamp(loss, max=10.0)

        return loss

class SimpleHistoryDrugRetrieval(nn.Module):
    """History drug retrieval module."""

    def __init__(self, voc_size, emb_dim, med_dim, device='cpu', temperature=0.06):
        super().__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.med_dim = med_dim #医疗特征维度
        self.device = device

        # 药物记忆库（缩放初始权重）
        self.drug_memory = nn.Parameter(torch.randn(voc_size[2] - 1, emb_dim, device=device) * 0.02)

        # 改进的患者表示投影
        self.patient_projection = nn.Sequential(
            nn.Linear(med_dim, emb_dim * 2, device=device),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim * 2, emb_dim, device=device),
            nn.LayerNorm(emb_dim)
        )

        # 位置编码，为历史序列中的每个位置添加位置信息
        self.position_encoder = nn.Embedding(100, emb_dim, device=device)

        # 双向 GRU 编码器：捕捉历史序列的上下文信息
        self.history_encoder = nn.GRU(emb_dim, emb_dim // 2, batch_first=True,
                                      bidirectional=True, device=device, dropout=0.1)
        self.history_proj = nn.Linear(emb_dim, emb_dim, device=device)

        # 多头自注意力：捕捉历史序列内部的依赖关系
        self.self_attention = nn.MultiheadAttention(emb_dim, num_heads=4,
                                                    batch_first=True, device=device, dropout=0.1)

        # 交叉注意力：将当前患者表示与历史序列进行交互
        self.cross_attention = nn.MultiheadAttention(emb_dim, num_heads=4,
                                                     batch_first=True, device=device, dropout=0.1)

        # 门控融合：动态决定当前表示和历史摘要的权重
        self.fusion_gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim, device=device), # 输入：拼接当前表示和历史摘要
            nn.Sigmoid()
        )

        # 对比学习模块，拉近相关表示，推远不相关表示
        self.contrastive_module = SimpleContrastiveLearningModule(emb_dim, temperature=temperature)

        # 可学习权重，用于调节损失和表示融合的贡献
        self.contrastive_weight = nn.Parameter(torch.tensor(0.1, device=device))
        self.fusion_weight = nn.Parameter(torch.tensor(0.5, device=device))
        self.num_hard_negatives = 3
        self.negative_jaccard_threshold = 0.3



    def build_dissimilar_history_negatives(self, current_proj, history_seq, history_drug_masks,
                                           history_drug_tensor=None, current_drug_tensor=None,
                                           cross_attention_weights=None):
        batch_size, seq_len, _ = history_seq.shape
        if seq_len == 0:
            return None, None

        valid_mask = history_drug_masks.bool()
        normalized_current = F.normalize(current_proj, dim=-1)
        normalized_history = F.normalize(history_seq, dim=-1)
        similarity = torch.bmm(
            normalized_history, normalized_current.unsqueeze(-1)
        ).squeeze(-1)

        if history_drug_tensor is not None and current_drug_tensor is not None:
            history_sets = history_drug_tensor.bool()
            current_sets = current_drug_tensor.bool().unsqueeze(1)
            intersection = (history_sets & current_sets).sum(dim=-1).to(torch.float32)
            union = (history_sets | current_sets).sum(dim=-1).to(torch.float32).clamp(min=1.0)
            jaccard = intersection / union
            low_jaccard_mask = (jaccard < self.negative_jaccard_threshold) & valid_mask
        else:
            low_jaccard_mask = valid_mask

        candidate_mask = low_jaccard_mask
        has_candidate = candidate_mask.any(dim=1, keepdim=True)
        candidate_mask = torch.where(has_candidate, candidate_mask, valid_mask)

        max_negatives = min(self.num_hard_negatives, seq_len)
        similarity = similarity.masked_fill(~candidate_mask, float("-inf"))
        neg_indices = torch.topk(
            similarity, k=max_negatives, dim=-1, largest=True
        ).indices

        expanded_indices = neg_indices.unsqueeze(-1).expand(-1, -1, history_seq.size(-1))
        negative_emb = torch.gather(history_seq, 1, expanded_indices)
        negative_mask = torch.gather(candidate_mask, 1, neg_indices)
        return negative_emb, negative_mask

    def encode_current_drugs_for_contrastive(self, current_drug_tensor):
        if current_drug_tensor is None:
            return None

        current_drug_tensor = current_drug_tensor.to(self.drug_memory.dtype)
        drug_counts = current_drug_tensor.sum(dim=-1, keepdim=True).clamp(min=1.0)
        current_drug_emb = torch.matmul(current_drug_tensor, self.drug_memory) / drug_counts
        return self.history_proj(current_drug_emb)

    def encode_history_v2(self, history_drug_tensor, history_drug_masks=None):
        batch_size, seq_len, _ = history_drug_tensor.shape
        if history_drug_masks is None:
            history_drug_masks = history_drug_tensor.sum(dim=-1) > 0
        history_drug_masks = history_drug_masks.bool()
        valid_history = history_drug_masks.any(dim=1)

        if seq_len == 0:
            return (
                torch.zeros(batch_size, 1, self.emb_dim, device=self.device),
                torch.zeros(batch_size, self.emb_dim, device=self.device),
                None,
                valid_history,
            )

        drug_embeddings = torch.matmul(history_drug_tensor, self.drug_memory)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_encoder(positions)
        drug_embeddings = drug_embeddings + pos_emb * 0.2
        drug_embeddings = drug_embeddings * history_drug_masks.unsqueeze(-1).to(drug_embeddings.dtype)

        encoded_seq = torch.zeros(batch_size, seq_len, self.emb_dim, device=self.device)
        attended_seq = torch.zeros_like(encoded_seq)
        attn_weights = None
        has_valid_history = bool(valid_history.any().item())

        if has_valid_history:
            valid_embeddings = drug_embeddings[valid_history]
            valid_encoded_seq, _ = self.history_encoder(valid_embeddings)
            valid_encoded_seq = self.history_proj(valid_encoded_seq)
            encoded_seq[valid_history] = valid_encoded_seq

            valid_attended_seq, attn_weights = self.self_attention(
                valid_encoded_seq,
                valid_encoded_seq,
                valid_encoded_seq,
                key_padding_mask=~history_drug_masks[valid_history]
            )
            attended_seq[valid_history] = valid_attended_seq

        attention_scores = attended_seq.mean(dim=-1)
        attention_scores = attention_scores.masked_fill(~history_drug_masks, -1e9)
        attention_weights = torch.zeros_like(attention_scores)
        if has_valid_history:
            attention_weights[valid_history] = F.softmax(attention_scores[valid_history], dim=-1)
        history_summary = torch.bmm(attention_weights.unsqueeze(1), attended_seq).squeeze(1)

        return attended_seq, history_summary, attn_weights, valid_history

    def retrieve_with_contrastive_v2(self, current_rep, history_drug_tensor, history_drug_masks=None,
                                     current_drug_tensor=None):
        if current_rep.dim() == 3:
            current_rep = current_rep.squeeze(1) if current_rep.size(1) == 1 else current_rep.mean(dim=1)

        current_proj = self.patient_projection(current_rep)
        history_seq, history_summary, attn_weights, valid_history = self.encode_history_v2(
            history_drug_tensor, history_drug_masks
        )
        history_summary = torch.where(valid_history.unsqueeze(-1), history_summary, current_proj)
        has_valid_history = bool(valid_history.any().item())

        cross_summary = current_proj
        cross_weights = None
        if has_valid_history and history_seq.size(1) > 0:
            cross_attended, cross_weights = self.cross_attention(
                current_proj[valid_history].unsqueeze(1),
                history_seq[valid_history],
                history_seq[valid_history],
                key_padding_mask=~history_drug_masks[valid_history]
            )
            cross_summary = cross_summary.clone()
            cross_summary[valid_history] = cross_attended.squeeze(1)

        gate_input = torch.cat([current_proj, history_summary], dim=-1)
        gate_value = self.fusion_gate(gate_input)
        fused_rep = gate_value * current_proj + (1 - gate_value) * history_summary
        fused_rep = fused_rep + 0.3 * cross_summary

        if has_valid_history:
            positive_summary = self.encode_current_drugs_for_contrastive(
                current_drug_tensor[valid_history] if current_drug_tensor is not None else None
            )
            if positive_summary is None:
                positive_summary = cross_summary[valid_history]
            else:
                positive_summary = positive_summary.detach()

            negative_emb, negative_mask = self.build_dissimilar_history_negatives(
                current_proj[valid_history],
                history_seq[valid_history],
                history_drug_masks[valid_history],
                history_drug_tensor=history_drug_tensor[valid_history],
                current_drug_tensor=current_drug_tensor[valid_history] if current_drug_tensor is not None else None,
                cross_attention_weights=cross_weights
            )
            has_explicit_negatives = negative_mask is not None and bool(negative_mask.any().item())
            contrastive_loss = self.contrastive_module(
                current_proj[valid_history],
                positive_summary,
                negative_emb=negative_emb if has_explicit_negatives else None,
                negative_mask=negative_mask if has_explicit_negatives else None
            )
            consistency_loss = F.mse_loss(fused_rep[valid_history], cross_summary[valid_history])
        else:
            contrastive_loss = current_proj.new_zeros(())
            consistency_loss = current_proj.new_zeros(())

        total_loss = contrastive_loss + 0.1 * consistency_loss
        return fused_rep.unsqueeze(1), attn_weights, total_loss


# =====================================================================


class demo_net(nn.Module):
    """Main TPA-CR model."""

    def __init__(self, voc_size, emb_dim=64, nhead=2, device='cuda',
                 mpnn_mole=None, ddi_graph=None, ehr_graph=None,
                 contrastive_temperature=0.06, temporal_gate_alpha=1.0):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.med_dim = emb_dim * 2
        self.voc_size = voc_size
        self.nhead = nhead
        self.temporal_gate_alpha = temporal_gate_alpha

        if ehr_graph is not None and ddi_graph is not None:
            self.gcn = GCN(voc_size, emb_dim, ehr_graph, ddi_graph, device)
            self.use_gcn = True

            # beta = sigmoid(logit)，初值 0.5；在表示层学习 DDI 分支的扣减强度。
            # 这是关系表示融合，并不是对最终 DDI rate 的显式约束。
            self.ddi_fusion_logit = nn.Parameter(torch.zeros((), device=device))

            # 将融合后的双图表示投影到与 patient_rep 相同的维度。
            self.gcn_projection = nn.Sequential(
                nn.Linear(emb_dim, self.med_dim, device=device),
                nn.LayerNorm(self.med_dim),
                nn.ReLU(),
            )
            print(f"GCN 已初始化，投影层维度: {emb_dim} -> {self.med_dim}")
        else:
            self.use_gcn = False
            self.gcn_projection = None
            print("GCN disabled")

        self.time2vec_encoder = Time2VecTimeEncoder(embedding_dimension=emb_dim)
        self.temporal_attention_dropout = nn.Dropout(p=0.2)
        self.temporal_information_importance = nn.Parameter(torch.FloatTensor([0.5]).to(device))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1)
        )

        # 可学习的时间衰减参数
        self.time_decay_tau = nn.Parameter(torch.tensor(10.0, device=device))

        self.rnp = RNP(emb_dim=emb_dim * 4)
        self.diag_emb = nn.Embedding(voc_size[0], emb_dim, padding_idx=0, device=device)
        self.proc_emb = nn.Embedding(voc_size[1], emb_dim, padding_idx=0, device=device)
        self.med_embedding = nn.Embedding(voc_size[2], emb_dim, padding_idx=0, device=device)

        self.dropout = nn.Dropout(p=0.2)

        self.diag_linear_1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2, device=device),
            nn.Tanh(),
            nn.Linear(emb_dim * 2, emb_dim, device=device),
            nn.Dropout(0.3)
        )
        self.proc_linear_1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2, device=device),
            nn.Tanh(),
            nn.Linear(emb_dim * 2, emb_dim, device=device),
            nn.Dropout(0.3)
        )
        self.diag_linear_2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2, device=device),
            nn.Tanh(),
            nn.Linear(emb_dim * 2, int(self.med_dim / 2), device=device),
            nn.Dropout(0.3)
        )
        self.proc_linear_2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2, device=device),
            nn.Tanh(),
            nn.Linear(emb_dim * 2, int(self.med_dim / 2), device=device),
            nn.Dropout(0.3)
        )
        self.final_linear = nn.Sequential(
            nn.Linear(self.med_dim, self.med_dim * 4, device=device),
            nn.Tanh(),
            nn.Linear(self.med_dim * 4, self.med_dim * 4, device=device),
            nn.Tanh(),
            nn.Linear(self.med_dim * 4, self.med_dim, device=device),
            nn.Dropout(0.3)
        )

        self.med_block = nn.Parameter(torch.randn([self.med_dim, voc_size[2] - 1], device=device))
        self.his_seq_med = nn.Parameter(torch.randn([self.med_dim, voc_size[2] - 1], device=device))
        self.diag_med_block = nn.Parameter(torch.randn([self.emb_dim, voc_size[2] - 1], device=device))
        self.proc_med_block = nn.Parameter(torch.randn([self.emb_dim, voc_size[2] - 1], device=device))
        self.his_seq_med_block = nn.Parameter(torch.randn([self.med_dim, voc_size[2] - 1], device=device))

        self.diag_encoder = nn.TransformerEncoderLayer(emb_dim, nhead, batch_first=True,
                                                       dropout=0.2, device=device)
        self.proc_encoder = nn.TransformerEncoderLayer(emb_dim, nhead, batch_first=True,
                                                       dropout=0.2, device=device)

        self.diag_integ = trd_encoder(emb_dim=int(self.med_dim / 2), device=device)
        self.proc_integ = trd_encoder(emb_dim=int(self.med_dim / 2), device=device)
        self.diag_prob_integ = trd_encoder(emb_dim=int(voc_size[2] - 1), device=device)
        self.proc_prob_integ = trd_encoder(emb_dim=int(voc_size[2] - 1), device=device)
        self.drug_mem_integ = trd_encoder(emb_dim=int(voc_size[2] - 1), device=device)

        self.diag_his_encoder = nn.GRU(int(self.med_dim / 2), int(self.med_dim / 2),
                                       batch_first=False, device=device, dropout=0.2)
        self.proc_his_encoder = nn.GRU(int(self.med_dim / 2), int(self.med_dim / 2),
                                       batch_first=False, device=device, dropout=0.2)
        self.temporal_gru = nn.GRU(emb_dim * 2, emb_dim, batch_first=False, device=device, dropout=0.2)

        self.patient_linear = nn.Sequential(
            nn.Linear(self.med_dim, self.med_dim, device=device),
            nn.Tanh()
        )
        self.patient_mem_contact = nn.Sequential(
            nn.Linear(self.med_dim * 2, self.med_dim * 4, device=device),
            nn.Tanh(),
            nn.Linear(self.med_dim * 4, voc_size[2] - 1, device=device),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        self.his_seq_contact = nn.Sequential(
            nn.Linear(self.med_dim * 2, self.med_dim * 4, device=device),
            nn.Tanh(),
            nn.Linear(self.med_dim * 4, voc_size[2] - 1, device=device),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        self.prob_seq_contact = nn.Sequential(
            nn.Linear((voc_size[2] - 1) * 4, (voc_size[2] - 1) * 8, device=device),
            nn.Tanh(),
            nn.Linear((voc_size[2] - 1) * 8, (voc_size[2] - 1), device=device),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        self.gender_block = nn.Parameter(torch.eye(voc_size[2] - 1, device=device, requires_grad=True))
        self.age_block = nn.Parameter(torch.randn(self.med_dim, voc_size[2], device=device, requires_grad=True))

        self.temporal_proj = nn.Linear(emb_dim, voc_size[2] - 1, device=device)

        self.history_retrieval = SimpleHistoryDrugRetrieval(voc_size, emb_dim, self.med_dim, device, temperature=contrastive_temperature)
        self.retrieved_projection = nn.Sequential(
            nn.Linear(emb_dim, self.med_dim, device=device),
            nn.LayerNorm(self.med_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.med_dim, self.med_dim, device=device)
        )


        self.time_gate_generator = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 3),  # 为三条路径生成门控：1. GCN，2. 历史记忆，3. 对比检索
            nn.Sigmoid()
        )

        # 2. 时间增强的患者表示融合
        self.temporal_patient_fusion = nn.Sequential(
            nn.Linear(self.med_dim * 2, self.med_dim),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

    def graph_recommendation_scores(self, patient_rep_temporal, gcn_gate):
        """Fuse EHR/DDI embeddings, match patients, then gate non-padding scores."""
        if not self.use_gcn or self.gcn_projection is None:
            return patient_rep_temporal.new_zeros(
                patient_rep_temporal.size(0), self.voc_size[2] - 1
            )
        ehr_node_emb, ddi_node_emb = self.gcn()
        beta = torch.sigmoid(self.ddi_fusion_logit)
        drug_node_emb = ehr_node_emb - beta * ddi_node_emb
        drug_node_emb_proj = self.gcn_projection(drug_node_emb)
        scores = patient_rep_temporal @ drug_node_emb_proj.T
        return scores[:, 1:] * gcn_gate

    def history_gate_unit(self, patient_rep, all_vst_drug, contacter, his_fuser=None):
        """History gate unit."""
        his_seq_mem = patient_rep[:-1]
        his_seq_mem = torch.cat([torch.zeros_like(patient_rep[0]).unsqueeze(dim=0), his_seq_mem], dim=0)

        his_seq_container = torch.zeros([patient_rep.size()[0],
                                         patient_rep.size()[0],
                                         patient_rep.size()[1] * 2],
                                        device=self.device)
        for i in range(len(patient_rep)):
            for j in range(len(his_seq_mem)):
                if j <= i:
                    his_seq_container[i, j] = torch.concat([patient_rep[i], his_seq_mem[j]], dim=-1)

        his_seq_container = contacter(his_seq_container)
        his_seq_filter_mask = torch.tril(torch.ones([patient_rep.size()[0],
                                                     patient_rep.size()[0]], device=self.device)).unsqueeze(dim=-1)
        his_seq_enhance = his_seq_filter_mask * his_seq_container * all_vst_drug
        his_seq_enhance = his_seq_enhance.sum(dim=1)

        return his_seq_enhance.reshape(-1, self.voc_size[2] - 1)

    def temporal_attention_aggregation(self, query_embedding, key_embeddings, time_tensor):
        """Temporal attention aggregation."""
        current_time = time_tensor[-1]
        time_diff = current_time - time_tensor
        time_diff[-1] = 0

        time_features = time_diff.unsqueeze(-1).unsqueeze(-1)
        time_embeddings = self.time2vec_encoder(time_features).squeeze(1)

        content_attention = torch.matmul(query_embedding, key_embeddings.T)

        time_score = self.time_mlp(time_embeddings).squeeze(-1)
        time_score = time_score.unsqueeze(0)

        time_decay_tau = F.softplus(self.time_decay_tau) + 1e-6
        time_decay = torch.exp(-time_diff / time_decay_tau)
        time_decay = time_decay.unsqueeze(0)

        combined_attention = content_attention + self.temporal_information_importance * time_score
        combined_attention = combined_attention * time_decay

        attention_weights = torch.softmax(combined_attention, dim=-1)
        aggregated = torch.matmul(attention_weights, key_embeddings)

        return aggregated

    def encoder(self, diag_seq, proc_seq, diag_mask=None, proc_mask=None):
        """Encode diagnosis and procedure sequence."""
        max_diag_num = diag_seq.size()[-1]
        max_proc_num = proc_seq.size()[-1]
        max_visit_num = diag_seq.size()[1]
        batch_size = diag_seq.size()[0]

        diag_seq = self.diag_linear_1(self.diag_emb(diag_seq).view(batch_size * max_visit_num,
                                                                   max_diag_num, self.emb_dim))
        proc_seq = self.proc_linear_1(self.proc_emb(proc_seq).view(batch_size * max_visit_num,
                                                                   max_proc_num, self.emb_dim))

        if diag_mask is not None:
            d_mask_matrix = diag_mask.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).unsqueeze(
                dim=1).repeat(1, self.nhead, max_diag_num, 1)
            d_mask_matrix = d_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_diag_num, max_diag_num)
        else:
            d_mask_matrix = None

        if proc_mask is not None:
            p_mask_matrix = proc_mask.view(batch_size * max_visit_num, max_proc_num).unsqueeze(dim=1).unsqueeze(
                dim=1).repeat(1, self.nhead, max_proc_num, 1)
            p_mask_matrix = p_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_proc_num, max_proc_num)
        else:
            p_mask_matrix = None

        diag_seq = self.diag_encoder(diag_seq, src_mask=d_mask_matrix).view(-1, max_diag_num, self.emb_dim)
        proc_seq = self.proc_encoder(proc_seq, src_mask=p_mask_matrix).view(-1, max_proc_num, self.emb_dim)

        diag_rep_1, diag_rep_2 = self.diag_integ(diag_seq)
        diag_rep = diag_rep_1 + diag_rep_2
        proc_rep_1, proc_rep_2 = self.proc_integ(proc_seq)
        proc_rep = proc_rep_1 + proc_rep_2

        patient_rep = torch.concat([diag_rep, proc_rep], dim=-1)
        patient_rep = self.final_linear(patient_rep)

        diag_seq = self.diag_linear_2(diag_seq)
        proc_seq = self.proc_linear_2(proc_seq)

        return diag_seq, proc_seq, patient_rep

    def decoder(self, diag, proc, ages, gender=None, drug_mem=None, patient_rep=None, time_list=None):
        """Temporal multi-path decoder."""
        visit_num = patient_rep.size(0)

        # 处理药物记忆
        visit_drug_onehot = torch.nn.functional.one_hot(
            drug_mem.squeeze(dim=0),
            num_classes=self.voc_size[2]
        ).sum(dim=-2)[:, 1:].to(torch.float32)

        shifted_drug_mem = torch.cat(
            [torch.zeros_like(visit_drug_onehot[:1]), visit_drug_onehot[:-1]], dim=0
        )[:visit_num]
        causal_mask = torch.tril(torch.ones(visit_num, visit_num, device=self.device))
        drug_mem_onehot = shifted_drug_mem.unsqueeze(0).repeat(visit_num, 1, 1).to(self.device)
        drug_mem_onehot = drug_mem_onehot * causal_mask.unsqueeze(-1)
        history_drug_masks = drug_mem_onehot.sum(dim=-1) > 0

        # 历史序列编码
        diag_his = diag.sum(dim=-2).reshape(-1, self.emb_dim)
        proc_his = proc.sum(dim=-2).reshape(-1, self.emb_dim)
        diag_his = self.diag_his_encoder(diag_his)[0]
        proc_his = self.proc_his_encoder(proc_his)[0]
        his_seq_rep = torch.concat([diag_his, proc_his], dim=-1)

        if patient_rep.dim() == 3:
            if patient_rep.size(1) == 1:
                patient_rep = patient_rep.squeeze(1)
            else:
                patient_rep = patient_rep.mean(dim=1)

        # 时间感知的多路径调制
        temporal_prob = torch.zeros(visit_num, self.voc_size[2] - 1, device=self.device)
        temporal_prob_list = []
        temporal_gate_weights_list = []
        temporal_agg_features_list = []

        if time_list is not None and len(time_list) > 0 and visit_num > 0:
            if isinstance(time_list, torch.Tensor):
                time_tensor = time_list.to(self.device)
            else:
                time_tensor = torch.FloatTensor(time_list).to(self.device)
            time_tensor = time_tensor.squeeze()[:visit_num]

            patient_seq = self.patient_linear(patient_rep)[:, :self.emb_dim]

            for i in range(visit_num):
                query = patient_seq[i].unsqueeze(0)
                keys = patient_seq[:i + 1]
                times = time_tensor[:i + 1]

                agg = self.temporal_attention_aggregation(query, keys, times)
                prob_i = self.temporal_proj(agg)
                temporal_prob_list.append(prob_i)

                gate_weights = self.time_gate_generator(agg)
                temporal_gate_weights_list.append(gate_weights)
                temporal_agg_features_list.append(agg)

            temporal_prob = torch.cat(temporal_prob_list, dim=0)
            temporal_gate_weights = torch.cat(temporal_gate_weights_list, dim=0)
            temporal_agg_features = torch.cat(temporal_agg_features_list, dim=0)

            patient_rep_temporal = self.temporal_patient_fusion(
                torch.cat([patient_rep, temporal_agg_features.repeat(1, 2)[:, :self.med_dim]], dim=-1)
            )
        else:
            temporal_gate_weights = torch.ones(visit_num, 3, device=self.device)
            patient_rep_temporal = patient_rep

        # 拆分门控权重
        gcn_gate_raw = temporal_gate_weights[:, 0].unsqueeze(-1)
        his_gate_raw = temporal_gate_weights[:, 1].unsqueeze(-1)
        retrieval_gate_raw = temporal_gate_weights[:, 2].unsqueeze(-1)
        alpha = self.temporal_gate_alpha
        gcn_gate = alpha * gcn_gate_raw + (1 - alpha) * torch.ones_like(gcn_gate_raw)
        his_gate = alpha * his_gate_raw + (1 - alpha) * torch.ones_like(his_gate_raw)
        retrieval_gate = alpha * retrieval_gate_raw + (1 - alpha) * torch.ones_like(retrieval_gate_raw)

        # 路径 1：时间增强的图卷积预测
        gcn_enhanced_prob = self.graph_recommendation_scores(
            patient_rep_temporal, gcn_gate
        )

        # 路径 2：时间调制的历史记忆预测
        his_enhance = self.history_gate_unit(
            patient_rep_temporal, drug_mem_onehot, self.patient_mem_contact
        )
        his_enhance = his_enhance * his_gate  # 施加时间门控

        # 路径 3：时间调制的对比检索预测
        history_drug_masks = history_drug_masks.bool()
        retrieved_rep, attention_weights, contrastive_loss = \
            self.history_retrieval.retrieve_with_contrastive_v2(
                patient_rep_temporal, drug_mem_onehot, history_drug_masks=history_drug_masks,
                current_drug_tensor=visit_drug_onehot[:visit_num].to(self.device)
            )
        retrieved_rep_mean = retrieved_rep.mean(dim=1)
        retrieved_rep_proj = self.retrieved_projection(retrieved_rep_mean)
        retrieved_prob = torch.matmul(retrieved_rep_proj, self.med_block)
        retrieved_prob = retrieved_prob * retrieval_gate  # 施加时间门控

        final_prob_1 = patient_rep_temporal @ self.med_block  # 同样使用时间增强表示

        diag_probseq = diag @ self.diag_med_block
        proc_probseq = proc @ self.proc_med_block

        diag_prob_1, diag_prob_2 = self.diag_prob_integ(diag_probseq)
        diag_prob = (diag_prob_1 + diag_prob_2).reshape(-1, self.voc_size[2] - 1)

        proc_prob_1, proc_prob_2 = self.proc_prob_integ(proc_probseq)
        proc_prob = (proc_prob_1 + proc_prob_2).reshape(-1, self.voc_size[2] - 1)

        final_prob_3 = diag_prob + proc_prob

        prob_seq_rep = torch.concat([diag_prob, proc_prob], dim=-1)
        prob_seq_enhance = self.history_gate_unit(
            prob_seq_rep, drug_mem_onehot, self.prob_seq_contact
        )

        # =====================================================================
        # 最终融合
        # =====================================================================
        prob = (
                final_prob_1 +
                his_enhance +
                final_prob_3 +
                prob_seq_enhance +
                gcn_enhanced_prob +  # 已被时间门控调制
                temporal_prob +  # 纯时间路径预测
                retrieved_prob  # 已被时间门控调制
        )

        prob = prob.reshape(-1, self.voc_size[2] - 1)
        prob = torch.sigmoid(prob)
        prob = torch.pow(prob, 0.8)

        prob_padder = torch.zeros_like(prob[:, :1])
        prob = torch.cat([prob_padder, prob], dim=-1)

        return (prob, prob * prob.T.unsqueeze(dim=-1),
                [diag_prob_1, diag_prob_2],
                [proc_prob_1, proc_prob_2],
                contrastive_loss,
                temporal_prob,
                gcn_enhanced_prob)

    def forward(self, input, diag_mask=None, proc_mask=None, ages=None, gender=None, drug_mem=None):
        """Forward pass."""
        if isinstance(input, (list, tuple)) and len(input) > 7:
            time_list = input[7]
        else:
            time_list = None

        diag_hid, proc_hid, patient_rep = self.encoder(input[0], input[1], input[3], input[4])
        decoder_out = self.decoder(diag_hid, proc_hid, input[5], input[6], input[2],
                                   patient_rep=patient_rep, time_list=time_list)

        return decoder_out









