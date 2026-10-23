from torch.utils import data
import torch


class mimic_data(data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)




def pad_batch_v2_train(batch):
    seq_length = torch.tensor([len(data) for data in batch])
    batch_size = len(batch)
    max_seq = max(seq_length)

    # 统计每一个seq疾病、手术、药物的数量，以及相应的最值
    d_length_matrix = []
    p_length_matrix = []
    m_length_matrix = []
    d_max_num = 0
    p_max_num = 0
    m_max_num = 0
    d_dec_list = []
    d_stay_list = []
    p_dec_list = []
    p_stay_list = []

    # 新增：收集时间信息
    time_list = []  # 存储每个患者的时间序列

    for data in batch:
        d_buf, p_buf, m_buf = [], [], []
        d_dec_list_buf, d_stay_list_buf = [], []
        p_dec_list_buf, p_stay_list_buf = [], []
        patient_times = []  # 当前患者的时间信息

        for idx, seq in enumerate(data):
            d_buf.append(len(seq[0]))
            p_buf.append(len(seq[1]))
            m_buf.append(len(seq[2]))
            d_max_num = max(d_max_num, len(seq[0]))
            p_max_num = max(p_max_num, len(seq[1]))
            m_max_num = max(m_max_num, len(seq[2]))

            # 提取时间信息（如果存在）
            # ✔ 直接用真实时间（第4个位置）
            if len(seq) >= 6:
                patient_times.append(seq[5])
            else:
                patient_times.append(idx)  # fallback（一般不会用到）

            if idx == 0:
                d_dec_list_buf.append([])
                d_stay_list_buf.append([])
                p_dec_list_buf.append([])
                p_stay_list_buf.append([])
            else:
                cur_d = set(seq[0])
                last_d = set(data[idx - 1][0])
                stay_list = list(cur_d & last_d)
                dec_list = list(last_d - cur_d)
                d_dec_list_buf.append(dec_list)
                d_stay_list_buf.append(stay_list)

                cur_p = set(seq[1])
                last_p = set(data[idx - 1][1])
                proc_stay_list = list(cur_p & last_p)
                proc_dec_list = list(last_p - cur_p)
                p_dec_list_buf.append(proc_dec_list)
                p_stay_list_buf.append(proc_stay_list)

        d_length_matrix.append(d_buf)
        p_length_matrix.append(p_buf)
        m_length_matrix.append(m_buf)
        d_dec_list.append(d_dec_list_buf)
        d_stay_list.append(d_stay_list_buf)
        p_dec_list.append(p_dec_list_buf)
        p_stay_list.append(p_stay_list_buf)
        time_list.append(patient_times)

    # 生成m_mask_matrix
    m_mask_matrix = torch.full((batch_size, max_seq, m_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(m_length_matrix[i])):
            m_mask_matrix[i, j, :m_length_matrix[i][j]] = 0.

    # 生成d_mask_matrix
    d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(d_length_matrix[i])):
            d_mask_matrix[i, j, :d_length_matrix[i][j]] = 0.

    # 生成p_mask_matrix
    p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(p_length_matrix[i])):
            p_mask_matrix[i, j, :p_length_matrix[i][j]] = 0.

    # 分别生成disease、procedure、medication的数据
    disease_tensor = torch.full((batch_size, max_seq, d_max_num), 0)
    procedure_tensor = torch.full((batch_size, max_seq, p_max_num), 0)
    medication_tensor = torch.full((batch_size, max_seq, m_max_num), 0)
    age_tensor = torch.full((batch_size, max_seq, 1), 0)
    gender_tensor = torch.full((batch_size, max_seq, 1), 0)

    # 新增：时间张量
    time_tensor = torch.full((batch_size, max_seq, 1), 0, dtype=torch.float32)

    # 分别拼接成一个batch的数据
    for b_id, data in enumerate(batch):
        for s_id, adm in enumerate(data):
            disease_tensor[b_id, s_id, :len(adm[0])] = torch.tensor(adm[0])
            procedure_tensor[b_id, s_id, :len(adm[1])] = torch.tensor(adm[1])
            medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(adm[2])

            if len(adm) >= 5:
                age_tensor[b_id, s_id, 0] = torch.tensor(adm[3])
                gender_tensor[b_id, s_id, 0] = torch.tensor(adm[4])

            # 添加时间信息
            if len(adm) >= 6:
                time_tensor[b_id, s_id, 0] = torch.tensor(adm[5], dtype=torch.float32)
            else:
                time_tensor[b_id, s_id, 0] = torch.tensor(s_id, dtype=torch.float32)

    # 返回8个元素：添加time_tensor
    return (disease_tensor, procedure_tensor, medication_tensor,
            d_mask_matrix, p_mask_matrix, age_tensor, gender_tensor, time_tensor)


def pad_batch_v2_eval(batch):
    """评估模式的批处理函数，同样包含时间信息"""
    seq_length = torch.tensor([len(data) for data in batch])
    batch_size = len(batch)
    max_seq = max(seq_length)

    d_length_matrix = []
    p_length_matrix = []
    m_length_matrix = []
    d_max_num = 0
    p_max_num = 0
    m_max_num = 0
    d_dec_list = []
    d_stay_list = []
    p_dec_list = []
    p_stay_list = []

    time_list = []  # 收集时间信息

    for data in batch:
        d_buf, p_buf, m_buf = [], [], []
        d_dec_list_buf, d_stay_list_buf = [], []
        p_dec_list_buf, p_stay_list_buf = [], []
        patient_times = []

        for idx, seq in enumerate(data):
            d_buf.append(len(seq[0]))
            p_buf.append(len(seq[1]))
            m_buf.append(len(seq[2]))
            d_max_num = max(d_max_num, len(seq[0]))
            p_max_num = max(p_max_num, len(seq[1]))
            m_max_num = max(m_max_num, len(seq[2]))

            # 提取时间信息
            # ✔ 直接用真实时间（第4个位置）
            if len(seq) >= 6:
                patient_times.append(seq[5])
            else:
                patient_times.append(idx)  # fallback（一般不会用到）

            if idx == 0:
                d_dec_list_buf.append([])
                d_stay_list_buf.append([])
                p_dec_list_buf.append([])
                p_stay_list_buf.append([])
            else:
                cur_d = set(seq[0])
                last_d = set(data[idx - 1][0])
                stay_list = list(cur_d & last_d)
                dec_list = list(last_d - cur_d)
                d_dec_list_buf.append(dec_list)
                d_stay_list_buf.append(stay_list)

                cur_p = set(seq[1])
                last_p = set(data[idx - 1][1])
                proc_stay_list = list(cur_p & last_p)
                proc_dec_list = list(last_p - cur_p)
                p_dec_list_buf.append(proc_dec_list)
                p_stay_list_buf.append(proc_stay_list)

        d_length_matrix.append(d_buf)
        p_length_matrix.append(p_buf)
        m_length_matrix.append(m_buf)
        d_dec_list.append(d_dec_list_buf)
        d_stay_list.append(d_stay_list_buf)
        p_dec_list.append(p_dec_list_buf)
        p_stay_list.append(p_stay_list_buf)
        time_list.append(patient_times)

    # 生成mask矩阵
    m_mask_matrix = torch.full((batch_size, max_seq, m_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(m_length_matrix[i])):
            m_mask_matrix[i, j, :m_length_matrix[i][j]] = 0.

    d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(d_length_matrix[i])):
            d_mask_matrix[i, j, :d_length_matrix[i][j]] = 0.

    p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(p_length_matrix[i])):
            p_mask_matrix[i, j, :p_length_matrix[i][j]] = 0.

    # 创建数据张量
    disease_tensor = torch.full((batch_size, max_seq, d_max_num), 0)
    procedure_tensor = torch.full((batch_size, max_seq, p_max_num), 0)
    medication_tensor = torch.full((batch_size, max_seq, m_max_num), 0)
    age_tensor = torch.full((batch_size, max_seq, 1), 0)
    gender_tensor = torch.full((batch_size, max_seq, 1), 0)
    time_tensor = torch.full((batch_size, max_seq, 1), 0, dtype=torch.float32)

    # 填充数据
    for b_id, data in enumerate(batch):
        for s_id, adm in enumerate(data):
            disease_tensor[b_id, s_id, :len(adm[0])] = torch.tensor(adm[0])
            procedure_tensor[b_id, s_id, :len(adm[1])] = torch.tensor(adm[1])
            medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(adm[2])

            if len(adm) >= 5:
                age_tensor[b_id, s_id, 0] = torch.tensor(adm[3])
                gender_tensor[b_id, s_id, 0] = torch.tensor(adm[4])

            # 添加时间信息
            if len(adm) >= 6:
                time_tensor[b_id, s_id, 0] = torch.tensor(adm[5], dtype=torch.float32)
            else:
                time_tensor[b_id, s_id, 0] = torch.tensor(s_id, dtype=torch.float32)

    # 返回8个元素
    return (disease_tensor, procedure_tensor, medication_tensor,
            d_mask_matrix, p_mask_matrix, age_tensor, gender_tensor, time_tensor)



def pad_num_replace(tensor, src_num, target_num):
    # replace_tensor = torch.full_like(tensor, target_num)
    return torch.where(tensor==src_num, target_num, tensor)
    
