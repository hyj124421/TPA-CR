import pandas as pd
import dill
import numpy as np
from collections import defaultdict


##### process medications #####
# load med data
def med_process(med_file):
   
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'})

    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                        'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
                        'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                        'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True) 

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

# medication mapping
def ndc2atc4(med_pd):
    with open(ndc_rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())

    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True) 

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])   
    # 根据RXCUI删除重复列
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)

    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)     # 删除特定的RXCUI
    
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')   
    med_pd = med_pd.reset_index(drop=True)
    
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])     # 合并两个表
    

    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True) # 干掉NDC\RXCUI，只剩ATC4了
    med_pd = med_pd.rename(columns={'ATC4':'NDC'})      # 重新命名为NDC
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])  # 只保留前四位
    med_pd = med_pd.drop_duplicates()    
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

# visit >= 2
def process_visit_lg2(med_pd):
    """筛除admission次数小于两次的患者数据"""
    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    return a


# most common medications
def filter_300_most_med(med_pd):
    # 按照NDC出现的次数降序排列，取前300
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]
    
    return med_pd.reset_index(drop=True)

##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file) #读取CSV文件到pandas DataFrame
    diag_pd.dropna(inplace=True) #删除包含缺失值的行
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True) #删除不需要的列
    diag_pd.drop_duplicates(inplace=True) #删除完全重复的行
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True) #按患者ID和住院ID排序
    diag_pd = diag_pd.reset_index(drop=True) #重置索引，删除旧索引行

    def filter_2000_most_diag(diag_pd):
        #只保留出现频率最高的前2000种诊断
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
        
        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True) #删除行ID列
    pro_pd.drop_duplicates(inplace=True) #删除完全重复的行
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd

def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]
    
    return pro_pd.reset_index(drop=True) 

###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):
    """药物、症状、proc的数据结合"""
    #从三个数据集种提取患者ID和住院ID的唯一组合
    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    # 时间
    adm = pd.read_csv(admission_file)
    adm = adm[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']] #只保留关键列
    adm.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    adm_key = adm[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    # 三个集合的交集，只保留在所有数据集中都存在的住院记录
    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(adm_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    #用交集键过滤每个数据集，只保留共有的记录
    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    adm = adm.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    #将每个住院事件的多余记录聚合成列表
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    # data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x)) #计算药物数量
    data = adm.merge(data, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    




    a = data[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x)) #计算每个患者的住院次数
    a = a[a['HADM_ID_Len'] > 1] #只保留住院次数>1的患者
    data = data.merge(a[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True)
    
    data = data.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])
    print(data)

    return data

def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape) #患者总数
    print('#clinical events ', len(data)) #临床事件总数（住院次数）
    
    diag = data['ICD9_CODE'].values #所有诊断代码列表
    med = data['NDC'].values #所有药物代码列表
    pro = data['PRO_CODE'].values #所有手术代码列表
    
    unique_diag = set([j for i in diag for j in list(i)]) #展平所有诊断代码并去重
    unique_med = set([j for i in med for j in list(i)]) #展平所有药物代码并去重
    unique_pro = set([j for i in pro for j in list(i)]) #展平所有手术代码并去重
    
    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))
    
    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['SUBJECT_ID'].unique(): #遍历每个患者
        item_data = data[data['SUBJECT_ID'] == subject_id] #获取该患者的所有住院记录
        x, y, z = [], [], [] #用于存储该患者的所有诊断、药物、手术代码
        visit_cnt = 0 #该患者的住院次数
        for index, row in item_data.iterrows(): #遍历该患者的每次住院
            visit_cnt += 1
            cnt += 1 #总住院次数计数
            x.extend(list(row['ICD9_CODE'])) #添加本次住院的诊断代码
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        x, y, z = set(x), set(y), set(z) #去重，得到该患者独有的代码集合
        avg_diag += len(x) #累计患者独特诊断数量
        avg_med += len(y) #累计患者独特药物数量
        avg_pro += len(z)
        avg_visit += visit_cnt #累计住院次数
        if len(x) > max_diag: #更新最大诊断数量
            max_diag = len(x)
        if len(y) > max_med: #更新最大药物数量
            max_med = len(y) 
        if len(z) > max_pro: #更新最大手术数量
            max_pro = len(z)
        if visit_cnt > max_visit: #更新最大住院次数
            max_visit = visit_cnt
        
    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of vists ', avg_visit/ len(data['SUBJECT_ID'].unique()))
    
    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)

##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc() 
    med_voc = Voc()
    pro_voc = Voc()
    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE']) 
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])
    
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open('voc_final.pkl','wb'))
    return diag_voc, med_voc, pro_voc

# create final records
def get_season(month):
    if month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    elif month in [9, 10, 11]:
        return 3
    else:
        return 4


def create_patient_record(df, diag_voc, med_voc, pro_voc):
    """
    保存list类型的记录
    每一项代表一个患者，患者中有多个visit，每个visit包含三者数组，按顺序分别表示诊断、proc与药物
    存储的均为编号，可以通过voc_final.pkl来查看对应的具体word
    """
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique(): #遍历每个患者
        item_df = df[df['SUBJECT_ID'] == subject_id] #获取该患者的所有住院记录
        patient = [] #存储单个患者的所有就诊记录
        begin = 0 #初始化起始时间
        for index, row in item_df.iterrows(): #遍历该患者的每次住院
            if begin == 0: #如果是第一次住院
                begin = pd.to_datetime(row['ADMITTIME']) #设置为时间起点
            timestamp = (pd.to_datetime(row['ADMITTIME']) - begin).days #计算距离第一次住院的天数
            admission = [] #单次住院的完整记录
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']]) #将诊断代码转换为数字索引
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['NDC']])
            admission.append(timestamp)
            #admission.append(get_season(pd.to_datetime(row['ADMITTIME']).month))
            patient.append(admission) #将本次住院添加到患者记录中
        records.append(patient) 
    dill.dump(obj=records, file=open('records_final.pkl', 'wb'))
    return records
        


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):

    TOPK = 40 # topk drug-drug interaction
    cid2atc_dic = defaultdict(set) #创建CID到ATC3的映射字典
    med_voc_size = len(med_voc.idx2word) #获取词汇表大小（药物数量）
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]    # 所有的药物的ATC4
    atc3_atc4_dic = defaultdict(set) #创建ATC3到ATC4的映射字典
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)   #提取前4为作为ATC3,映射到完整额ATC4
    
    with open(cid_atc, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',') #按逗号分割每行
            cid = line_ls[0] #STITCH化合物ID
            atcs = line_ls[1:] #对应的所有ATC编码
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0: #如果该ATC3在数据集中存在
                    cid2atc_dic[cid].add(atc[:4]) #建立CID到ATC3的映射
            
    #print(cid2atc_dic)
    # 加载DDI数据
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect，也是采取topK的形式
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:,:] #取频率最高的40种副作用
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    #过滤原始DDI数据，只保留这些严重副作用相关的记录
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)


    # weighted ehr adj 
    ehr_adj = np.zeros((med_voc_size, med_voc_size)) #初始化共现矩阵
    for patient in records: #遍历每个患者
        for adm in patient: #遍历患者的每次住院
            med_set = adm[2] #获取本次住院使用的药物索引列表
            #计算本次住院中所有药物两两共现的次数
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i: #避免重复计数（只计算上三角）
                        continue
                    # ehr_adj[med_i, med_j] = 1
                    # ehr_adj[med_j, med_i] = 1
                    ehr_adj[med_i, med_j] += 1 #药物共现次数+1
                    ehr_adj[med_j, med_i] += 1 #对称位置也+1（无向图）
    dill.dump(ehr_adj, open('ehr_adj_final.pkl', 'wb')) #保存共现矩阵

    # ddi adj，DDI表是CID编码的，因此需要将CID映射到ACT编码，才能记录数据集中药物之间的冲突信息
    ddi_adj = np.zeros((med_voc_size,med_voc_size)) #初始化DDI矩阵
    for index, row in ddi_df.iterrows(): #遍历DDI数据中的每一对相互作用
        # ddi
        cid1 = row['STITCH 1'] #药物1的STITCH ID
        cid2 = row['STITCH 2']
        
        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:
                
                if med_voc.word2idx[atc_i] != med_voc.word2idx[atc_j]: #避免自循环
                        #标记这俩种药物有相互作用
                        ddi_adj[med_voc.word2idx[atc_i], med_voc.word2idx[atc_j]] = 1
                        ddi_adj[med_voc.word2idx[atc_j], med_voc.word2idx[atc_i]] = 1
    dill.dump(ddi_adj, open('ddi_A_final.pkl', 'wb')) 

    return ddi_adj

#计算数据集中药物相互作用的发生率
def ddi_rate_score(record, path):
    # ddi rate
    if isinstance(path, str):
        ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0 #所有药物组合的总数
    dd_cnt = 0 #有相互作用的药物组合的数量
    for patient in record: #遍历每个患者
        for adm in patient: #遍历患者的每次住院记录
            med_code_set = adm[2] #获取本次住院使用的药物索引列表
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i: #避免重复计数
                        continue
                    all_cnt += 1 #总组合数+1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1 #有相互作用的组合数+1
    if all_cnt == 0: #避免除零错误
        return 0
    return dd_cnt / all_cnt #返回DDI发生率比例



if __name__ == '__main__':
    # MIMIC数据文件，分别包括药物、诊断和proc
    med_file = '../datas/mimic-iii/PRESCRIPTIONS.csv'
    diag_file = '../datas/mimic-iii/DIAGNOSES_ICD.csv'
    procedure_file = '../datas/mimic-iii/PROCEDURES_ICD.csv'
    admission_file = '../datas/mimic-iii/ADMISSIONS.csv'

    # 药物信息
    med_structure_file = '../datas/mimic-iii/atc3toSMILES.pkl'   # 药物到分子式的映射

    # drug code mapping files
    ndc2atc_file = '../datas/mimic-iii/ndc2atc_level4.csv'   #用于读取xnorm到ATC
    cid_atc = '../datas/mimic-iii/drug-atc.csv'              #用于处理DDI表
    ndc_rxnorm_file = '../datas/mimic-iii/ndc2rxnorm_mapping.txt'


    ddi_file = '../datas/mimic-iii/drug-DDI.csv'

    # 处理MIMIC中的药物数据
    med_pd = med_process(med_file) # 处方

    # visit_counts = med_pd.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    # print('less than two visit: ', len(visit_counts[visit_counts<2]), len(med_pd))
    
    med_pd = ndc2atc4(med_pd) # 只有atc4
    # med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)   # 注意这里仅仅是针对med表中出现了两次以上admission的patient
    # med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True) # 只要出现了两次以上admission的patient
    # 为了公平比较
    NDCList = dill.load(open(med_structure_file, 'rb'))
    med_pd = med_pd[med_pd.NDC.isin(list(NDCList.keys()))]

    #med_pd = filter_300_most_med(med_pd) # 出现次数排名前300的药物
    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file) # 出现次数排名前2000的诊断集合
    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    print ('complete procedure processing')

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    statistics(data)

    data.to_pickle('data_final.pkl')

    print ('complete combining')
    print(len(data))

    # ddi_matrix
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)   # diag,proc,medication按顺序存储
    print(len(records))

    ## check
    visit_counts = data.groupby('SUBJECT_ID')['HADM_ID'].nunique()
    print('less than two visit: ', len(visit_counts[visit_counts<2]), len(data))
    # for user in records:
    #     if len(user) <=1:
    #         print('yes!!!',user)

    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    ddi_rate = ddi_rate_score(records, "ddi_A_final.pkl")
    print("ddi_rate", ddi_rate)

