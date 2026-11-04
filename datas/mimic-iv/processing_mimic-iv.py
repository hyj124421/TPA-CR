import pandas as pd
from datetime import datetime
import dill
import numpy as np
from collections import defaultdict
import os
import ast


def log_shape(name, df):
    print(f'{name}: {df.shape}')


def normalize_ndc(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value.endswith('.0'):
        value = value[:-2]
    value = ''.join(ch for ch in value if ch.isdigit())
    if value == '' or int(value) == 0:
        return np.nan
    return value.zfill(11)

##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={'ndc': str}, low_memory=False)

    med_pd.drop(columns=['pharmacy_id','stoptime','drug_type','drug','gsn','prod_strength',
                        'form_rx','dose_val_rx','dose_unit_rx','form_val_disp','form_unit_disp',
                        'doses_per_24_hrs','route'], axis=1, inplace=True)

    med_pd['ndc'] = med_pd['ndc'].map(normalize_ndc)
    med_pd.dropna(subset=['ndc'], inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    # med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['starttime'] = pd.to_datetime(med_pd['starttime'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['subject_id', 'hadm_id', 'starttime'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    # med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

# medication mapping
def ndc2atc4(med_pd):
    with open(ndc_rxnorm_file, 'r') as f:
        ndc2rxnorm = ast.literal_eval(f.read())
    ndc2rxnorm = {
        normalize_ndc(k): v
        for k, v in ndc2rxnorm.items()
        if not pd.isna(normalize_ndc(k))
    }
    med_pd['RXCUI'] = med_pd['ndc'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)
    # print("med_pd", med_pd)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
    
    med_pd['RXCUI'] = pd.to_numeric(med_pd['RXCUI'], errors='coerce')
    med_pd.dropna(subset=['RXCUI'], inplace=True)
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    rxnorm2atc['RXCUI'] = pd.to_numeric(rxnorm2atc['RXCUI'], errors='coerce')
    rxnorm2atc.dropna(subset=['RXCUI'], inplace=True)
    rxnorm2atc['RXCUI'] = rxnorm2atc['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['ndc', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4':'ndc'})
    med_pd['ndc'] = med_pd['ndc'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()    
    med_pd = med_pd.reset_index(drop=True)
    # print(med_pd)
    return med_pd

# visit >= 2
def process_visit_lg2(med_pd):
    a = med_pd[['subject_id', 'hadm_id']].groupby(by='subject_id')['hadm_id'].unique().reset_index()
    a['hadm_id_len'] = a['hadm_id'].map(lambda x:len(x))
    a = a[a['hadm_id_len'] > 1]
    return a 

# most common medications
def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['ndc']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['ndc'].isin(med_count.loc[:299, 'ndc'])]
    
    return med_pd.reset_index(drop=True)

##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['seq_num'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['subject_id','hadm_id'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['icd_code']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['icd_code'].isin(diag_count.loc[:2000, 'icd_code'])]
        
        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'icd_code':'category'})
    pro_pd.drop(columns=['chartdate'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'], inplace=True)
    pro_pd.drop(columns=['seq_num'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd

def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['icd_code']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['icd_code'].isin(pro_count.loc[:2000, 'icd_code'])]
    
    return pro_pd.reset_index(drop=True) 

###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):

    med_pd_key = med_pd[['subject_id', 'hadm_id']].drop_duplicates()
    diag_pd_key = diag_pd[['subject_id', 'hadm_id']].drop_duplicates()
    pro_pd_key = pro_pd[['subject_id', 'hadm_id']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['subject_id', 'hadm_id'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['subject_id', 'hadm_id'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['subject_id','hadm_id'])['icd_code'].unique().reset_index()  
    med_pd = med_pd.groupby(by=['subject_id', 'hadm_id'])['ndc'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['subject_id','hadm_id'])['icd_code'].unique().reset_index().rename(columns={'icd_code':'pro_code'})  
    med_pd['ndc'] = med_pd['ndc'].map(lambda x: list(x))
    pro_pd['pro_code'] = pro_pd['pro_code'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['subject_id', 'hadm_id'], how='inner')
    data = data.merge(pro_pd, on=['subject_id', 'hadm_id'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['ndc_len'] = data['ndc'].map(lambda x: len(x))

    a = data[['subject_id', 'hadm_id']].groupby(by='subject_id')['hadm_id'].unique().reset_index()
    a['HADM_ID_Len'] = a['hadm_id'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    data = data.merge(a[['subject_id']], on='subject_id', how='inner').reset_index(drop=True)


    return data


def add_adm_time(data, admission_file):
    adm_data = pd.read_csv(admission_file)
    adm_data = adm_data[['subject_id', 'hadm_id', 'admittime']]
    adm_data.dropna(inplace=True)
    data = data.merge(adm_data, on=['subject_id', 'hadm_id'], how='inner')
    data['admittime'] = data['admittime'].map(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return data

def statistics(data):
    print('#patients ', data['subject_id'].unique().shape)
    print('#clinical events ', len(data))
    if data.empty:
        raise ValueError('No valid MIMIC-IV records after filtering. Check the printed shapes above, especially NDC->ATC mapping and SMILES filtering.')
    
    diag = data['icd_code'].values
    med = data['ndc'].values
    pro = data['pro_code'].values
    
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    
    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))
    
    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['subject_id'].unique():
        item_data = data[data['subject_id'] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['icd_code']))
            y.extend(list(row['ndc']))
            z.extend(list(row['pro_code']))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y) 
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
    
    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of vists ', avg_visit/ len(data['subject_id'].unique()))
    
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
        diag_voc.add_sentence(row['icd_code'])
        med_voc.add_sentence(row['ndc'])
        pro_voc.add_sentence(row['pro_code'])
    
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open('voc_final_4.pkl','wb'))
    return diag_voc, med_voc, pro_voc

# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['subject_id'].unique():
        item_df = df[df['subject_id'] == subject_id]
        item_df = item_df.sort_values(by=['admittime'])
        patient = []
        begin = 0
        for index, row in item_df.iterrows():
            if begin == 0:
                begin = pd.to_datetime(row['admittime'])
            timestamp = (pd.to_datetime(row['admittime']) - begin).days 
            admission = []
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['icd_code']])
            admission.append([pro_voc.word2idx[i] for i in row['pro_code']])
            admission.append([med_voc.word2idx[i] for i in row['ndc']])
            admission.append(timestamp)
            patient.append(admission)
        records.append(patient) 
    dill.dump(obj=records, file=open('records_final_4.pkl', 'wb'))
    return records
        


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):

    TOPK = 40 # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)
    
    with open(cid_atc, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])
            
    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect 
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:,:]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)

    # weighted ehr adj 
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    ehr_adj_num = np.zeros((med_voc_size, med_voc_size))
    med_dict = dict()

    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                if med_dict.get(med_i) == None:
                    med_dict[med_i] = 0
                else:
                    med_dict[med_i] += 1                
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    if med_dict.get(med_j) == None:
                        med_dict[med_j] = 0
                    else:
                        med_dict[med_j] += 1
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
                    ehr_adj_num[med_i, med_j] += 1
                    ehr_adj_num[med_j, med_i] += 1
    dill.dump(ehr_adj, open('ehr_adj_final_4.pkl', 'wb'))  
    dill.dump(ehr_adj_num, open('ehr_adj_num_final_4.pkl', 'wb'))  
    dill.dump(med_dict, open('med_dict.pkl', 'wb'))


    # ddi adj
    ddi_adj = np.zeros((med_voc_size,med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']
        
        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:
                
                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open('ddi_A_final_4.pkl', 'wb')) 

    return ddi_adj


def ddi_rate_score(record, path):
    # ddi rate
    if isinstance(path, str):
        ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        adm = patient[-1]
        #for adm in patient:
        med_code_set = adm[-2]
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


if __name__ == '__main__':

    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    # please change into your own MIMIC folder

    med_file = "../datas/mimic-iv/prescriptions.csv"
    diag_file ="../datas/mimic-iv/diagnoses_icd.csv"
    procedure_file ="../datas/mimic-iv/procedures_icd.csv"
    admission_file ="../datas/mimic-iv/admissions.csv"

    med_structure_file = '../datas/mimic-iv/atc3toSMILES.pkl'

    # drug code mapping files
    ndc2atc_file = '../datas/mimic-iv/ndc2atc_level4.csv'
    cid_atc = '../datas/mimic-iv/drug-atc.csv'
    ndc_rxnorm_file = '../datas/mimic-iv/ndc2rxnorm_mapping.txt'

    # ddi information
    ddi_file = '../datas/mimic-iv/drug-DDI.csv'

    base_dir = os.path.dirname(os.path.abspath(__file__))
    med_file = os.path.join(base_dir, 'prescriptions.csv')
    diag_file = os.path.join(base_dir, 'diagnoses_icd.csv')
    procedure_file = os.path.join(base_dir, 'procedures_icd.csv')
    admission_file = os.path.join(base_dir, 'admissions.csv')
    med_structure_file = os.path.join(base_dir, 'atc3toSMILES.pkl')
    ndc2atc_file = os.path.join(base_dir, 'ndc2atc_level4.csv')
    cid_atc = os.path.join(base_dir, 'drug-atc.csv')
    ndc_rxnorm_file = os.path.join(base_dir, 'ndc2rxnorm_mapping.txt')
    ddi_file = os.path.join(base_dir, 'drug-DDI.csv')

    # for med
    med_pd = med_process(med_file)
    log_shape('after med_process', med_pd)
    # med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)    
    # med_pd = med_pd.merge(med_pd_lg2[['subject_id']], on='subject_id', how='inner').reset_index(drop=True) 
    med_pd = ndc2atc4(med_pd)
    log_shape('after ndc2atc4', med_pd)
    NDCList = dill.load(open(med_structure_file, 'rb'))
    # print("NDCList", NDCList)
    # print("pos1", med_pd)
    print('SMILES medicine size:', len(NDCList))
    print('matched SMILES medicines:', med_pd.ndc.isin(list(NDCList.keys())).sum())
    med_pd = med_pd[med_pd.ndc.isin(list(NDCList.keys()))]
    log_shape('after SMILES filter', med_pd)
    # print("pos2", med_pd)
    #  med_pd = filter_300_most_med(med_pd)
    # print("pos2", med_pd)


    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file)
    log_shape('after diagnosis processing', diag_pd)

    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    pro_pd = filter_1000_most_pro(pro_pd)
    log_shape('after procedure processing', pro_pd)

    print ('complete procedure processing')

    # combine
    # print("med_pd,diag_pd,pro_pd")
    # print(med_pd, diag_pd, pro_pd)
    data = combine_process(med_pd, diag_pd, pro_pd)
    log_shape('after combine_process', data)
    data = add_adm_time(data, admission_file)
    log_shape('after add_adm_time', data)
    statistics(data)
    data.to_pickle('data_final_4.pkl')

    print ('complete combining')


    # ddi_matrix
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    ddi_rate = ddi_rate_score(records, "ddi_A_final_4.pkl")
    print("ddi_rate", ddi_rate)
