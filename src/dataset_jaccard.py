import dill
import matplotlib.pyplot as plt
import numpy as np

def jaccard_calculate(pre,temp):
    return len(pre & temp)/len(pre | temp)

def data_set_preprocess(load_file,dump_file):
    data = dill.load(open(load_file, 'rb'))
    for patient in range(len(data)):
        for vst in range(len(data[patient])):
            for label in range(len(data[patient][vst][1])):
                data[patient][vst][1][label] += 2000
            for label in range(len(data[patient][vst][2])):
                data[patient][vst][2][label] += 4000
    for patient in range(len(data)):
        for vst in range(len(data[patient])):
            data[patient][vst] = data[patient][vst][0] + data[patient][vst][1] + data[patient][vst][2]
            data[patient][vst] = set(data[patient][vst])
    dill.dump(data,open(dump_file,'wb'))
    return dump_file

def calculate_dataset_jaccard(load_file,dump_file):
    data = dill.load(open(load_file,'rb'))
    dataset_jaccard = []
    for patient in range(len(data)):
        if len(data[patient])>1:
            for vst in range(1,len(data[patient])):
                dataset_jaccard.append(jaccard_calculate(data[patient][vst-1],data[patient][vst]))
    dill.dump(dataset_jaccard,open(dump_file,'wb'))
    return None

def draw_jaccard(jaccard_path,step=1,top=50):

    def ten_div(input):
        i = 1
        while True:
            if input<=i:
                return int(i/10)
            i*=10
    
    interval_ranges = [] 
    step = 1
    top = 50
    for i in range(0,top,step):
        interval_ranges.append([i/top,(i+step)/top])

    data = dill.load(open(jaccard_path,'rb'))

    interval_counts = [0] * len(interval_ranges)

    for value in data:
        for i, (start, end) in enumerate(interval_ranges):
            if start <= value < end:
                interval_counts[i] += 1
                break

    labels = [f"{end}" for start, end in interval_ranges]
    x = range(len(interval_ranges))
    plt.bar(x, interval_counts, tick_label=labels)
    step = ten_div(top)
    plt.xticks(x[::step],labels[step-1::step])
    # plt.xticks()

    plt.xlabel('Jaccard')
    plt.ylabel('Frequency')
    # plt.title('Distribution of Values in Intervals')
    plt.show()


jaccard_load = data_set_preprocess('records_final.pkl','数据集分析\iii_dataset.pkl')
calculate_dataset_jaccard(jaccard_load,'数据集分析\iii_jaccard.pkl')
draw_jaccard('数据集分析\iii_jaccard.pkl')
