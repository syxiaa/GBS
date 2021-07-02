import numpy as np
from GBS_list import GBList

def main(train_data, train_label, purity = 1.0):
    """
        Function function: according to the specific purity threshold, get the particle partition and unbalanced sampling points under the purity threshold
        Input: training set sample, training set label, purity threshold
        Output: sample after pellet sampling, sample label after pellet sampling
    """
    numberSample, numberFeature = train_data.shape

    # Record which class is minority class and which class is majority class, and record the number of minority classes and the number of majority classes
    number_set = set(train_label)
    label_1 = number_set.pop()
    label_2 = number_set.pop()

    if(train_label[(train_label == label_1)].shape[0] < train_label[(train_label == label_2)].shape[0]):
        less_label = label_1
        many_label = label_2
    else:
        less_label = label_2
        many_label = label_1
    DataAll = np.empty(shape=[0, numberFeature])
    DataAllLabel = []


    train = np.hstack((train_data, train_label.reshape(numberSample, 1)))  #Compose a new two dimensional array
    index = np.array(range(0, numberSample)).reshape(numberSample, 1)  #Index column, into two-dimensional array format
    train = np.hstack((train, index))  #Add index column

    granular_balls = GBList.GBList(train, train)
    granular_balls.init_granular_balls(purity=purity, min_sample=numberFeature * 2)  #Initialization
    init_l = granular_balls.granular_balls

    many_len = 0
    less_number = 0
    #A few classes were sampled
    for granular_ball in init_l:
        if granular_ball.label == less_label:
            data = granular_ball.boundaryData
            if granular_ball.purity >= purity:
                DataAll_index = []
                index_i = 0
                for data_item in granular_ball.data:
                    if data_item[numberFeature] == less_label:
                        DataAll_index.append(index_i)

                        less_number += 1
                    index_i += 1
                DataAll = np.vstack((DataAll, granular_ball.data[DataAll_index, : numberFeature]))
                DataAllLabel.extend(granular_ball.data[DataAll_index, numberFeature])
            else:
                DataAll = np.vstack((DataAll, data[:, : numberFeature]))
                DataAllLabel.extend(data[:, numberFeature])
                for data_item in data:
                    if data_item[numberFeature] == less_label:
                        less_number += 1
                    else:
                        many_len += 1
    dict = {}
    number = 0

    # Most classes are sampled
    for granular_ball in init_l:
        if(granular_ball.label == many_label):
            dict[number] = granular_ball.num
        number += 1
    sort_list = sorted(dict.items(), key=lambda item: item[1])

    gb_index = 0
    for sort_item in sort_list:
        granular_ball = init_l[sort_item[0]]
        if granular_ball.purity < purity:
            data = granular_ball.boundaryData
            DataAll = np.vstack((DataAll, data[:, : numberFeature]))
            DataAllLabel.extend(data[:, numberFeature])

            for data_item in data:
                if data_item[numberFeature] == less_label:
                    less_number += 1
                else:
                    many_len += 1
        else:
            if (granular_ball.dim * 2 * (len(dict) - gb_index) + many_len) < less_number:
                DataAll_index = []
                index_i = 0
                for data_item in granular_ball.data:
                    if data_item[numberFeature] == many_label:
                        DataAll_index.append(index_i)
                        many_len += 1
                    index_i += 1
                DataAll = np.vstack((DataAll, granular_ball.data[DataAll_index, : numberFeature]))
                DataAllLabel.extend(granular_ball.data[DataAll_index, numberFeature])
            else:
                data = granular_ball.boundaryData
                DataAll = np.vstack((DataAll, data[:, : numberFeature]))
                DataAllLabel.extend(data[:, numberFeature])
                many_len += data.shape[0]
                if (many_len >= less_number):
                    break
        gb_index += 1
    return DataAll, DataAllLabel