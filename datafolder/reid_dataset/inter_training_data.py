import os


# from .reiddataset_downloader import *


def inter_training_data(data_dir):
    #dataset_dir = os.path.join(data_dir, dataset_name)

    # if not os.path.exists(dataset_dir):
    #     print('Please Download ' + dataset_name + ' Dataset')

    name_dir = data_dir
    #data_group = ['train', 'query', 'gallery']
    group='train'
    file_list = sorted(os.listdir(name_dir))
    globals()[group] = {}
    globals()[group]['data'] = []
    globals()[group]['ids'] = []
    for name in file_list:
        if name[-3:] == 'npy':
            id = name.split('_')[0]
            cam = int(name.split('_')[1][1])
            images = os.path.join(name_dir, name)
            if (id != '0000' and id != '-1'):
                if id not in globals()[group]['ids']:
                    globals()[group]['ids'].append(id)
                globals()[group]['data'].append(
                    [images, globals()[group]['ids'].index(id), id, cam, name.split('.')[0]])
    return train