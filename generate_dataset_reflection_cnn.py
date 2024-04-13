import torch
import torch.nn.functional as F
import numpy as np
import csv
from pathlib import Path
import glob
from tqdm import tqdm


def compute_similarities(base, result, flip_type):
    base = base.flatten()
    if flip_type == 'flip_h':
        flipped_back_result = torch.flip(result, [1]).flatten()
    elif flip_type == 'flip_v':
        flipped_back_result = torch.flip(result, [0]).flatten()
    else:
        flipped_back_result = result.flatten()

    sim = F.cosine_similarity(base, flipped_back_result, dim=0)
    return sim.item()


# feature maps
feature_map_path = 'feature_map/'

class_id_to_name = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

with open('data_3.csv', 'w', newline='') as csv_file:
    fieldNames = [
        'img_id',
        'class_id', 'class_name',
        'flip_type',
        'eq',
        'x0_sim', 'x1_sim', 'x2_sim', 'x3_sim', 'x4_sim'
    ]
    writer = csv.DictWriter(csv_file, fieldNames)
    writer.writeheader()

    # iterate through each class folder
    for i in range(10):

        # glob all files with label = i
        files = glob.glob(feature_map_path + str(i) + "/*.pth")

        for file in tqdm(files):

            img_id = int(file.split("/")[-1].split(".")[0])
            class_id = int(file.split("/")[-2])
            class_name = class_id_to_name[class_id]

            content = torch.load(file)

            # process feature maps
            # for eq in range(2):

                # eq_field = 'eq' if eq == 1 else 'CNN'
            eq_field = 'CNN'
            base = dict()
            for flip_type, value in content[eq_field].items():

                if flip_type == 'original':
                    base['x0'] = value['x0']
                    base['x1'] = value['x1']
                    base['x2'] = value['x2']
                    base['x3'] = value['x3']
                    base['x4'] = value['x4']

                new_row = dict()
                new_row['img_id'] = img_id
                new_row['class_id'] = class_id
                new_row['class_name'] = class_name
                new_row['flip_type'] = flip_type
                new_row['eq'] = 0

                # compute cos similarity
                for name, feature_map in value.items():
                    new_row[name + "_sim"] = compute_similarities(base[name], feature_map, flip_type)

                writer.writerow(new_row)
