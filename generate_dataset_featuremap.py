import torch
import torch.nn.functional as F
import numpy as np
import csv
from pathlib import Path
import glob
from tqdm import tqdm


def compute_similarities(result_1, result_2):
    # flipv
    flipped_back_result_1 = torch.flip(result_1, [0]).flatten()
    # fliph
    flipped_back_result_2 = torch.flip(result_2, [1]).flatten()

    sim = F.cosine_similarity(flipped_back_result_1, flipped_back_result_2, dim=0)
    return sim.item()


# feature maps
feature_map_path = 'feature_map/'

class_id_to_name = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

with open('data_flip.csv', 'w', newline='') as csv_file:
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

        for file in tqdm(files[:1]):

            img_id = int(file.split("/")[-1].split(".")[0])
            class_id = int(file.split("/")[-2])
            class_name = class_id_to_name[class_id]

            content = torch.load(file)

            # process feature maps
            for eq in range(2):

                eq_field = 'eq' if eq == 1 else 'CNN'


                flip_h_values = content[eq_field]['flip_h']
                flip_v_values = content[eq_field]['flip_v']

                new_row = dict()
                new_row['img_id'] = img_id
                new_row['class_id'] = class_id
                new_row['class_name'] = class_name
                new_row['eq'] = eq

                for i in range(len(flip_h_values)):
                    new_row[f"x{i}_sim"] = compute_similarities(flip_v_values[f'x{i}'], flip_h_values[f'x{i}'])

                writer.writerow(new_row)
