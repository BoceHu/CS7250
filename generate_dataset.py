import torch
import torch.nn.functional as F
from torcheval.metrics.functional import mean_squared_error
import numpy as np
import csv
from pathlib import Path
import glob
from tqdm import tqdm
from rotate_imgs import rotate_image, circle_mask


def cos_score(base, result, angle):

    base = base /torch.max(base)
    result = result / torch.max(result)

    base = base.flatten()
    rotated_back_result =rotate_image(result.unsqueeze(0), -angle).flatten()

    # eval_base = eval_mask(base)
    # eval_result = eval_mask(rotated_back_result)+
    # sim = F.cosine_similarity(eval_base, eval_result, dim=0)

    sim = F.cosine_similarity(base, rotated_back_result, dim=0)

    return sim.item()

def mse_score(base, result, angle):

    base = base /torch.max(base)
    result = result / torch.max(result)

    base = base.flatten()
    rotated_back_result =rotate_image(result.unsqueeze(0), -angle).flatten()

    # eval_base = eval_mask(base)
    # eval_result = eval_mask(rotated_back_result)

    # sim = mean_squared_error(eval_base, eval_result)

    sim = mean_squared_error(base, rotated_back_result)

    return sim.item()


# feature maps
feature_map_path = 'feature_map/'


class_id_to_name = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

with open('data_back_up.csv', 'w', newline='') as csv_file:

    fieldNames = [
        'img_id', 
        'class_id', 
        'rotation', 
        'eq', 
        'cos', 'mse'
    ]
    writer = csv.DictWriter(csv_file, fieldNames)
    writer.writeheader()

    # iterate through each class folder
    for i in range(10):

        # glob all files with label = i
        files = glob.glob(feature_map_path+str(i)+"/*.pth")

        for file in tqdm(files):

            img_id = int(file.split("/")[-1].split(".")[0])
            class_id = int(file.split("/")[-2])

            content = torch.load(file)

            # process feature maps
            for eq in range(2):

                eq_field = 'eq' if eq == 1 else 'CNN'

                base = dict()
                for rotation, value in content[eq_field].items():
                                            
                    rot = int(rotation)
                    if rot == 0:
                        base['map'] = value['map']

                    new_row = dict()
                    new_row['img_id'] = img_id
                    new_row['class_id'] = class_id
                    new_row['rotation'] = rot
                    new_row['eq'] = eq
                    for name, feature_map in value.items():
                        new_row["cos"] = cos_score(base[name], feature_map, rot)
                        new_row["mse"] = mse_score(base[name], feature_map, rot)

                    writer.writerow(new_row)
