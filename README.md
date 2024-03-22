## The Final Project of CS7250


### Train Network
Train corresponding network by commenting and uncommenting for either equivariant network or regular CNN. Run training script by:

```
python train.py
```

### Generate Feature Map
Create directory in project root directory to store the feature maps
```
mkdir feature_map
```
In project root directory, run the feature map generation script by
```
python generate_feature_map.py
```

### Generate CSV data
Using extracted feature maps, compute similarities and store in csv file by running
```
python generate_dataset.py
```