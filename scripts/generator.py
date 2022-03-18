import os

long_dataset = ['Crop', 'ElectricDevices', 'StarLightCurves', 'Wafer', 'TwoPatterns',
                'ECG5000', 'FordA', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ']


short_dataset = ['Coffee', 'Beef', 'OliveOil', 'Rock', 'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ',
                'Wine', 'FaceFour', 'Meat', 'Car']

for long in long_dataset:
    for dataset in short_dataset:
        with open('scripts/ucr_finetune.sh', 'a') as f:
            f.write('python train.py --backbone fcn --dataroot /dev_data/zzj/hzy/datasets/UCR --dataset '+dataset 
            +' --mode finetune --epoch 1000 --batch_size 32 --loss cross_entropy --source_dataset '+long + '\n')    