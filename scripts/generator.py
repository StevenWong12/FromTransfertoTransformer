import os

long_dataset = ['Crop', 'ElectricDevices', 'StarLightCurves', 'Wafer', 'TwoPatterns',
                'ECG5000', 'FordA', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ']


short_dataset = ['Coffee', 'Beef', 'OliveOil', 'Rock', 'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ',
                'Wine', 'FaceFour', 'Meat', 'Car']

for long in long_dataset:
     with open('scripts/dilated_pretrain.sh', 'a') as f:
        f.write('python train.py --backbone dilated --dataroot /dev_data/zzj/hzy/datasets/UCR --dataset '+ long +' --mode pretrain --epoch 2000 --batch_size 128 --save_dir dilated_result --loss cross_entropy  \n')    