import os

from numpy import sort

long_dataset = ['Crop', 'ElectricDevices', 'StarLightCurves', 'Wafer', 'TwoPatterns',
                'ECG5000', 'FordA', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ']


short_dataset = ['Coffee', 'Beef', 'OliveOil', 'Rock', 'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ',
                'Wine', 'FaceFour', 'Meat', 'Car']

'''
for long in long_dataset:
   with open('scripts/reconstruct_fcn_pretrain.sh', 'a') as f:
      f.write('python train.py --backbone fcn --task reconstruction --dataroot /dev_data/zzj/hzy/datasets/UCR --dataset '+ long +' --mode pretrain --epoch 2000 --batch_size 128 --save_dir ./rnn_result --loss reconstruction\n') 
      '''

long_dataset = sorted(long_dataset)
short_dataset = sorted(short_dataset)   

for long in long_dataset:
   with open('scripts/rnn_dilated_pretrain.sh', 'a') as f:
      f.write('python train.py --backbone dilated --dataroot /dev_data/zzj/hzy/datasets/UCR --dataset ' + long +' --mode pretrain --epoch 2000 --batch_size 128 --save_dir ./result_v3/dilated_rnn_v3  --task reconstruction --loss reconstruction\n')