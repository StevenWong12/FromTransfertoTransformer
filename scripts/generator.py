import os

long_dataset = ['Crop', 'ElectricDevices', 'StarLightCurves', 'Wafer', 'TwoPatterns',
                'ECG5000', 'FordA', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ']


short_dataset = ['Coffee', 'Beef', 'OliveOil', 'Rock', 'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ',
                'Wine', 'FaceFour', 'Meat', 'Car']

for long in long_dataset:
   for short in short_dataset:
      with open('scripts/dilated_finetune.sh', 'a') as f:
         f.write('python train.py --backbone dilated --dataroot /dev_data/zzj/hzy/datasets/UCR --dataset '+ short +' --mode finetune --epoch 2000 --batch_size 32 --save_dir dilated_result --loss cross_entropy --source_dataset  '+ long+'\n')    