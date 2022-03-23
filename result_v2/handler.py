import os 
import pandas as pd

results = os.listdir('./result_v2/dilated_v2')
#results.remove('handler.py')

result = results[0]
df = pd.read_csv(os.path.join('./result_v2/dilated_v2', result, 'finetune_result.csv'))
df = df.iloc[:, -3:]
indices = df['target'].to_numpy()
out = pd.DataFrame(index=indices, columns=results)

for index, row in df.iterrows():
    print(row['target'], row['accuracy'], row['var'])

for result in results:
    df = pd.read_csv(os.path.join('./result_v2/dilated_v2', result, 'finetune_result.csv'))
    df = df.iloc[:, -3:]

    for _, row in df.iterrows():
        out.loc[row['target']][result] = str('%.2f' % row['accuracy']) + '±' + str('%.2f' % row['var'])

out = out.sort_index()
out = out.sort_index(axis=1)
out.to_csv('./result_v2/dilated_v2/dilated_v2.csv')
