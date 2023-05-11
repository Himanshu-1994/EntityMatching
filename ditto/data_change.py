import pandas as pd
import numpy as np
from sklearn import metrics as m
import os
from pprint import pprint


def read_file(path):

    with open(path , 'r') as fp:
        data = fp.readlines()

    left = []
    right = []
    label = []
    for line in data:
        temp = line.split('\t')
        left.append(temp[0])
        right.append(temp[1])
        label.append(temp[2].strip('\n'))

    l = []
    for s in left:
        elements = s.strip().split('COL ')
        d = {}
        for col in elements:
            if not col:
                continue
            temp = col.strip().split(' VAL')
            d[temp[0]] = temp[1].strip() if temp[1] else ''
        l.append(d)

    r = []
    for s in right:
        elements = s.strip().split('COL ')
        d = {}
        for col in elements:
            if not col:
                continue
            temp = col.strip().split(' VAL')
            d[temp[0]] = temp[1].strip() if temp[1] else ''
        r.append(d)
    
    df_l = pd.DataFrame(l)
    gcols = [col for col in df_l.columns]
    df_l.columns = [str(col) + '_left' for col in df_l.columns]
    df_r = pd.DataFrame(r)
    df_r.columns = [str(col) + '_right' for col in df_r.columns]
    df_label = pd.DataFrame(label , columns=['target'])

    df = pd.concat([df_l , df_r, df_label] , axis = 1)

    l0 = len(df[df['target'] == '0'])
    l1 = len(df[df['target'] == '1'])
    
    for column in df.columns:
        if column == 'target':
            continue
        nulls_0 = df[(df[column] == '') & (df['target'] == '0')]
        nulls_1 = df[(df[column] == '') & (df['target'] == '1')]
        print(f'{column} having target 0 has {len(nulls_0)} null values out of {l0} values')
        print(f'{column} having target 1 has {len(nulls_1)} null values out of {l1} values')
        print('#######################################')

    new_df = df[(df['price_left'] != '') & (df['price_right'] != '') & (df['target'] == '1')]
    temp = new_df['price_left'].astype(float) - new_df['price_right'].astype(float)
    print(f'Mean and Std of price column with target = 1 is {np.mean(temp.to_numpy())} and {np.std(temp.to_numpy())}')

    count = 0
    
    for i in range(len(df)):
        if df.iloc[i]['target'] == '1':
            for col in gcols:
                cleft = df.iloc[i][col + '_left']
                cright = df.iloc[i][col + '_right']
                if cleft and cright:
                    continue
                elif not cleft and not cright:
                    continue
                else:
                    if cleft:
                        df.iloc[i][col + '_right'] = cleft
                    else:
                        df.iloc[i][col + '_left'] = cright
                    count += 1
                    
    print(f'{count} null values replaced')
                

    return df, gcols

def write_to_file(df, gcols, fname):

    cols = ['COL ' + x for x in gcols]
    out = []
    fp = open(fname , 'w')
    
    for i in range(len(df)):
        df_l = df.iloc[i, :len(gcols)]
        df_r = df.iloc[i, len(gcols):-1]
        label = '\t' + df.iloc[i, -1] + '\n'

        left = [' VAL ' + x for x in df.iloc[i, :len(gcols)].to_list()]
        right = [' VAL ' + x for x in df.iloc[i, len(gcols):-1].to_list()]

        left = [cols[i] + left[i] for i in range(len(cols))]
        right = [cols[i] + right[i] for i in range(len(cols))]

        res = ' '.join(left)+' ' + '\t' + ' '.join(right)+' '

        res += label

        fp.write(res)

        out.append(res)

    fp.close()

    return out

path = 'data/er_magellan/Structured/Walmart-Amazon/train.txt'
fname = 'data/er_magellan/Structured/Walmart-Amazon/train.txt.augment2'
df, gcols = read_file(path)
write_to_file(df, gcols, fname)
