import pandas as pd
import numpy as np
from sklearn import metrics as m

def parse_file(file):

    df = pd.read_csv(file)
    print(m.classification_report(df['targets'] , df['match']))
    
    res = pd.DataFrame()
    
    for loc in ['left' , 'right']:
        d = []
        for i in range(len(df)):
            s = df[loc].iloc[i]
            t = {}
            for val in s.split('COL '):
                if not val:
                    continue
                temp = val.strip().split(' VAL')
                t[temp[0] + '_' + loc] = temp[1].strip() if temp[1] else ''
            d.append(t)
            
        res = pd.concat([res , pd.DataFrame(d)] , axis = 1)

    remaining = [x for x in df.columns if x not in ['left', 'right']]

    res = pd.concat([res, df[remaining]] , axis = 1)
    filtered_res_fp = res[(res['match'] == 1) & (res['targets'] == 0)]
    filtered_res_fn = res[(res['match'] == 0) & (res['targets'] == 1)]
    filtered_res = pd.concat([filtered_res_fp , filtered_res_fn] , axis = 0)
    
    return filtered_res,res

if __name__=="__main__":
    input_path = '/staging/pandotra/roberta/output_roberta.csv'
    output_path = '/staging/pandotra/roberta/output_roberta_analyse.csv'
    output_path2 = '/staging/pandotra/roberta/output_roberta_analyse_all.csv'

    df,df2 = parse_file(input_path)
    df.to_csv(output_path)
    df2.to_csv(output_path2)
