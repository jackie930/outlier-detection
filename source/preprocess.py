# -*- coding: utf-8 -*-
# @Time    : 10/14/20 4:55 PM
# @Author  : Jackie
# @File    : preprocess.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def corr(input1, label):
    x_cols = [col for col in input1.columns if col not in [label] if input1[col].dtype!='object']#处理目标的其他所有特征
    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(input1[col].values, input1.Quality_modify,values)[0, 1])

    corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
    corr_df = corr_df.sort_values(by = 'corr_values')

    ind = np.arange(len(labels))
    width = 0.5
    fig,ax = plt.subplots(figsize = (12,40))
    rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

    ax.set_yticks(ind)
    ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
    ax.set_xlabel('Correlation coefficient')
    ax.set_title('Correlation coefficient of the variables')

    plt.show()
    plt.savefig('corr.png')


def main(input_file):
    """Preprocess module for combine records for same machine"""
    #read data
    df = pd.read_excel(input_file)
    #key data
    key_df = df.iloc[1:2,:]
    #data content
    df_m = df.iloc[2:,1:]
    #process parent ID
    df_m['ParentID'] = df_m['FM1_GunType'].apply(lambda x: '-'.join(x.split('-')[:-1]))
    #combine result for same process
    df_m['Quality'] = df_m['Quality'].fillna(0)
    df_m['Quality_modify'] = df_m['Quality'].apply(lambda x: 1 if x=='Bad' else 0)

    #process dtypes
    error_list = []
    for i in df_m.columns:
        #print (i)
        try:
            df_m[i]=df_m[i].astype("float")
        except:
            print ("error for: ", i)
            error_list.append(i)
            continue


    #aggregate
    input1 = df_m.groupby('ParentID').agg({'Quality_modify':np.sum,
                                           'FM1_ISOCansTemperature':np.mean,
                                           'FM1_ISOOutletPressure':np.mean,
                                           'FM1_POLOutletPressure':np.mean,
                                           'FM1_POLCansTemperature':np.mean,
                                           'FM1_GunTemperaturePOL':np.mean,
                                           'FM1_GunCurrentAmountOfMaterial':np.mean,
                                           'FM1_GunTemperatureISO':np.mean,
                                           'FM2_ISOCansTemperature':np.mean,
                                           'FM2_ISOFlowRate':np.mean,
                                           'FM2_ISOOutletPressure':np.mean,
                                           'FM2_POLOutletPressure':np.mean,
                                           'FM2_POLFlowRate':np.mean,
                                           'FM2_GunTemperaturePOL':np.mean,
                                           'FM2_GunCurrentAmountOfMaterial':np.mean,
                                           'FM2_GunTemperatureISO':np.mean,
                                           'EM_Temperature':np.mean,
                                           'EM_Humidity':np.mean,
                                           'FoamingPlatformEastCurrentHeight':np.mean,
                                           'FoamingPlatformNewTemperatureProbe_InWater':np.mean,
                                           'FoamingPlatformSetDwellTime':np.mean,
                                           'FoamingPlatformTemperature_BackWater':np.mean,
                                           'FoamingPlatformWestCurrentHeight':np.mean}).reset_index()

    for i in error_list:
        try:
            input1.drop(i,axis=1, inplace=True)
        except:
            continue
    #fill null
    input1 = input1.fillna(0)
    return input1

if __name__ == "__main__":
    input_file= '~/Documents/zhongji/qingdao/data_IOT.xlsx'
    #数据预处理
    df = main(input_file)
    #相关性矩阵
    corr(df, 'Quality_modify')