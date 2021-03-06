# -*- coding: utf-8 -*-
# @Time    : 10/14/20 4:55 PM
# @Author  : Jackie
# @File    : preprocess.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def corr(input1, label):
    x_cols = [col for col in input1.columns if col not in [label] if input1[col].dtype != 'object']  # 处理目标的其他所有特征
    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(input1[col].values, input1.Quality_modify, values)[0, 1])

    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_df = corr_df.sort_values(by='corr_values')

    ind = np.arange(len(labels))
    width = 0.5
    fig, ax = plt.subplots(figsize=(12, 40))
    rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

    ax.set_yticks(ind)
    ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
    ax.set_xlabel('Correlation coefficient')
    ax.set_title('Correlation coefficient of the variables')

    plt.show()
    plt.savefig('corr.png')


def main(input_file):
    """Preprocess module for combine records for same machine"""
    # read data
    kind = input_file.split('.')[-1]
    print("<<<<< input file ", input_file)
    print("<<<< file name type: ", kind)
    if kind == 'xlsx':
        df = pd.read_excel(input_file)
        # key data
        key_df = df.iloc[1:2, :]
        key_df.to_csv('./key.csv')
        # data content
        df_m = df.iloc[2:, 1:]
    if kind == 'csv':
        df = pd.read_csv(input_file, encoding='gbk', delimiter=';')
        df_m = df.iloc[:, :]

    # process parent ID
    df_m['ParentID'] = df_m['FM1_GunType'].apply(lambda x: '-'.join(x.split('-')[:-1]))
    df_m['process_number'] = df_m['FM1_GunType'].apply(lambda x: x.split('-')[-1])

    # combine result for same process
    df_m['Quality'] = df_m['Quality'].fillna(0)
    df_m['Quality_modify'] = df_m['Quality'].apply(lambda x: 1 if x == 'Bad' else 0)

    df_2 = df_m.groupby(['ParentID']).agg({'Quality_modify': max}).reset_index()
    df_2.columns = ['ParentID', 'Quality_res']

    df_res = pd.merge(df_m, df_2, on='ParentID')

    df_m = df_res[df_res['process_number'] == '1.0']

    print(df_m.head())

    # process dtypes
    error_list = []
    for i in df_m.columns:
        (filename, extension) = os.path.splitext(i)
        if extension in ['xlsx', 'csv']:
            # print (i)
            try:
                df_m[i] = df_m[i].astype("float")
            except:
                print("error for: ", i)
                error_list.append(i)
                continue
        else:
            continue

    # non-aggregate
    input1 = df_m[['Quality_res',
                   'FM1_ISOCansTemperature',
                   'FM1_ISOOutletPressure',
                   'FM1_POLOutletPressure',
                   'FM1_POLCansTemperature',
                   'FM1_GunTemperaturePOL',
                   'FM1_GunCurrentAmountOfMaterial',
                   'FM1_GunTemperatureISO',
                   'FM2_ISOCansTemperature',
                   'FM2_ISOFlowRate',
                   'FM2_ISOOutletPressure',
                   'FM2_POLOutletPressure',
                   'FM2_POLFlowRate',
                   'FM2_GunTemperaturePOL',
                   'FM2_GunCurrentAmountOfMaterial',
                   'FM2_GunTemperatureISO',
                   'EM_Temperature',
                   'EM_Humidity',
                   'FoamingPlatformEastCurrentHeight',
                   'FoamingPlatformNewTemperatureProbe_InWater',
                   'FoamingPlatformSetDwellTime',
                   'FoamingPlatformTemperature_BackWater',
                   'FoamingPlatformWestCurrentHeight']]

    for i in error_list:
        try:
            input1.drop(i, axis=1, inplace=True)
        except:
            continue
    # fill null
    input1 = input1.fillna(0)
    return input1


def process_file_list(folder):
    files = os.listdir(folder)
    ls = []
    for i in files:
        try:
            df = main(os.path.join(folder, i))
            ls.append(df)
        except:
            continue
    print(ls)
    df = ls[0]
    for i in range(1, len(ls)):
        df = df.append(ls[i])
    print("<<<<<< df shape", df.shape)
    print(df.head())
    # save file
    df.to_excel('res.xlsx', encoding='utf-8')
    return df


if __name__ == "__main__":
    # input_file= '~/Documents/zhongji/qingdao/data_IOT.xlsx'
    # 数据预处理
    # df = main(input_file)
    # 相关性矩阵
    # corr(df, 'Quality_modify')
    input_folder = '/Users/liujunyi/Desktop/zj/data'
    process_file_list(input_folder)