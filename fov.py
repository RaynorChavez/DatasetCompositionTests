from PIL import Image
import seaborn as sns
import os
import numpy as np
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import shutil
import json


def fov_get(path):
    print(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read()).get('field_of_view_rads', -1)

def copy_files(src_dst_tup):
    src, dst = src_dst_tup
    
    src = 'D:/Projects/omnidata_starter_dataset/rgb/' + src
    dst = 'D:/Projects/omnidata_starter_dataset/rgb/' + dst

    if not os.path.exists(src):
        return
    
    shutil.copy(src, dst)
    return 

if __name__=='__main__':
    os.chdir('D:/Projects/omnidata_starter_dataset/point_info')
    # dim = Image.open('point_4_view_3_domain_depth_euclidean.png', 'r')
    # path = 'point_4_view_3_domain_rgb.png'
    # print(bright(DIR+path))

    
    pool = mp.Pool(8)
    di = {}
    classes = {}
    kmeans = KMeans(n_clusters=3)
    for DIR in ['replica/frl_apartment_0/', 'taskonomy/allensville/'][1:]:
        df = pd.DataFrame([])
        df['dirs'] = [DIR + x for x in os.listdir(DIR) if os.path.isfile(DIR+x)]
        df['rgb_dirs'] = df['dirs'].map(lambda x: x.replace('point_info.json', 'rgb.png')) # fixatedpose.json
        df['fov'] = pool.map(fov_get, df['dirs'])
        
        df.loc[df['fov']!=-1, 'class'] = kmeans.fit_predict(np.array(df[df['fov']!=-1]['fov']).reshape(-1, 1))
        df.loc[df['fov']!=-1, 'new_dir'] = df[df['class'].notna()].apply(lambda x: '/'.join(x['dirs'].split('/')[0:2] + ['fov_groups', str(int(x['class']))]), axis=1)
        sns.scatterplot(x=df.loc[df['fov']!=-1, 'fov'], y=df.loc[df['fov']!=-1, 'class']).set(title=f"Scatterplot by Class of {DIR}")
        plt.show()
        sns.histplot(df.loc[df['fov']!=-1, 'fov']).set(title=f"FOV Distribution of {DIR}")
        plt.show()
        [os.makedirs('D:/Projects/omnidata_starter_dataset/rgb/' + x) for x in df['new_dir'].unique() if type(x) is not float]
        
        # pool.map(copy_files, df.loc[df['class'].notna(), ['rgb_dirs', 'new_dir']].to_numpy())
        
        # df.to_csv('D:/Projects/omnidata_starter_dataset/'+DIR.strip('/').split('/')[-1]+'_fov.csv', encoding='utf-8', index=False)
        print(df['class'].value_counts())
        print(df.dropna().set_index('class').mean(level=0))
        break
        
    
    pool.close()
    '''
    sns.histplot([x for x in li if type(x) is not str])
    plt.show()
    '''
