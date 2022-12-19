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


def bright(path):
    print(path)
    try:
        image = cv2.imread(path)
        L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    except Exception as e:
        return path
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L)

def copy_files(src_dst_tup):
    src, dst = src_dst_tup
    src = 'D:/Projects/omnidata_starter_dataset/rgb/' + src
    dst = 'D:/Projects/omnidata_starter_dataset/rgb/' + dst
    shutil.copy(src, dst)
    return 

if __name__=='__main__':
    os.chdir('D:/Projects/omnidata_starter_dataset/rgb')
    # dim = Image.open('point_4_view_3_domain_depth_euclidean.png', 'r')
    # path = 'point_4_view_3_domain_rgb.png'
    # print(bright(DIR+path))

    
    pool = mp.Pool(8)
    di = {}
    classes = {}
    kmeans = KMeans(n_clusters=3)
    for DIR in [['replica/frl_apartment_0/', 'taskonomy/allensville/'][0]]:
        df = pd.DataFrame([])
        df['dirs'] = [DIR + x for x in os.listdir(DIR)]
        df['luminosity'] = [x if type(x) is not str else -1 for x in pool.map(bright, df['dirs'])]
        
        df.loc[df['luminosity']!=-1, 'class'] = kmeans.fit_predict(np.array(df[df['luminosity']!=-1]['luminosity']).reshape(-1, 1))
        df.loc[df['luminosity']!=-1, 'new_dir'] = df[df['class'].notna()].apply(lambda x: '/'.join(x['dirs'].split('/')[0:2] + ['brightness_groups', str(int(x['class']))]), axis=1)
        sns.scatterplot(x=df.loc[df['luminosity']!=-1, 'luminosity'], y=df.loc[df['luminosity']!=-1, 'class']).set(title=f"Scatterplot by Class of {DIR}")
        plt.show()
        sns.histplot(df.loc[df['luminosity']!=-1, 'luminosity']).set(title=f"Luminosity Distribution of {DIR}")
        plt.show()
        [os.makedirs('D:/Projects/omnidata_starter_dataset/rgb/' + x) for x in df['new_dir'].unique() if type(x) is not float]
        
        # pool.map(copy_files, df.loc[df['class'].notna(), ['dirs', 'new_dir']].to_numpy())
            
        # df.to_csv('D:/Projects/omnidata_starter_dataset/'+DIR.split('/')[-1]+'.csv', encoding='utf-8', index=False)
        print(df['class'].value_counts())
        print(df.dropna().set_index('class').mean(level=0))
        break
    
    pool.close()
    '''
    sns.histplot([x for x in li if type(x) is not str])
    plt.show()
    '''
