
# coding: utf-8

# In[57]:


import numpy as np
np.random.seed(500)
import os
import pandas as pd
from scipy.io import wavfile
import librosa
from os import listdir
from tqdm import tqdm, tqdm_pandas
from scipy.stats import skew


# In[2]:


train = pd.read_csv("train.csv")


# In[74]:


train_path="audio_train/"
test_path="audio_test/"

train_audio_files = listdir(train_path)
test_audio_files = listdir(test_path)
output = pd.read_csv('sample_submission.csv')


# In[66]:


#給label數值
labels = list(train.label.unique())
label_to_idx=dict()
idx_to_label=dict()
idx=0
for label in labels:
    label_to_idx[label]=idx
    idx=idx+1
for key in label_to_idx:
    idx_to_label[label_to_idx[key]]=key


# In[23]:


SAMPLE_RATE= 44100

#參考https://www.kaggle.com/thebrownviking20/xgb-using-lda-and-mfcc-opanichev-s-features/code做修改
def get_mfcc(name, path):
   
    b, _ = librosa.core.load(path + name, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        print('processing')
        b=b - np.mean(b)
        mfcc = librosa.feature.mfcc(b, sr = SAMPLE_RATE, n_mfcc=20)
        contrast = librosa.feature.spectral_contrast(b)[0]
        zcr=librosa.feature.zero_crossing_rate(b)[0]
        centroid=librosa.feature.spectral_centroid(b)[0]
        rolloff = librosa.feature.spectral_rolloff(b)[0]
        
        
        #參考https://www.mathworks.com/help/audio/ref/mfcc.html
        #Change in coefficients from one frame of data to another, returned as an L-by-M matrix or an L-by-M-by-N array. The delta array is of the same size and data type as the coeffs array.
        delta_mfcc  = librosa.feature.delta(mfcc)
        mfcc_f=np.hstack((np.mean(mfcc , axis=1), np.std(mfcc , axis=1), skew(mfcc , axis = 1), np.median(mfcc , axis = 1)))
        contrast_f=np.hstack((np.mean(contrast), np.std(contrast), skew(contrast), np.median(contrast)))
        zcr_f=np.hstack((np.mean(zcr), np.std(zcr), skew(zcr), np.median(zcr)))
        centroid_f=np.hstack((np.mean(centroid), np.std(centroid), skew(centroid), np.median(centroid)))
        rolloff_f=np.hstack((np.mean(rolloff), np.std(rolloff), skew(rolloff), np.median(rolloff)))
        
        #get intersection mfcc features
        #source: https://www.kaggle.com/miklgr500/catboost-mfcc/code
        length_mfcc = len(mfcc) 
        step = length_mfcc//(10)

        for i in range(step, length_mfcc, step):
            fmfcc = mfcc[:][i-step:i+step]
            fdmfcc=delta_mfcc[:][i-step:i+step]
            inter_mfcc=np.hstack((np.mean(fmfcc,axis=1),np.std(fmfcc,axis=1),np.min(fmfcc,axis=1),np.max(fmfcc,axis=1),np.median(fmfcc,axis=1),skew(fmfcc,axis=1)))
            inter_dmfcc=np.hstack((np.mean(fdmfcc,axis=1),np.std(fdmfcc,axis=1),np.min(fdmfcc,axis=1),np.max(fdmfcc,axis=1),np.median(fdmfcc,axis=1),skew(fdmfcc,axis=1)))

        output= np.hstack((mfcc_f,contrast_f,zcr_f, centroid_f,rolloff_f,inter_mfcc,inter_dmfcc))   
        
        print(len(output))
        return pd.Series(output)
    except:
        print('bad file')
        return pd.Series([0]*144)


# In[24]:



train_data = pd.DataFrame()
train_data = train['fname'].apply(get_mfcc, path=train_path)


# In[31]:


np.save('xgb_processed_train_132.npy',train_data)


# In[36]:


train_data=np.load('xgb_processed_train_132.npy')


# In[37]:


len(train_data[0])


# In[48]:


label=[]
for item in train["label"]:
    label.append(label_to_idx[item])


# In[49]:


from xgboost import XGBClassifier


# In[50]:


train_x=train_data[0:int(len(train_data)*0.9)]
train_y=label[0:int(len(label)*0.9)]
valid_x=train_data[int(len(train_data)*0.9):len(train_data)]
valid_y=label[int(len(label)*0.9):len(label)]


# In[51]:


clf =XGBClassifier(max_depth=8, learning_rate=0.03, n_estimators=3500,
                    n_jobs=-1, random_state=0, reg_alpha=0.2, 
                    colsample_bylevel=0.9, colsample_bytree=0.9)
clf.fit(train_x, train_y)


# In[52]:


import pickle
pickle.dump(clf, open("xgb_132.pickle.dat", "wb"))


# In[53]:


from sklearn.metrics import accuracy_score


# In[54]:


# make predictions for test data
y_pred = clf.predict(valid_x)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(valid_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


clf.fit(train_data, label)


# In[ ]:


pickle.dump(clf, open("xgb_132.pickle.dat", "wb"))


# In[ ]:


test_data = pd.DataFrame()
test_data['fname'] = test_audio_files

test_data = test_data['fname'].apply(get_mfcc, path=test_path)


# In[ ]:


np.save('xgb_processed_test_132.npy',test_data)


# In[ ]:


test_data.head()


# In[80]:


test_data=np.load('xgb_processed_test_132.npy')


# In[81]:


len(test_data)


# In[82]:


def proba2labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids


# In[83]:


predict=clf.predict_proba(test_data)
ans,ids=proba2labels(predict,idx_to_label,3)   


# In[84]:


len(test_audio_files)


# In[85]:


out = pd.DataFrame()
out['fname'] = test_audio_files
out['label']= ans


# In[86]:


out.head()


# In[87]:


out.to_csv('xgb132.pickle.dat.csv', index=False)


# In[63]:


answer=[]
for p in predict:
    sorted_result=np.argsort(p)[::-1]
    answer.append(' '.join([idx_to_label[sorted_result[1]],idx_to_label[sorted_result[2]],idx_to_label[sorted_result[3]]]))

