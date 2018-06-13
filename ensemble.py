
# coding: utf-8

# In[1]:


import pickle
clf_132 = pickle.load(open("xgb_132.pickle.dat", "rb"))
clf_192= pickle.load(open("xgb_192.pickle.dat", "rb"))
clf_276= pickle.load(open("xgb_276.pickle.dat", "rb"))


# In[3]:


import numpy as np
test_132=np.load('xgb_processed_test_132.npy')
test_192=np.load('xgb_processed_test_192.npy')
test_276=np.load('xgb_processed_test_276.npy')

# In[4]:


predict_132=clf_132.predict_proba(test_132)
predict_192=clf_192.predict_proba(test_192)
predict_276=clf_276.predict_proba(test_276)


# In[11]:


predict=[]
for i in range(len(predict_132)):
    temp=[]
    for j in range(0,41):
        new=predict_132[i][j]+predict_192[i][j]+predict_276[i][j]
        new=new/3.0
        temp.append(new)
    predict.append(temp)


# In[14]:


len(predict[0])


# In[15]:


import pandas as pd
train = pd.read_csv("train.csv")
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


# In[16]:


def proba2labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids


# In[17]:


ans,ids=proba2labels(predict,idx_to_label,3)   


# In[20]:


from os import listdir
test_path="audio_test/"
test_audio_files = listdir(test_path)


# In[21]:


out = pd.DataFrame()
out['fname'] = test_audio_files
out['label']= ans


# In[22]:


out.head()


# In[23]:


out.to_csv('xgb_ensemble.csv', index=False)

