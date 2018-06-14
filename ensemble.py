
# coding: utf-8

# In[1]:


import pandas as pd
output = pd.read_csv('sample_submission.csv')


# In[4]:


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


# In[5]:


from os import listdir
test_path="audio_test/"
test_audio_files = listdir(test_path)


# In[6]:


import numpy as np
test_276=np.load('xgb_processed__276.npy')
test_192=np.load('xgb_processed__192.npy')
test_132=np.load('xgb_processed_test.npy')


# In[19]:


import numpy as np
test_audio_files = np.load('test_list.npy')


# In[20]:


out_dict=dict()
num=0
for item in test_audio_files:
    out_dict[item]=num
    num=num+1


# In[9]:


import pickle
clf_132 = pickle.load(open("xgb_132.pickle.dat", "rb"))
clf_192= pickle.load(open("xgb_192.pickle.dat", "rb"))
clf_276= pickle.load(open("xgb_276.pickle.dat", "rb"))


# In[10]:


predict_132=clf_132.predict_proba(test_132)
predict_192=clf_192.predict_proba(test_192)
predict_276=clf_276.predict_proba(test_276)


# In[12]:


predict_132


# In[13]:


import numpy as np
test_audio_files = np.load('test_list.npy')


# In[33]:


predict_132_final=[]
predict_192_final=[]
predict_276_final=[]
for i in range(len(output)):
    name=output['fname'][i]
    predict_132_final.append(predict_132[out_dict[name]])
    predict_192_final.append(predict_192[out_dict[name]])
    predict_276_final.append(predict_276[out_dict[name]])


# In[34]:


predict=[]
for i in range(len(predict_132)):
    temp=[]
    for j in range(0,41):
        new=predict_132_final[i][j]+predict_192_final[i][j]+predict_276_final[i][j]
        new=new/3.0
        temp.append(new)
    predict.append(temp)


# In[35]:


def proba2labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids


# In[36]:


ans,ids=proba2labels( predict,idx_to_label,3)   


# In[37]:


out = pd.DataFrame()
out['fname'] = output['fname']
out['label']= ans


# In[38]:


out.head()


# In[39]:


out.to_csv('xgb_ensemble.csv', index=False)

