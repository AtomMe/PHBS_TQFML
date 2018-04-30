
# coding: utf-8

import numpy as np

# dataframes in python
import pandas as pd



import matplotlib.pyplot as plt

#defaults
plt.rcParams['figure.figsize'] = (20.0, 20.0)
plt.rcParams.update({'font.size': 10})
plt.rcParams['xtick.major.pad']='5'
plt.rcParams['ytick.major.pad']='5'
plt.style.use('ggplot')


# #### Data preparation
# We cache the data set from the above mentioned repository in a local directory. 

verify_sample=pd.read_csv("./data/verify_sample.csv")
model_sample = pd.read_csv("./data/model_sample.csv")
test_result = verify_sample.iloc[:,[0,1]]
verify_sample = verify_sample.drop('y', axis=1)


# #### Convert the data
# We use pandas to read the data from its original excel format into a dataframe

# In[5]:

df = model_sample
df = df.fillna(0)
verify_sample = verify_sample.fillna(0)


# #### Clean up
# We lowercase the column name, and rename the column names when required,
# In particular, remarkably this dataset misses a colum `PAY_1`. In the analysis here below we assume that PAY_0 is actually pay_1, to be consider the repayment of the month prior to the month where we calculate the defaulting (which is October 2005, in this particular dataset)

df = df.drop('user_id', axis=1)

# ### Feature engineering
# 
# It's not about blind feature conversion to values between 0 and 1, it's about understanding data. In this case we see that money they exibits a log/log distribution, so first off, we are going to take the log of the money.


# help func
# select those need log
def selcols():
    colnames = []
    colnames.extend(['x_041','x_043','x_044','x_045','x_046','x_047'])
    colnames.extend(['x_052','x_053','x_054','x_055','x_057','x_058','x_059'])
    colnames.extend(['x_060','x_061','x_064','x_067'])
    colnames.extend(['x_070','x_073','x_078','x_079'])
    colnames.extend(['x_080','x_085','x_086','x_087'])
    colnames.extend(['x_108','x_111','x_114','x_117','x_120','x_125','x_126','x_127'])
    colnames.extend(['x_130','x_131','x_133','x_135','x_136','x_138'])
    colnames.extend(['x_140','x_141','x_143','x_145','x_146','x_147','x_148'])
    colnames.extend(['x_159','x_160','x_161','x_172','x_173','x_174'])
    colnames.extend(['x_185','x_186','x_187'])
    return colnames

# generate the new dataset by log
colindex = selcols()
for col in colindex:
    df[col] = df[col].apply(lambda x: np.log(x+1) if x>0 else 0) 
    verify_sample[col] = verify_sample[col].apply(lambda x: np.log(x+1) if x>0 else 0)


# ### Seperate the labels and input variables


y = df.iloc[:, 0]
X = df.iloc[:,1:]
verify_X = verify_sample.iloc[:,1:]


# #### Feature scaling

# three diffrent methods
#- MinMaxScaler
#- StandardScaler N(0,1)
#- RobustScaler avoid outliers.

#- X_test's standardization should use the mean and std of X_train

def MinMaxNorm(X_train,X_test):
    from sklearn.preprocessing import MinMaxScaler

    mms = MinMaxScaler().fit(X_train)
    X_train_norm = mms.transform(X_train)
    X_test_norm = mms.transform(X_test)
    return X_train_norm,X_test_norm

def StdNorm(X_train,X_test):
    from sklearn.preprocessing import StandardScaler

    stdsc = StandardScaler().fit(X_train)
    X_train_std = stdsc.transform(X_train)
    X_test_std = stdsc.transform(X_test)
    return X_train_std,X_test_std

def RobustNorm(X_train,X_test):
    from sklearn.preprocessing import RobustScaler

    rbs = RobustScaler().fit(X_train)
    X_train_std = rbs.transform(X_train)
    X_test_std = rbs.transform(X_test)
    return X_train_std,X_test_std



X_prep, verify_X_test_prep = RobustNorm(X, verify_X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)
#X_train = X_prep
#y_train = y


# ### 对数据不平衡进行处理

# oversampling
def data_resample(X_train,y_train):
    from sklearn.utils import resample
    X_upsampled, y_upsampled = resample(X_train[y_train == 1],
                                    y_train[y_train == 1],
                                    replace=True,
                                    n_samples=X_train[y_train == 0].shape[0])
    X_bal = np.vstack((X_train[y_train == 0], X_upsampled))
    y_bal = np.hstack((y_train[y_train == 0], y_upsampled))
    X_train = X_bal
    y_train = y_bal
    return X_train,y_train



# #### Feature selection

# ### Models

# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, f1_score

feat_labels = df.columns[1:]

forest = RandomForestClassifier(n_estimators=500, min_samples_split=5,min_samples_leaf =2,
                                max_features = 'log2',n_jobs=4,criterion='gini')

forest.fit(X_train, y_train)
importances = forest.feature_importances_

# #### Random Forest 
# Quite popular a few years back, bootstrap aggregating ensamble of decision trees


def f1_score_set(rf,X_test,y_test):
    thresholds = np.linspace(0.2,0.3,100)
    #y_train_pred = rf.predict_proba(X_train)[:,1]
    y_test_pred = rf.predict_proba(X_test)[:,1]
    score_set = []
    index_threshold = 0
    sum_score = 0
    for threshold in thresholds:
        y_test_pred1  = (y_test_pred > threshold).astype(int)
    
        score = f1_score(y_test, y_test_pred1)
        score_set.append(score)
    for i in range(95):
        temp = sum(score_set[i:i+5])
        if temp > sum_score:
            sum_score = temp
            index_threshold = i+2
    
    return thresholds[index_threshold],sum_score/5


# In[44]:

indices = np.argsort(importances)[100:]

#X_train,y_train = data_resample(X_train,y_train)


X_train = X_train[:,indices]
X_test = X_test[:,indices]
verify_X_test_prep = verify_X_test_prep[:,indices]


from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
rf = RandomForestClassifier(n_estimators=500, min_samples_split=5,min_samples_leaf =2,max_features = 'log2',n_jobs=4,criterion='gini')
rf.fit(X_train,y_train)



best_f1_threshold,f1 = f1_score_set(rf,X_test,y_test)

print best_f1_threshold,f1
# ### 预测测试集

# In[33]:

threshold = best_f1_threshold
verify_y_test_pred = rf.predict_proba(verify_X_test_prep)[:,1]
user_id = verify_sample.loc[:,['user_id']]
y_prediction  = (verify_y_test_pred > threshold).astype(int)
predict_result = pd.DataFrame(y_prediction,columns=['y_prediction'])
predict_result = user_id.join(predict_result)
p_y = predict_result['y_prediction'].values
real_y = test_result['y'].values


# In[34]:

# 预测结果的f1值
score = f1_score(real_y, p_y)

print score


