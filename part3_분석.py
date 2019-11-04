#!/usr/bin/env python
# coding: utf-8

# # part3. 분석(각 제품군 별 수요 특징 및 재고전략)
# 

# ## 1. 제품군 정의

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plot 에서 한글 사용시 깨지지 않게 하기
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/나눔고딕코딩.ttf").get_name()
rc('font', family=font_name)


# In[ ]:


# 사용자의 path에 맞게 정의
path = ''


# In[2]:


cvs = pd.read_csv(path + 'korea_cvs.csv')
cvs.columns = [c.split('.')[-1] for c in cvs.columns]
cvs['sale_dt'] = pd.to_datetime(cvs['sale_dt'].astype(str), format = '%Y%m%d')
cvs.set_index('sale_dt', inplace = True)


# In[4]:


# quantity 확인
cvs.groupby('category').sum()


# In[13]:


# 월별 coefficient of variance 확인
cvs_m = cvs.groupby([cvs['category'],[i.month for i in cvs.index]])['adj_qty'].sum()
cvs_m.unstack(0).apply(lambda x : np.var(x) / np.mean(x))


# ## 2. 제품 수요 별 유의한 날씨 feature 파악
# 
# ### 2가지 test
# - correlation analysis
# - cointegration test

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 3. 수요 예측

# ## 아이스크림 수요 예측
# - 아이스크림 수요를 기반으로 수요 예측 방식 진행
# - 탐색 모델 
#     - 1. winter's method (X)
#         - STL 분해 결과 기본 가정인 확정적 추세, 계절성이 아니므로 사용 X
#     - 2. seasonal ARIMA (O)
#         - 확률적 추세, 계절성 반영을 위해 사용
#     - 3. RNN(LSTM) (O)
#         - 날씨 정보 반영, 비선형 탐지

# ### 아이스크림 수요 특징 파악
# - 기온(최대기온, max_ta) 와 판매량 scatter plot
# - 판매량과 기온의 시간에 따른 추이 비교

# In[4]:


path = 'C:/python/2019_weather_bigcontest/data_origin/'


# In[7]:


product = '아이스크림'

w = pd.read_csv(path + 'bigcon_weather.csv')
c = pd.read_csv(path + 'korea_cvs.csv')

for df in [w,c]:
    df.columns = [c.split('.')[-1] for c in df.columns]
w['tm'] = pd.to_datetime(w['tm'].astype(str), format = '%Y%m%d')
c['sale_dt'] = pd.to_datetime(c['sale_dt'].astype(str), format = '%Y%m%d')

w.set_index('tm', inplace = True)
c.set_index('sale_dt', inplace = True)

w_place, c_place = w.copy(), c.copy()

w_place_wk, c_place_wk = w_place.resample('W-MON').mean().drop('stn_id',axis = 1) , c_place.groupby('category').resample('W-MON')['adj_qty'].sum().unstack(0)
w_place_day, c_place_day = w_place.resample('D').mean().drop('stn_id', axis = 1), c_place.groupby('category').resample('D')['adj_qty'].sum().unstack(0)

data_wk, data_day = pd.concat([w_place_wk, c_place_wk], axis = 1), pd.concat([w_place_day, c_place_day], axis = 1)


# In[8]:


plt.scatter(data_day['max_ta'], data_day[product])
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel('max_ta', fontsize = 14)
plt.ylabel(product, fontsize = 12)
plt.title('{}, {} scatter plot'.format('max_ta',product) , fontsize = 18)


# In[9]:


input_feat = data_wk.corr()[product].drop(product).apply(np.abs).sort_values().index[-3:]
input_feat

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler().fit(data_wk[[product] + list(input_feat)])
pd.DataFrame(minmax.transform(data_wk[[product] + list(input_feat)]), index = data_wk.index, columns = [product] + list(input_feat)).plot(figsize = (20,6))
plt.title('기온과 {} 판매량 추이 비교(minmax scaling)'.format(product), fontsize = 30)
plt.legend(fontsize = 17)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)


# ## 탐색 모델 1. Winter's Method
# 엑셀로 작업하여 해당 코드는 없음

# In[3]:


from IPython.display import Image


# ### 탐색 모델 2. seasonal ARIMA
# - 모델 사용을 위한 전처리
#     - 정상성으로 변형
# - 모수 추청
#     - 1. acf, pacf를 통한 1차적 추정
#     - 2. auto.arima() function 을 통한 모수 최적화
# - 예측(시나리오 방식)
# 
# 
# 

# In[6]:


Image("image_file/seasonal ARIMA.PNG", width = 800, height = 180) 


# ## 탐색 모델 3. RNN(LSTM)
# - 기온 데이터를 구간화 하여 feature 생성
# - 예측(시나리오 방식)

# In[7]:


Image("image_file/RNN.PNG", width = 800, height = 180) 


# In[15]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, glob
import keras, sklearn
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras.backend as K
from keras.callbacks import EarlyStopping 

matplotlib.rcParams['axes.unicode_minus'] = False

# plot 에서 한글 사용시 깨지지 않게 하기
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/나눔고딕코딩.ttf").get_name()
rc('font', family=font_name)

# 모델 성능 평가지표 함수 구축
def RMSE(y_true, y_pred):
    return np.mean( (y_true - y_pred) ** 2 ) ** (1/2)
def ME(y_true, y_pred):
    return np.mean(y_true - y_pred)
def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def MPE(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[2]:


w = pd.read_csv(path + 'bigcon_weather.csv') # 날씨 데이터
c = pd.read_csv(path + 'korea_cvs.csv') # 편의점 판매량 데이터

for df in [w,c]:
    df.columns = [c.split('.')[-1] for c in df.columns]
w['tm'] = pd.to_datetime(w['tm'].astype(str), format = '%Y%m%d')
c['sale_dt'] = pd.to_datetime(c['sale_dt'].astype(str), format = '%Y%m%d')

w.set_index('tm', inplace = True)
c.set_index('sale_dt', inplace = True)
w_place, c_place = w.copy(), c.copy()

w_place_wk, c_place_wk = w_place.resample('W-MON').mean().drop('stn_id',axis = 1) , c_place.groupby('category').resample('W-MON')['adj_qty'].sum().unstack(0)
w_place_day, c_place_day = w_place.resample('D').mean().drop('stn_id', axis = 1), c_place.groupby('category').resample('D')['adj_qty'].sum().unstack(0)

data_wk, data_day = pd.concat([w_place_wk, c_place_wk], axis = 1), pd.concat([w_place_day, c_place_day], axis = 1)


# In[3]:


data_wk.head()


# In[4]:


product = '아이스크림'
data_wk[product].plot(figsize = (15,5))
plt.title('주별 {} 판매량(전체) 2016~2018년'.format(product), fontsize = 20)
plt.xticks(fontsize = 15)
plt.show()

data_day[product].plot(figsize = (15,5))
plt.show()


# In[6]:


input_feat = data_wk.corr()[product].drop(product).apply(np.abs).sort_values().index[-3:]
input_feat

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler().fit(data_wk[[product] + list(input_feat)])
pd.DataFrame(minmax.transform(data_wk[[product] + list(input_feat)]), index = data_wk.index, columns = [product] + list(input_feat)).plot(figsize = (20,6))
plt.title('기온과 {} 판매량 추이 비교(minmax scaling)'.format(product), fontsize = 30)
plt.legend(fontsize = 17)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)


# ### train, test set 부분 시각화

# In[ ]:


# 시간에 따라 train, test로 나누기
print(data_wk.shape , '\n')

data_cut = '2018-01'

train, test = data_wk[data_wk.index < data_cut], data_wk[data_wk.index >= data_cut]
print(train.shape)
print(test.shape)


# In[12]:


fig = plt.figure(figsize = (20,6))

plt.plot(data_wk[data_wk.index < data_cut][product], color = 'C0', label = 'Train set')
plt.plot(data_wk[data_wk.index >= data_cut][product], color = 'C1', label = 'Test set')
plt.xticks(fontsize = 16)
plt.legend(fontsize = 18)


# ### 기온 max_ta 에 따른 아이스크림의 판매량에 대한 feature추가

# In[13]:


# 기온과 판매량 scatter plot(일별)
plt.scatter(data_day['max_ta'], data_day[product])
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel('max_ta', fontsize = 14)
plt.ylabel(product, fontsize = 12)
plt.title('{}, {} scatter plot'.format('max_ta',product) , fontsize = 18)


# In[14]:


# 기온과 판매량 scatter plot(주별, train set)
data_cut = '2018-01'
data_wk_cut = data_wk[data_wk.index < data_cut]
plt.scatter(data_wk_cut['max_ta'], data_wk_cut[product])


# In[15]:


# 구간화 및 correlation feature 생성
minval, maxval = data_wk_cut['max_ta'].min(), data_wk_cut['max_ta'].max()
cut = [minval-5,  10, 25, maxval + 5]

cut_coeff = data_wk_cut[['아이스크림']].groupby(pd.cut(data_wk_cut['max_ta'], cut)).corr(data_wk_cut['max_ta'])

data_wk['max_ta_coeff'] = pd.cut(data_wk['max_ta'], cut, labels = cut_coeff.values.reshape(-1))
data_wk['max_ta_coeff'] = data_wk['max_ta_coeff'].astype(float)
data_wk['max_ta_coeff_prod'] = (data_wk['max_ta'] - minval).multiply(data_wk['max_ta_coeff'])


# 차분 값 생성
data_wk['{}_diff'.format(product)] = data_wk[product].diff(1)
cut_coeff


# In[17]:


# correlation 확인
data_wk[['max_ta','max_ta_coeff','max_ta_coeff_prod','아이스크림']].corr()


# ### RNN(LSTM) 모델링

# In[19]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import copy


# weather feature : 최대 기온, 구간화 한 correlation, 앞의 두 값을 곱한 값
w_feature = ['max_ta', 'max_ta_coeff', 'max_ta_coeff_prod'] 
# product feature : 과거 수요량, 과거 수요의 차분값
product_feature = [product, '{}_diff'.format(product)]

feature = w_feature + product_feature

# 몇 시점 후를 예측할지 정해준다.
predict_term = 1
data_wk_shift = pd.concat([ data_wk[w_feature].shift(-predict_term), data_wk[product_feature] ],axis = 1)
data_wk_shift.dropna(how = 'any', axis = 0, inplace = True)


print(data_wk_shift.shape , '\n')

data_cut = '2018-01'

train, test = data_wk_shift[data_wk_shift.index < data_cut], data_wk_shift[data_wk_shift.index >= data_cut]

train_index = train.index
test_index = test.index
print(train.shape)
print(test.shape)

# train, test 를 numpy array 로 변경
X_train, X_test = train[feature].values,  test[feature].values

# y target 값을 추출하여 y로 설정
y_train, y_test  = train[product].values.reshape(-1,1), test[product].values.reshape(-1,1)

# y값에 대한 scaler pipe
y_scaler = Pipeline([('minmax',MinMaxScaler((0,1.0)))])

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

# X값에 대한 scaler pipe
my_scaler_pipe = Pipeline([('minmax',MinMaxScaler((-0.5,0.5))), ('standard',StandardScaler())])


#################################################################################################################

# RNN용 데이터를 만드는 함수 X : (sample, time step, feature), y : (sample, target value)
def createDataset(X_train, X_test, y_train, y_test, look_back = 14, predict_term = 1, scaler_pipe = None, pca_range = None ,
                  pca_target_var = True, outlier_upper = None, scale_target_var = True):
    X_train_ts, X_test_ts, y_train_ts, y_test_ts = [], [], [], []
    
    # 과거 것을 update해준다.
    y_test = np.concatenate((y_train[-look_back-predict_term+1:],y_test), axis = 0)
    X_test = np.concatenate((X_train[-look_back-predict_term+1:],X_test), axis = 0)

    if scaler_pipe != None:
        print('scaling X variables using assigned scaler pipe instances')
        if scale_target_var == True:
            
            scaler_pipe.fit(X_train)

            X_train = scaler_pipe.transform(X_train)
            X_test = scaler_pipe.transform(X_test)
        else : 
            print("... doesn't scaling target variable in X")
            scaler_pipe.fit(X_train[:,:-1])
            
            X_train = np.concatenate((scaler_pipe.transform(X_train[:,:-1]), X_train[:,-1].reshape(-1,1)), axis = 1)
            X_test = np.concatenate((scaler_pipe.transform(X_test[:,:-1]), X_test[:,-1].reshape(-1,1)), axis = 1)
    if pca_range != None:
        print('using PCA')
        pca = PCA()
        if pca_target_var == True:
            pca.fit(X_train[:,:-1])

            pca_num = 0
            while pca.explained_variance_ratio_.cumsum()[pca_num] < pca_range:
                pca_num +=1
            print('using PC to {} ( explained_ratio_cumsum : {:.2f})'.format(pca_num, pca.explained_variance_ratio_.cumsum()[pca_num]))
            X_train = np.concatenate([pca.transform(X_train[:,:-1])[:,:pca_num], X_train[:,-1].reshape(-1,1)], axis= 1)
            X_test = np.concatenate([pca.transform(X_test[:,:-1])[:,:pca_num], X_test[:,-1].reshape(-1,1)], axis= 1)
        else:
            pca.fit(X_train)
            pca_num = 0
            while pca.explained_variance_ratio_.cumsum()[pca_num] < pca_range:
                pca_num +=1
            print('using PC to {} ( explained_ratio_cumsum : {:.2f})'.format(pca_num, pca.explained_variance_ratio_.cumsum()[pca_num]))
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

    
    if outlier_upper != None: 
        print('outlier cut using assigned outlier cut')
        y_train = np.where(y_train > outlier_upper, outlier_upper, y_train)
        
    for i in range(len(X_train)- look_back-predict_term+1):
        X_train_ts.append(X_train[i:(i+look_back)])
        y_train_ts.append(y_train[i+look_back+predict_term-1])
        #y_train_ts.append(y_train[i+1 :(i+look_back+1)])
        
    for i in range(len(X_test)- look_back-predict_term+1):
        X_test_ts.append(X_test[i:(i+look_back)])
        y_test_ts.append(y_test[i+look_back+predict_term-1])
        #y_test_ts.append(y_test[i+1 : (i+look_back+1)])
        
    
        
    return np.array(X_train_ts), np.array(X_test_ts), np.array(y_train_ts), np.array(y_test_ts)


#################################################################################################################
# 과거 몇 시점의 정보를 반영할지 정한다.
look_back = 12

X_train_ts, X_test_ts, y_train_ts, y_test_ts = createDataset(X_train, X_test, y_train, y_test, look_back = look_back, predict_term = predict_term,
                                                             scaler_pipe = my_scaler_pipe, pca_range = None, outlier_upper = None, pca_target_var = False)

# validation set을 만들어준다.
val_cut = int(len(train) * 0.04)
X_val_ts, y_val_ts = X_train_ts[-val_cut:], y_train_ts[-val_cut:]
X_train_ts, y_train_ts = X_train_ts[:-val_cut], y_train_ts[:-val_cut]


# data set의 형태를 확인
print(X_train_ts.shape)
print(y_train_ts.shape)
print(X_val_ts.shape)
print(y_val_ts.shape)
print(X_test_ts.shape)
print(y_test_ts.shape)


# In[22]:


# RNN 모델링 structure 코드
import keras
from keras import optimizers, initializers
from keras.layers import LSTM
from keras.layers import ELU, LeakyReLU
import tensorflow as tf

tf.set_random_seed(1)

# 데이터 형태 한번 더 확인
for tr, te in zip([X_train_ts, X_val_ts, X_test_ts], [y_train_ts, y_val_ts, y_test_ts]):
    print(tr.shape)
    print(te.shape)

K.clear_session()

size, ts, feat= X_train_ts.shape


# modeling structure
model = Sequential()
model.add(LSTM(size ,input_shape = (ts,feat), activation = 'relu', recurrent_dropout = 0.0, dropout = 0.1))
                    
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'logcosh')

model.summary()


# In[23]:


# 세부 옵션

## early stopping 
### validation set에 대해 loss를 추적, 70번 까지 loss가 증가하는 것을 허용, epoch 중 validation loss가 가장 작았던 weight를 저장
early_stop = EarlyStopping(monitor='val_loss', patience= 70, verbose=1, restore_best_weights = True)

model.fit(X_train_ts, y_train_ts ,epochs= 1000, validation_data=(X_val_ts, y_val_ts), shuffle = False, 
          batch_size =6,  verbose = 1, callbacks = [early_stop] )


# train set에 대해 actual, prediction 데이터 프레임을 만든다.
train_result = pd.DataFrame({'actual' : y_scaler.inverse_transform(y_train_ts).reshape(-1), 'prediction' : y_scaler.inverse_transform(model.predict(X_train_ts)).reshape(-1)},
                           index = train_index[look_back+predict_term-1:look_back+predict_term-1+X_train_ts.shape[0]])


# In[22]:


train_result.plot(figsize = (15,8))


# In[23]:


X_test_ts.shape


# ### 시나리오를 적용하여 예측
# - test set
# - 12주차 정보로 1 시점 후 예측, 8주 후 확보된 판매량 데이터로 모델 update

# In[29]:


i = 0
i_to = 0
update_term = 8
# to_step = 30
test_result = {'actual' : y_test_ts.reshape(-1), 'prediction' : []}

print('====== model update 주기 : {} time stamps ==========='.format(update_term) + '\n')
while i_to < len(y_test_ts):
    i_to = min(i+update_term, len(y_test_ts))
    # i ~ i_to 까지 예측(해당 시점 전 수요 예측을 위함)
    print('{} ~ {} 까지 예측'.format(i, i_to-1) )
    if update_term > 2:
        test_result['prediction'].extend(model.predict( X_test_ts[i:i_to]).reshape(-1))
    else:
        test_result['prediction'].append(model.predict( X_test_ts[i:i_to])[0][0])    
        
    # 해당 시점까지 지난 후 실제 발생한 actual data를 다시 학습(model update)
    
    cur_test_X = np.concatenate((X_train_ts, X_test_ts[:i_to]), axis = 0)
    cur_test_y = np.concatenate((y_train_ts, y_test_ts[:i_to]), axis = 0)
    
    print('해당 시간이 지난 후 model update')
    print('학습 data shape : {}'.format(cur_test_X.shape) + '\n')
    ####===================================================================
    K.clear_session()

    size, ts, feat= cur_test_X.shape

    model = Sequential()
    model.add(LSTM(size ,input_shape = (ts,feat), activation = 'relu', recurrent_dropout = 0.0, dropout = 0.1))
                        #,return_state = True, return_sequences = True,  dropout = 0.1))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'logcosh')
    early_stop = EarlyStopping(monitor = 'val_loss', patience= 80, verbose=0, restore_best_weights = True)
    model.fit(cur_test_X, cur_test_y ,epochs= 1000, validation_split = 0.2,  shuffle = True, 
          batch_size =5,  verbose = 0, callbacks = [early_stop] )
    ###======================================================================

    i = i_to
#     print('{} complete'.format(i))
    
# 여기서 스케일링으로 y값을 다시 바꿔준다.


# In[30]:


# test 구간에 대해 actual, prediction의 길이가 같은지 확인
print(len(test_result['actual']))
print(len(test_result['prediction']))


# test 구간에 대해 데이터 프레임과 날짜를 붙여준다.
test_result = pd.DataFrame(test_result, index = test_index)    
test_result = pd.DataFrame(y_scaler.inverse_transform(test_result), columns = test_result.columns, index = test_index)


# ### 시나리오를 적용한 RNN(LSTM)의 성능

# In[32]:


import warnings
warnings.filterwarnings('ignore')

# 모델에 붙일 이름 (몇 시점 전 정보까지 반영할지, 몇 시점 후를 예측할지, update 기간)
naming = (look_back, predict_term, update_term)

fig, ax = plt.subplots(2,1, figsize = (18, 14))
# test_result = pd.DataFrame({'actual' : y_scaler.inverse_transform(y_test_ts).reshape(-1), 'prediction' : y_scaler.inverse_transform(model.predict(X_test_ts)).reshape(-1)})
# test_result = pd.DataFrame(test_result)
train_perform = []
test_perform = []
for func in [RMSE, ME, MAE, MPE, MAPE]:
    train_perform.append(func(train_result['actual'], train_result['prediction']))
    test_perform.append(func(test_result['actual'], test_result['prediction']))


train_result.plot( color = ['C0','C1'], ax = ax.ravel()[0])
ax.ravel()[0].set_title('''==== Train data :  RMSE | ME | MAE | MPE | MAPE === 
      prediction : {:.3f} | {:.3f} | {:.3f} | {:.3f} % | {:.3f} %'''.format(train_perform[0], train_perform[1], train_perform[2], train_perform[3], train_perform[4]), fontsize = 18)
# plt.show()      

test_result.plot(color = ['C0','C1'], style = ['-o','--o'], ax = ax.ravel()[1] )

ax.ravel()[1].set_title('''==== Test data :  RMSE | ME | MAE | MPE | MAPE === 
      prediction : {:.3f} | {:.3f} | {:.3f} | {:.3f} % | {:.3f} %'''.format(test_perform[0], test_perform[1], test_perform[2], test_perform[3], test_perform[4]), fontsize = 18)
# plt.show()                

fig.suptitle('Performance Plot Of {} Forecasting Model_{}'.format(product, naming), fontsize = 23)


# ### machine learning model 과 앙상블

# ### 머신러닝 모델 - 랜덤포레스트

# In[39]:


product_feature


# In[50]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
print(data_cut)


fig2, ax2 = plt.subplots(2,1, figsize = (18, 14))

# machine-learning model용 데이터
X_train_rf, X_test_rf = np.array([list(xt[-1]) for xt in X_train_ts]),np.array([list(xt2[-1]) for xt2 in  X_test_ts])
y_train_rf, y_test_rf = y_train_ts.reshape(-1), y_test_ts.reshape(-1)

print(X_train_rf.shape)
print(X_test_rf.shape)

rf = RandomForestRegressor(n_estimators = 500, min_samples_leaf = 4, max_depth = 19, min_samples_split = 4)
rf.fit(X_train_rf, y_train_rf)

machine_model = type(rf).__name__

train_result_rf = pd.DataFrame({'actual' : y_train_rf, 'prediction' : rf.predict(X_train_rf)}, index = train_index[look_back+predict_term-1:look_back+predict_term-1+X_train_ts.shape[0]])
test_result_rf = pd.DataFrame({'actual' : y_test_rf, 'prediction' : rf.predict(X_test_rf)}, index = test_index)

train_result_rf = pd.DataFrame(y_scaler.inverse_transform(train_result_rf), index = train_result_rf.index, columns = train_result_rf.columns)
test_result_rf = pd.DataFrame(y_scaler.inverse_transform(test_result_rf), index = test_result_rf.index, columns = test_result_rf.columns)

train_perform_rf = []
test_perform_rf = []
for func in [RMSE, ME, MAE, MPE, MAPE]:
    train_perform_rf.append(func(train_result_rf['actual'], train_result_rf['prediction']))
    test_perform_rf.append(func(test_result_rf['actual'], test_result_rf['prediction']))




train_result_rf.plot( color = ['C0','C1'], ax = ax2.ravel()[0])
ax2.ravel()[0].set_title('''==== Train data :  RMSE | ME | MAE | MPE | MAPE === 
      prediction : {:.3f} | {:.3f} | {:.3f} | {:.3f} % | {:.3f} %'''.format(train_perform_rf[0], train_perform_rf[1], train_perform_rf[2], train_perform_rf[3], train_perform_rf[4]), fontsize = 18)
# plt.show()      

test_result_rf.plot(color = ['C0','C1'], style = ['-o','--o'], ax = ax2.ravel()[1] )

ax2.ravel()[1].set_title('''==== Test data :  RMSE | ME | MAE | MPE | MAPE === 
      prediction : {:.3f} | {:.3f} | {:.3f} | {:.3f} % | {:.3f} %'''.format(test_perform_rf[0], test_perform_rf[1], test_perform_rf[2], test_perform_rf[3], test_perform_rf[4]), fontsize = 18)
# plt.show()                

fig2.suptitle('Performance Plot Of {} Forecasting Model_{}'.format(product, machine_model), fontsize = 23)


# In[56]:


# 사용한 모델 및 예측값 저장
import os
import pickle
import json
# 사용한 데이터
path = os.getcwd().replace('\\','/') + '/'.format(product)

## 2차원 데이터프레임
with open(path + 'data_wk_shift_{}.pkl'.format(naming), 'wb') as f:
    pickle.dump( data_wk_shift, f)
    
data_dump = [X_train_ts, X_val_ts, X_test_ts, y_train_ts, y_val_ts, y_test_ts]
with open(path + 'train_test_{}.pkl'.format(naming), 'wb') as f:
    pickle.dump( data_dump, f)
# plot 저장
fig.savefig(path + 'performance plot_{}.png'.format(naming), dpi = 130)

fig2.savefig(path + 'performance plot_{}.png'.format(machine_model), dpi = 130)

fig3.savefig(path + 'performance plot_{}.png'.format(str(naming) + '+' + machine_model), dpi = 130)

# 예측 데이터 프레임 저장

test_result_total = pd.merge(test_result.rename(columns = {'actual' : '{}_actual'.format(naming), 'prediction' : '{}_prediction'.format(naming)}), 
                    test_result_rf.rename(columns = {'actual' : '{}_actual'.format(machine_model), 'prediction' : '{}_prediction'.format(machine_model)}),
                             left_index = True, right_index = True, how = 'outer')

test_result_total.to_csv(path + 'Hojae_test_result_total_{}.csv'.format(product), encoding = 'utf-8')


# ## 4. 앙상블 방법으로 수요 예측 보완
# - 수요 예측 시 시즌 시작 지점을 더 잘 예측하기 위함
#     - 단순히 각 모델 예측값의 평균을 사용하지 않음
#     - 시즌 구간의 수요를 잘 예측하는 모델에 더 큰 가중치를 주기 위해 다음과 같이 식을 고안
# - 사용 모델
#     - seasonal ARIMA
#     - RNN(LSTM)
#     - RandomForestRegressor
# 
#     

# In[8]:


Image('image_file/ensemble_weight.PNG', width = 500, height = 100)


# In[2]:


# 각 예측 테이블의 경로
df_path = glob.glob('*.csv')
df_path


# In[3]:


pro_list = ['맥주','생수','스타킹','아이스크림','탄산음료']

result_dict = {}

for pro in pro_list:
    df_list = [pd.read_csv(pp) for pp in [p for p in df_path if pro in p]]
    for i in range(len(df_list)):
        if 'Unnamed: 0' in df_list[i].columns:
            df_list[i] = df_list[i].set_index('Unnamed: 0')
            df_list[i] = df_list[i].drop([dc for dc in df_list[i].columns if 'actual' in dc], axis = 1)
        else:
            df_list[i] = df_list[i].set_index('sale_dt')
    
    result_dict[pro] = pd.concat(df_list, axis = 1).dropna(how = 'any').rename(columns = {pro : 'ARIMA'})
    result_dict[pro].index = pd.to_datetime(result_dict[pro].index, format = '%Y-%m-%d')
    # result_dict[품목명] : 각 품목에 대한 test 구간의 actual, seasonal ARIMA, RNN, RandomForestRegressor 의 값 


# In[4]:


# 확인
result_dict['아이스크림'].head()


# In[72]:


# 위 식의 penalty를 함수로 정의
def penalty( y_pred, y_true):
    return np.mean((np.abs(y_true - y_pred) *  y_true ))


# In[75]:


# 각 품목에서 모델의 penalty 값을 계산한 딕셔너리
result_perform = {}

for pro in result_dict.keys():
    perform_df = result_dict[pro]
    perform = perform_df.apply(penalty, y_true = perform_df['actual']) 
    result_perform[pro] = perform


# In[76]:


# 확인 : 해당 품목에서 actual, prediction을 비교하였을 때 각 모델의 penalty
result_perform['아이스크림']


# In[77]:


result_ensemble = {}

for pro in result_dict.keys():
    pred_df = result_dict[pro].drop('actual', axis = 1)
    actual = result_dict[pro]['actual']
    
    # 위 식의 penalty 역수를 통한 각 모델의 weight
    pred_weight = (1/ result_perform[pro].drop('actual') ) / (1/ result_perform[pro].drop('actual')).sum() 
    
    # 구한 weight를 기반으로 모델 예측값의 weighted sum을 수행
    ensemble_df = pred_df.multiply(pred_weight).sum(axis = 1)
    # 해당 품목의 최종 ensemble 에측값
    result_ensemble[pro] =  pd.concat([ensemble_df, actual], axis = 1).rename(columns = {0 : 'ensemble_pred'})


# In[78]:


# 각 모델의 weight 예시
pred_weight


# ### 성능측정

# In[79]:


# plot 에서 한글 사용시 깨지지 않게 하기
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/나눔고딕코딩.ttf").get_name()
rc('font', family=font_name)


# In[80]:



# 모델 성능 평가지표 함수 구축
def RMSE(y_true, y_pred):
    return np.mean( (y_true - y_pred) ** 2 ) ** (1/2)
def ME(y_true, y_pred):
    return np.mean(y_true - y_pred)
def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def MPE(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[81]:


result_ensemble.keys()


# In[7]:


result_dict['아이스크림'].columns


# In[15]:


# 이름 다시 설정하여 plotting : (8,1,3) -> LSTM
# 각 모델의 prediction과 actual값의 비교
result_dict['아이스크림'].rename(columns = dict(zip(result_dict['아이스크림'].columns, ['sesonal ARIMA','actual','LSTM','RandomForestRegressor']))).plot(figsize = (16,8), color= ['C0','black','C1','C2'], style = ['-o','--o','-o','-o'])
plt.legend(fontsize = 20)


# ### 품목별 actual, ensemble prediction 성능 평가

# In[83]:


import warnings
warnings.filterwarnings('ignore')

pro_list = list(result_ensemble.keys())

fig, ax = plt.subplots(len(pro_list),1, figsize = (18, 10 * len(pro_list)))

# test_result = pd.DataFrame({'actual' : y_scaler.inverse_transform(y_test_ts).reshape(-1), 'prediction' : y_scaler.inverse_transform(model.predict(X_test_ts)).reshape(-1)})
# test_result = pd.DataFrame(test_result)


for pro, axes in zip(pro_list, ax.ravel()):
    test_result = result_ensemble[pro]
    
    test_perform = []
    for func in [RMSE, ME, MAE, MPE, MAPE]:
        test_perform.append(func(test_result['actual'], test_result['ensemble_pred']))


    test_result.plot(color = ['red','black'], style = ['-o','--o'], ax = axes )
    axes.legend(fontsize = 25)
    axes.set_title(''' {}
                 ==== Test data :  RMSE | ME | MAE | MPE | MAPE === 
          prediction : {:.3f} | {:.3f} | {:.3f} | {:.3f} % | {:.3f} %'''.format(pro, test_perform[0], test_perform[1], test_perform[2], test_perform[3], test_perform[4]), fontsize = 18)
    # plt.show()                

    fig.suptitle('Performance Plot Of Ensemble Model', fontsize = 23)


# In[165]:


# ensemble 예측 plot 저장
fig.savefig('Ensemble test result by ARIMA, RNN, ML model.png', dpi = 130 )

# ensemble 예측 데이터 프레임 저장
import pickle
with open('result_ensemble.pkl','wb') as f:
    pickle.dump(result_ensemble, f)

