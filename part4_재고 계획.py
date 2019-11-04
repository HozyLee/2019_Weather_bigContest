#!/usr/bin/env python
# coding: utf-8

# # part 4.창고 재고 관리 계획 모델
# ### part3 의 test set의 수요 예측 값 기반

# ### 지역의 수요를 창고에 배정
# - 거리별 가장 가까운 warehouse로 구 배정
# - 용량 제한 없음

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys


# ### 지역 위치 정리(위도, 경도)

# ### crawling 시작
# - vworld API를 사용하여 각 지역의 위도, 경도 데이터 확보
#     - 지역 통합 시 : 각 지역의 위도, 경도 평균값 사용

# In[5]:


data_ori = pd.read_csv(path + 'korea.csv')
loc_unique = (data_ori['korea_cvs.pvn_nm'] + ' ' + data_ori['korea_cvs.bor_nm']).unique()
loc_unique


# In[147]:


import urllib.request
import urllib.parse
import urllib
import requests
import json

apikey = 'masking'
apiurl = 'http://api.vworld.kr/req/address?service=address&request=getCoord&key='+apikey+'&'

address_dict = {}

for address in loc_unique:
    l_cnt = [address.split(' ')[-1].count(l) for l in ['시','군','구']]
    if max(l_cnt) > 1:
        l = ['시','군','구'][l_cnt.index(max(l_cnt))]
        l_each = address.split(' ')[-1].split(l)
        l_each.remove('')
        l_each = [le + l for le in l_each]
        
        y_list, x_list = [], []
        
        for l_e in l_each:
            values = {'address' : address.split(' ')[0] + ' ' + l_e,
                  'type' : 'ROAD'}
            param = urllib.parse.urlencode(values)
            
            apiurl_param = apiurl + param

            re = requests.get(apiurl_param)
            re_json = json.loads(re.text)

            lat_long = re_json['response']['result']['point']
            y_list.append(float(lat_long['y']))
            x_list.append(float(lat_long['x']))
        address_dict[address] = [float(np.mean(y_list)), float(np.mean(x_list))]
        
    else:
        values = {'address' : address,
                  'type' : 'ROAD'}
        param = urllib.parse.urlencode(values)

        apiurl_param = apiurl + param

        re = requests.get(apiurl_param)
        re_json = json.loads(re.text)

        lat_long = re_json['response']['result']['point']

        address_dict[address] = [float(lat_long['y']), float(lat_long['x'])] # 위도(y), 경도(x) 순  # x : 경도


# In[150]:


# 전체 다 한건지 확인해주고
set(address_dict.keys()) == set(loc_unique)


# In[160]:


address_total = pd.DataFrame(address_dict).T
address_total.columns = ['위도','경도']

address_total.reset_index(inplace = True)
address_total['pvn_nm'] = address_total['index'].apply(lambda x : x.split(' ')[0])
address_total['bor_nm'] = address_total['index'].apply(lambda x : x.split(' ')[-1])
address_total.drop('index', axis = 1, inplace = True)

address_total

# 저장 : 지역의 위도, 경도 데이터
address_total.to_csv('C:/python/2019_weather_bigcontest/cvs_lat_long.csv', index = False)

# ### gs리테일 물류센터
# - 네이버 지도에서 'gs리테일 물류센터' 검색 후 나오는 장소 중 서울,인천,경기에 가까운 5개의 주소 선택
# - 마찬가지로 주소를 기준으로 vworld API를 사용하여 위도, 경도 확보

# In[116]:



logistics_loc_list = ['경기도 김포시 고촌읍 아라육로57번길 15',
                      '경기도 남양주시 마치로 306',
                      '인천광역시 미추홀구 염전로144번길 25',
                      '경기도 용인시 처인구 포곡로 100',
                      '경기도 파주시 탄현면 한록산길 7',]
#                       '강원도 원주시 원문로 2570']


# In[117]:


import urllib.request
import urllib.parse
import urllib
import requests
import json


apikey = 'masking'
apiurl = 'http://api.vworld.kr/req/address?service=address&request=getCoord&key='+apikey+'&'


logistics_loc_dict = {}

for logi in logistics_loc_list:
    values = {'address' : logi,
              'type' : 'ROAD'}
    param = urllib.parse.urlencode(values)

    apiurl_param = apiurl + param

    re = requests.get(apiurl_param)
    re_json = json.loads(re.text)

    lat_long = re_json['response']['result']['point']

    logistics_loc_dict[logi] = [float(lat_long['y']), float(lat_long['x'])] # 위도(y), 경도(x) 순  # x : 경도
    


# In[118]:


logistics_loc_dict


# In[119]:


logistics_loc = pd.DataFrame(logistics_loc_dict).T
logistics_loc = logistics_loc.reset_index()
logistics_loc.columns = ['location', '위도','경도']


# In[120]:


logistics_loc

# 저장 : 각 gs리테일 창고 위도, 경도
logistics_loc.to_csv(path + 'warehouse_loc_lat_long.csv', index = False, encoding = 'utf-8')
# ### 지역, 창고 간의 haversine 최소 거리 구하기

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_ori = pd.read_csv(path + 'korea_cvs.csv')
data_ori.columns = [c.split('.')[-1] for c in data_ori.columns]

loc_lat_long = pd.read_csv(path + 'cvs_lat_long.csv')
logistics_lat_long = pd.read_csv(path + 'warehouse_loc_lat_long.csv')


# In[9]:


loc_lat_long.head()


# In[10]:


logistics_lat_long.head()


# In[29]:


# 지역과 warehouse의 거리를 계산
from haversine import haversine
dist_dict = {}
for logi_loc in logistics_lat_long.iterrows():
    lo_nm = logi_loc[1]['location']
    dist_dict[lo_nm] = {}
    
    
    for cvs_loc in loc_lat_long.iterrows():
        
        cvs_nm_pvn, cvs_nm_bor  = cvs_loc[1]['pvn_nm'],cvs_loc[1]['bor_nm']
        cvs_nm = cvs_nm_pvn + ' ' + cvs_nm_bor
        c_ll, lo_ll = tuple([cvs_loc[1]['위도'],cvs_loc[1]['경도']]), tuple([logi_loc[1]['위도'], logi_loc[1]['경도']])
        dist = haversine(c_ll, lo_ll)
        
        dist_dict[lo_nm][cvs_nm] = dist


# In[ ]:


dist_df = pd.DataFrame(dist_dict).reset_index()
dist_df['pvn_nm'] = dist_df['index'].apply(lambda x : x.split(' ')[0])
dist_df['bor_nm'] = dist_df['index'].apply(lambda x : x.split(' ')[-1])
dist_df.drop('index' ,axis = 1, inplace= True)
dist_df.set_index(['pvn_nm','bor_nm'], inplace = True)


# In[126]:


dist_df.head()

# 저장 : 지역, 창고와의 haversine 거리 데이터프레임
dist_df.to_csv(path + 'loc_w_dist.csv', index = True, encoding = 'utf-8')
# ### 가장 가까운 창고로 지역 배정
# - 지역, 창고의 haversine 최소 거리를 기반으로

# In[2]:


dist = pd.read_csv('C:/python/2019_weather_bigcontest/loc_w_dist.csv')
dist.set_index(['pvn_nm','bor_nm'], inplace = True)

# 지역(행)에 가장 가까운 창고의 이름을 column으로 추가
warehouse_name = dist.columns
dist['shortest_warehouse'] = dist.agg(lambda x : warehouse_name[np.argmin(list(x))], axis = 1)


# In[5]:


dist.head()


# In[6]:


warehouse_assign = {}

unique_warehouse = list(dist['shortest_warehouse'].unique())

for uw in unique_warehouse:
    warehouse_assign[uw] = list(dist[dist['shortest_warehouse'] == uw].index)


# In[7]:


warehouse_assign


# ### folium에 그리기

# In[9]:


import folium

cvs_ll = pd.read_csv('C:/python/2019_weather_bigcontest/cvs_lat_long.csv')
ware_ll = pd.read_csv('C:/python/2019_weather_bigcontest/warehouse_loc_lat_long.csv')


# In[13]:


ware_ll.set_index('location', inplace = True)
cvs_ll.set_index(['pvn_nm','bor_nm'], inplace = True)


# In[15]:


warehouse_assign_xy = {}

for w in warehouse_assign.keys():
    warehouse_assign_xy[tuple(ware_ll.loc[w])] = [tuple(cvs_ll.loc[c]) for c in warehouse_assign[w]]


# In[16]:


Map = folium.Map(
    location=[37.5838699,127.0565831],
    zoom_start=9
)
# color = ['#FF4A33','#FFA133','#0CB70E','#3765F2']
color = ['blue','red','green','purple','black']

print(warehouse_assign.keys())
for color_idx, w_xy in enumerate(warehouse_assign_xy.keys()):
    l_xy_list = warehouse_assign_xy[w_xy]
    
    folium.Marker(location = w_xy,
                  icon = folium.Icon(color = color[color_idx], icon = 'star')).add_to(Map)
    
    for l_xy in l_xy_list:
        folium.CircleMarker(location = l_xy, color = color[color_idx], fill_color = color[color_idx],
                   radius = 5).add_to(Map)
     
    
path = os.getcwd().replace('\\','/') + '/'
Map.save(path + 'warehouse_assign_only_distance.html')


# In[17]:


Map

# 저장 : 최소 거리 기반으로 창고에 지역을 배정한 딕셔너리
import pickle
with open('waerhouse_assign_only_distance.pkl', 'wb') as f:
    pickle.dump(warehouse_assign, f)
# ### 창고에 배정 된 수요 데이터 프레임 만들기

# In[12]:


data = pd.read_csv(path + 'cvs_winsorized_99 + smoothing.csv')
data['sale_dt'] = pd.to_datetime(data['sale_dt'].astype(str), format = '%Y-%m-%d')
data.set_index('sale_dt', inplace = True)
data.fillna(0, inplace = True)


# In[18]:


data.head()


# In[31]:


data.index


# In[28]:


ware_data = {}

for w in warehouse_assign.keys():
    ass_plc = [p[0] + '+' + p[1] for p in warehouse_assign[w]]
    print(len(ass_plc))
    ass_df = data[data['level_1'].apply(lambda x : x in ass_plc)].drop('level_1', axis=1)
    print(ass_df.shape[0])
    ass_df = ass_df.groupby(ass_df.index).sum()
    
    ware_data[w] = ass_df
    print()


# In[30]:


with open('warehouse_data.pkl', 'wb') as f:
    pickle.dump(ware_data, f)


# # 재고 괸리 모델(발주 계획)

# # 정량/정기 발주 계획
# ## 정량 발주 : 시즌성 상품
# - 변동성이 심한 상품
# - 시즌성 상품의 경우 시즌 구간에서 매 주 ROP, Q를 업데이트 한다.
# ## 정기 발주 : 비시즌성 상품
# - 변동성이 적은 상품
# - 월별로 변동성이 작기 때문에 적정 수준의 재고 유지 전략 사용

# In[2]:


from IPython.display import Image


# In[4]:


Image('image_file/inventory_strategy.PNG', width = 500, height = 200)


# In[1]:


import pandas as pd
import numpy as np
import math
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
rc('font', family='NanumGothic')
fm._rebuild()

import warnings
warnings.filterwarnings('ignore')


# In[2]:


path = '../data_set/edit/'


# In[3]:


kimpo = pd.read_csv(path + 'kimpo.csv') #김포시 창고에 제품별 수요예측값


# In[4]:


kimpo.head()


# In[5]:


import pickle
with open(path+'result_ensemble.pkl','rb') as f:
    season = pickle.load(f) #김포시 시즌상품 제품별 수요예측값


# In[6]:


season['아이스크림']


# In[7]:


def setIndex2(df):
    df['sale_dt'] = pd.to_datetime(df['sale_dt'].astype(str), format = '%Y-%m-%d')
    df.set_index('sale_dt', inplace = True)
    return df


# In[8]:


kimpo = setIndex2(kimpo) #index변경


# In[10]:


kimpo.head()


# In[14]:


temp = kimpo['우산']
temp[0] = 6000 #우산 판매 예측값이 0이하인 것은 6000으로 대체
kimpo['우산'] = temp


# In[9]:


#파라미터 이것 사용!!
a = 400000; h = 20; l = 1.5; k = 15
#a: ordering cost(주문 운송비)
#h: holding cost(개*일 재고비용)
#l: lead time(납기일)
#k: stockout cost(개 유실비용)


# # 정량발주 (R,T)

# In[22]:


Q = (((kimpo*2*a)/h).agg(np.sqrt)).astype(int)
Z = (k*kimpo/(k*kimpo + h*Q)).agg(norm.ppf)
TB = ((((2*a*kimpo)/h).agg(np.sqrt))/kimpo) #Time betwwen reviews
TR = kimpo*(TQ+l)+ Z*((kimpo*l/30).agg(np.sqrt)) #Target Inventory level


# In[23]:


(TB*30).head()


# In[24]:


TR.head()


# In[19]:


(TB*30).resample('Q').mean() #월별이 아니라 분기별로 계획을 다르게함


# In[20]:


TR.resample('Q').mean()


# # 정기 계획 (Q, r)

# In[21]:


df = season['아이스크림']['ensemble_pred']


# In[26]:


Q = (((df*2*a)/h).agg(np.sqrt)).astype(int) #최적 주문량
Z = (k*df/(k*df + h*Q)).agg(norm.ppf)
R = (df*l/7 + Z*((df*l/7).agg(np.sqrt))).astype(int) #최적 reorder point


# In[27]:


Q[Q.index >= '2018-07']


# In[28]:


R[R.index >= '2018-07']


# In[29]:


norm.cdf(Z[Z.index >= '2018-07'])


# # 구별 판매 추이 탐색

# In[25]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import math

#plot에서 한글 사용시 깨지지 않게 하기
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc
rc('font', family='NanumGothic')
fm._rebuild()


# In[30]:


def setIndex(df):
    df['sale_dt'] = pd.to_datetime(df['sale_dt'].astype(str), format = '%Y%m%d')
    df.set_index('sale_dt', inplace = True)
    return df


# In[31]:


#l : smoothing span
#g : moving average span
#m : sig 관리도 수준

def RQ_MAG(l,g,m,cur_df,bor_set,week_set):
    #make v_array
    n = len(bor_set)
    v_list = []
    for bor in bor_set:
        temp = []
        df = cur_df[cur_df['bor_nm']==bor]['adj_qty']
        for t in range(l,len(week_set)):
            s = 0
            for a in range(l):
                s = s + df.iloc[t-a]
            temp.append(s)
        temp = np.array(pd.Series(temp).rank(ascending = False))
        temp = (n+1-temp)*100/n
        v_list.append(temp)
    v_array = np.array(v_list)
    
    #make u_array
    u_list = []
    for bor in range(len(bor_set)):
        temp = []
        for t in range(len(week_set)-l):
            u = 0
            for h in range(g):
                u = u + v_list[bor][t-h]
            u = u/g
            temp.append(u)
        u_list.append(temp)
    u_array = np.array(u_list)
    
    #make sig
    sig_list = []
    for bor in range(len(bor_set)):
        temp = []
        for t in range(len(week_set)-l):
            sig = 0
            for h in range(g):
                sig = sig + math.pow((u_list[bor][t-h]-v_list[bor][t-h]),2)
            sig = sig/g
            sig = math.sqrt(sig)
            temp.append(sig)
        sig_list.append(temp)
    sig_array = np.array(sig_list)
    
    #make UAL, LAL
    ual_array = m*sig_array
    lal_array = -m*sig_array
    
    return v_array, ual_array, lal_array, l


# In[32]:


cvs = pd.read_csv(path+'cvs_weather.csv')


# In[34]:


cvs.head() #전처리가 끝난 cvs파일


# In[35]:


cvs = setIndex(cvs) #index변경


# In[36]:


cvs_week = cvs.groupby(['bor_nm']).resample('W-MON')['adj_qty'].sum().reset_index(level=0) #주단위로 묶음


# In[38]:


cvs_qty = cvs_week[['bor_nm', 'adj_qty']] #(구, 판매량)


# In[40]:


cvs_qty.head()


# In[39]:


cvs_bor_set = np.unique(cvs_qty['bor_nm'])
cvs_week_set = np.unique(cvs_qty.index)
cvs_borToindex = {cvs_bor_set[i] : i for i in range(len(cvs_bor_set))}
cvs_indexTobor = {i : cvs_bor_set[i] for i in range(len(cvs_bor_set))}


# In[41]:


cvs_rq, cvs_ual, cvs_lal, l = RQ_MAG(12,12,3,cvs_qty,cvs_bor_set, cvs_week_set) #l,g,m,cur_df,bor_set,week_set -> RQ, ual, lal


# In[42]:


cvs_borToindex.get('강남구') #보고싶은 구를 get()에 입력


# In[47]:


i = 1 #위에서 나온 숫자를 i에 입력
plt.figure(figsize = (20,10))
plt.plot(cvs_week_set[l:], cvs_rq[i], label = cvs_bor_set[i] + '_RQ')
plt.plot(cvs_week_set[l:], cvs_ual[i],  label = cvs_bor_set[i] + '_Up' )
plt.plot(cvs_week_set[l:], cvs_lal[i],  label = cvs_bor_set[i] + '_Low')
plt.legend(fontsize = 20)
plt.show()

