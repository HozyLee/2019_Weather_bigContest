#!/usr/bin/env python
# coding: utf-8

# ### 1. 날씨 데이터에 존재하지 않는 지역
# - 여러 지역이 통합된 경우
#     - 각 지역의 평균 값 사용
# - 기타지역
#     - 가까운 관측치 값 참고

# In[591]:


import pandas as pd
import numpy as np
import matplotlib as plt
from datetime import datetime


# ### 데이터 열기

# In[592]:


path = 'raw_data/'


# In[593]:


cvs = pd.read_csv(path + 'korea_cvs.csv', engine = 'python', encoding = 'utf-8') # GS25


# In[594]:


weather = pd.read_csv(path + 'bigcon_weather2.csv', engine = 'python', encoding = 'utf-8') # 날씨
social_dust = pd.read_csv(path + 'social_pm.csv', engine = 'python', encoding = 'utf-8') # 미세먼지 SNS


# In[595]:


# 열 이름 바꾸기
cvs.columns = [c.split('.')[-1] for c in cvs.columns]


# In[596]:


cvs.columns


# In[597]:


cvs.head()


# In[598]:


# 열 이름 바꾸기
weather.columns = [c.split('.')[-1] for c in weather.columns]


# In[599]:


weather.rename(columns = {'tm' : 'sale_dt', 'avg_wa' : 'avg_ws'}, inplace = True)


# In[600]:


weather.columns


# In[601]:


weather.head()


# In[602]:


# 열 이름 바꾸기
social_dust.columns = [c.split('.')[-1] for c in social_dust.columns]


# In[603]:


social_dust.head()


# In[604]:


# 처리를 쉽게 하기 위해 각 데이터들의 pvn_nm과 bor_nm을 통합한다.
for data in [cvs, weather]:
    #data['pvn_nm'] = data['pvn_nm'].apply(lambda x : x[ :2] + '_')
    data['area_nm'] = data['pvn_nm'] +'+'+ data['bor_nm']
    data.drop(['pvn_nm', 'bor_nm'], axis = 1, inplace = True)


# In[490]:


weather.isnull().sum()


# ### 결측치 채우기

# In[491]:


print('날씨 지역 :', weather['area_nm'].unique())
print()
print('편의점 지역 : ', cvs['area_nm'].unique())


# In[492]:


set(cvs['area_nm'].unique()) - set(weather['area_nm'].unique())


# In[493]:


# 판매 데이터와 날씨 데이터 모두에 포함되어 있는 지역들
notnull_area = list(set(weather['area_nm'].unique()) - (set(cvs['area_nm'].unique()) - set(weather['area_nm'].unique())))


# In[494]:


weather_notnull = weather[weather['area_nm'].isin(notnull_area)]


# In[495]:


##합쳐진 지역 처리
#temp_weather = weather
#temp_weather['sale_dt'] = pd.to_datetime(temp_weather['sale_dt'],format = '%Y%m%d')

empty_df = pd.DataFrame()
for merge_area in [['경기도+여주시','경기도+이천시'],['경기도+연천군','경기도+포천군'],['경기도+오산시','경기도+안성시','경기도+평택시'],['경기도+가평군','경기도+양평군']]:
    # 1. 여주시와 이천시의 날씨 지표값을 평균으로 계산하여 '경기_여주시이천시'의 결측치로 채워넣자
    if len(merge_area)==2:
        
        area_1,area_2 = merge_area[0], merge_area[1]
        temp = pd.concat([weather[weather['area_nm'] == area_1], weather[weather['area_nm'] == area_2]], axis = 0)
        temp = temp.groupby('sale_dt')

        temp_mean = temp.mean() 
        temp_mean['area_nm'] = '경기도'+'+'+area_1.split('+')[1]+area_2.split('+')[1]
        # 시간에 따라 보간하는 방식으로 결측치를 채워넣는다.
        temp_mean = temp_mean.reset_index()

        temp_mean['sale_dt'] = pd.to_datetime(temp_mean['sale_dt'],format='%Y%m%d')
        temp_mean.set_index('sale_dt',inplace=True)

        for col in temp_mean.columns : 
            temp_mean[col].interpolate(method = 'time', inplace = True)

        temp_mean.reset_index(drop = False, inplace = True)
        temp_mean['sale_dt'] = pd.to_datetime(temp_mean['sale_dt'],format='%Y%m%d')
        print(temp_mean.isnull().sum())
        #temp_weather = temp_weather[(temp_weather['area_nm']!=area_1)&(temp_weather['area_nm']!=area_2)]
        empty_df = pd.concat([empty_df,temp_mean],axis=0)
        
    else:
        area_1,area_2,area_3 = merge_area[0], merge_area[1],merge_area[2]
        temp = pd.concat([weather[weather['area_nm'] == area_1], weather[weather['area_nm'] == area_2]])
        temp = pd.concat([temp,weather[weather['area_nm']==area_3]], axis = 0)
        temp = temp.groupby('sale_dt')

        temp_mean = temp.mean() 
        temp_mean['area_nm'] = '경기도'+'+'+area_1.split('+')[1]+area_2.split('+')[1]+area_3.split('+')[1]

         # 시간에 따라 보간하는 방식으로 결측치를 채워넣는다.
        temp_mean = temp_mean.reset_index()

        temp_mean['sale_dt'] = pd.to_datetime(temp_mean['sale_dt'],format='%Y%m%d')
        temp_mean.set_index('sale_dt',inplace=True)
        
        for col in temp_mean.columns : 
            temp_mean[col].interpolate(method = 'time', inplace = True)
            
        temp_mean.reset_index(drop = False, inplace = True)
        temp_mean['sale_dt'] = pd.to_datetime(temp_mean['sale_dt'],format='%Y%m%d')
        print(temp_mean.isnull().sum())

        #temp_weather = temp_weather[(temp_weather['area_nm']!=area_1)&
         #                           (temp_weather['area_nm']!=area_2)&
          #                         (temp_weather['area_nm']!=area_3)]
        empty_df = pd.concat([empty_df,temp_mean],axis=0)


# In[496]:


# 5. 다른 지역의 결측치도 채워보자.
# 지도상에서 가장 가까운 관측소에서의 관측치로 채워넣는다.
# 단, 최대한 주어진 기상 데이터에 있는 관측소로 쓰되, 자료가 없는 경우에만 추가 데이터를 활용한다.
    # 고양시 - 금곡(540), 광명시 - 금천(417), 군포시 - 수원(119), 김포시 - 김포(570), 부천시 - 부평(649), 성남시 - 성남(572)
    # 수원시 - 수원(119), 안산시 - 안산(545), 안양시 - 과천(590), 의왕시 - 수원(119), 하남시 - 강동(402)
    # 계양구, 남동구, 동구, 미추홀구, 중구 - 인천(112) : 추가 데이터
ic_weather = pd.read_csv("인천_날씨.csv", engine = 'python')
ic_weather.columns = ['stn_id', 'sale_dt', 'avg_ta', 'min_ta', 'max_ta', 'sum_rn', 'max_ws', 'avg_ws', 'avg_rhm']

ic_rain_1 = pd.read_csv("ic_rain_1.csv", engine = 'python')
ic_rain_2 = pd.read_csv("ic_rain_2.csv", engine = 'python')
ic_rain_3 = pd.read_csv("ic_rain_3.csv", engine = 'python')

ic_rain = pd.concat([ic_rain_1, ic_rain_2, ic_rain_3], axis = 0, ignore_index = True)
ic_rain.columns = ['stn_id', 'sale_dt', 'sum_rn', 'rn_qc']
ic_rain.drop('rn_qc', axis = 1, inplace = True)
ic_rain['sale_dt'] = ic_rain['sale_dt'].apply(lambda x: str(x).split(' ')[0])

weather_fornull = weather[weather['stn_id'].isin([402, 417, 540, 545, 570, 572, 590, 649])]


# In[497]:


# 시간에 따라 보간하는 방식으로 결측치를 채워넣는다.
ic_weather['sale_dt'] = pd.to_datetime(ic_weather['sale_dt'],format="%Y-%m-%d")
ic_weather.set_index('sale_dt', inplace = True)

for i in ['max_ws', 'avg_ws'] :
    ic_weather[i].interpolate(method = 'time', inplace = True)
    
ic_weather.reset_index(drop = False, inplace = True)
ic_weather['sale_dt'] = pd.to_datetime(ic_weather['sale_dt'],format="%Y-%m-%d")


# In[498]:


# 인천(112) 관측소
# 강수량 시간 자료 중 결측치는 제거하고 일자별 평균을 낸 후, 다른 인천 기상자료와 결합한다.
# 강수량 결측치는 0으로 처리한다.
ic_rain.dropna(axis = 0, inplace = True)
ic_rain['sale_dt'] = pd.to_datetime(ic_rain['sale_dt'],format='%Y-%m-%d')
ic_rain = ic_rain.groupby('sale_dt').mean()
ic_weather = pd.merge(ic_weather, ic_rain, on = ['sale_dt', 'stn_id', 'sum_rn'], how = 'left')
ic_weather.fillna(0, inplace = True)
ic_weather.head()              


# In[499]:


# 수원(119) 관측소
weather['sale_dt'] = pd.to_datetime(weather['sale_dt'],format='%Y%m%d')
weather_fornull_sw = weather[weather['stn_id'] == 119]

# 시간에 따라 보간하는 방식으로 결측치를 채워넣는다.
weather_fornull_sw.set_index('sale_dt', inplace = True)
for i in weather_fornull_sw.columns : 
    weather_fornull_sw[i].interpolate(method = 'time', inplace = True)
    
weather_fornull_sw.reset_index(drop = False, inplace = True)
weather_fornull_sw['sale_dt'] = pd.to_datetime(weather_fornull_sw['sale_dt'],format='%Y%m%d')

weather_fornull_sw = pd.concat([weather_fornull_sw] * 3, axis = 0)
weather_fornull_sw.sort_values(by = 'sale_dt', axis = 0, inplace = True)
weather_fornull_sw.reset_index(drop = True, inplace = True)
weather_fornull_sw['sale_dt'] = pd.to_datetime(weather_fornull_sw['sale_dt'],format='%Y%m%d')

weather_fornull_sw['area_nm'] = ['경기도+군포시', '경기도+수원시', '경기도+의왕시'] * 1096
print(weather_fornull_sw.isnull().sum())
weather_fornull_sw.head(10)


# In[500]:


weather_fornull_ic = pd.concat([ic_weather] * 5, axis = 0)
weather_fornull_ic.sort_values(by = 'sale_dt', axis = 0, inplace = True)
weather_fornull_ic.reset_index(drop = True, inplace = True)
weather_fornull_ic['area_nm'] = ['인천광역시+계양구', '인천광역시+남동구', '인천광역시+동구', '인천광역시+미추홀구', '인천광역시+중구'] * 1096
print(weather_fornull_ic.isnull().sum())
weather_fornull_ic.head()


# In[501]:


# 그 외 여러 관측소 한꺼번에 보간하기
weather_fornull['area_nm'] = weather_fornull['stn_id'].apply(lambda x : '경기도+하남시' if x == 402 else ('경기도+광명시' if x == 417 else ('경기도+고양시' if x == 540 else ('경기도+안산시' if x == 545 else ('경기도+김포시' if x == 570 else ('경기도+성남시' if x == 572 else ('경기도+부천시' if x == 649 else '경기도+안양시')))))))
print(weather_fornull['stn_id'].unique())
print(weather_fornull['area_nm'].unique())
print(weather_fornull.isnull().sum())

# 시간에 따라 보간하는 방식으로 결측치를 채워넣는다.
weather_fornull['sale_dt'] = pd.to_datetime(weather_fornull['sale_dt'],format='%Y%m%d')
weather_fornull.set_index('sale_dt', inplace = True)

for i in weather_fornull.columns[ : -2] :
    weather_fornull[i].interpolate(method = 'time', inplace = True)
    
weather_fornull.reset_index(drop = False, inplace = True)
weather_fornull['sale_dt'] = pd.to_datetime(weather_fornull['sale_dt'],format='%Y%m%d')
# 강수량 결측치는 0으로 처리한다.
weather_fornull.fillna(0, inplace = True)
print(weather_fornull.isnull().sum())


# In[502]:


weather_fornull.head()


# In[503]:


weather_temp= pd.concat([empty_df,weather_notnull,weather_fornull_ic, weather_fornull_sw, weather_fornull],axis=0)
weather_temp.head()
print(weather_temp.isnull().sum())


# In[504]:


print(len(empty_df))
print(len(weather_fornull_ic))
print(len(weather_fornull_sw))
print(len(weather_fornull))
print(len(weather_notnull))
print(len(weather_temp))


# In[505]:


# 나머지 결측치 처리
# 강수량을 제외한 나머지 날씨 변수들은 시간에 따라 보간
weather_temp['sum_rn'].fillna(0, inplace = True)
weather_rev = weather_temp
weather_rev.head()


# In[506]:


weather_rev['sale_dt'] = weather_rev['sale_dt'].apply(lambda x: str(x).split(' ')[0])
weather_rev['sale_dt'] = pd.to_datetime(weather_rev['sale_dt'], format= "%Y-%m-%d")
weather_rev.set_index('sale_dt', inplace = True)

for i in weather_rev[ : -2] :
    weather_rev[i].interpolate(method = 'time', inplace = True)
    
weather_rev.reset_index(drop = False, inplace = True)
weather_rev['sale_dt'] = pd.to_datetime(weather_rev['sale_dt'], format= "%Y-%m-%d")


# In[507]:


weather_rev.isnull().sum()


# ### 빈 날짜 채워넣기

# In[508]:


# 중간에 데이터가 존재하지 않는 날짜가 존재하므로 이를 채워주자.
dt_index = pd.date_range(start = '20160101', end = '20181231')  # start부터 end까지의 날짜를 생성하는 코드

dt_list = list(dt_index.strftime('%Y%m%d'))

area_nm_list = list(set(cvs['area_nm']))
area_nm_list = sorted(area_nm_list * len(dt_list))


# In[509]:


dt_area_list = pd.DataFrame(zip(dt_list * len(area_nm_list), area_nm_list))
dt_area_list.columns = ['sale_dt', 'area_nm']
dt_area_list['sale_dt'] = pd.to_datetime(dt_area_list['sale_dt'],format = '%Y-%m-%d')


# In[510]:


weather_rev = pd.merge(weather_rev, dt_area_list, on = ['sale_dt', 'area_nm'], how = 'outer')


# In[512]:


weather_rev.isnull().sum()


# In[513]:


# 빈 날짜들의 결측치는 강수량을 제외하고 모두 시간에 따라 보간
weather_rev['sum_rn'].fillna(0, inplace = True)


# In[517]:


weather_rev['sale_dt'] = weather_rev['sale_dt'].apply(lambda x : str(x), "%Y%m%d")
weather_rev['sale_dt'] = pd.to_datetime(weather_rev['sale_dt'],format = '%Y-%m-%d')
weather_rev.set_index('sale_dt', inplace = True)

for i in weather_rev.columns :
    weather_rev[i].interpolate(method = 'time', inplace = True)
    
weather_rev.reset_index(drop = False, inplace = True)
weather_rev['sale_dt'] = pd.to_datetime(weather_rev['sale_dt'],format = '%Y-%m-%d')


# In[518]:


weather_rev.isnull().sum()


# ## 2. 이상치 처리

# In[541]:


# 중간 과정의 날씨 데이터 저장
weather_rev.to_csv(path + "weather_before.csv", index = False, encoding = 'utf-8')


# ### 미세먼지 데이터와 결합하기

# In[542]:


weather_rev = pd.read_csv(path + "weather_before.csv", engine = 'python', encoding = 'utf-8')
weather_rev['sale_dt'] = pd.to_datetime(weather_rev['sale_dt'],format='%Y-%m-%d')


# In[543]:


# 날씨 & 미세먼지 데이터
weather_dust = pd.read_csv(path + 'dust_raw.csv', engine = 'python')
# 미세먼지 관측소는 총 세 곳이다. (서울 - 서울, 인천 - 강화, 경기 - 수원)
weather_dust.columns = ['stn_id', 'pvn_nm', 'sale_dt', 'dust']
weather_dust.drop('stn_id', axis = 1, inplace = True)
weather_dust['sale_dt'] = pd.to_datetime(weather_dust['sale_dt'],format = '%Y-%m-%d')
weather_dust['pvn_nm'] = weather_dust['pvn_nm'].apply(lambda x : '서울' if x == '서울' else ('경기' if x == '수원' else '인천'))


# In[544]:


weather_dust.head()


# In[545]:


weather_rev['pvn_nm'] = weather_rev['area_nm']
weather_rev['pvn_nm'] = weather_rev['pvn_nm'].apply(lambda x : x[ : 2])


# In[546]:


weather_rev = pd.merge(weather_rev, weather_dust, on = ['sale_dt', 'pvn_nm'], how = 'left')


# In[547]:


weather_rev.head()


# In[548]:


# 미세먼지 결측치를 지역 대분류(서울, 인천, 경기) 값에 따라 처리한다.
dust_seoul = weather_rev[weather_rev['pvn_nm'] == '서울']
dust_ic = weather_rev[weather_rev['pvn_nm'] == '인천']
dust_gg = weather_rev[weather_rev['pvn_nm'] == '경기']

dust_seoul['dust'].fillna(method = 'pad', inplace = True)
dust_ic['dust'].fillna(method = 'pad', inplace = True)
dust_gg['dust'].fillna(method = 'pad', inplace = True)


# In[549]:


weather_rev = pd.concat([dust_seoul, dust_ic, dust_gg], axis = 0, ignore_index = True)


# ### 미세먼지 SNS 데이터와 결합하기

# In[555]:


social_dust.head()


# In[556]:


social_dust.columns = ['sale_dt', 'pm_blog', 'pm_twitter', 'pm_news', 'pm_total']
social_dust['sale_dt'] = pd.to_datetime(social_dust['sale_dt'],format='%Y-%m-%d')


# In[557]:


weather_rev = pd.merge(weather_rev, social_dust, on = 'sale_dt', how = 'left')
weather_rev.head()


# In[558]:


weather_rev.isnull().sum()


# ### 주말 변수

# In[562]:


# 주말 여부를 Dummy Variable로 만든다.
weather_rev['weekend'] = weather_rev['sale_dt'].apply(lambda x : datetime.weekday(x))
weather_rev['weekend'] = weather_rev['weekend'].apply(lambda x : 1 if (x == 5) or (x == 6) else 0)


# ### 공휴일 변수
# 증권시장 휴장일 데이터를 기준으로 공휴일을 잡는다.

# In[564]:


rest_2016 = pd.read_csv(path+"2016_공휴일.csv", engine = 'python', encoding = 'cp949')
rest_2017 = pd.read_csv(path+"2017_공휴일.csv", engine = 'python', encoding = 'cp949')
rest_2018 = pd.read_csv(path+"2018_공휴일.csv", engine = 'python', encoding = 'cp949')


# In[572]:


restday = pd.concat([rest_2016, rest_2017, rest_2018], axis = 0, ignore_index = True, sort = False)
restday.rename({'일자 및 요일' : 'sale_dt', '요일구분' : 'holiday'}, axis = 1, inplace = True)
restday['sale_dt'] = pd.to_datetime(restday['sale_dt'],format='%Y-%m-%d')
restday['holiday'] = str(1)


# In[573]:


restday.head()


# In[574]:


weather_rev = pd.merge(weather_rev, restday, on = 'sale_dt', how = 'left')
weather_rev.fillna(str(0), inplace = True)


# In[ ]:


# API를 사용하긴 했지만 실제로 데이터 전처리 시 반영하지는 않음
# 코드만 참고할 것
'''
# 공공데이터 API 사용 시 코드

import requests
import xml.etree.ElementTree as elemTree

api_key = 
url_format = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo?solYear={year}&solMonth={month}&ServiceKey={api_key}'

year_list = ['2016', '2017', '2018']
month_list = [str(i) for i in range(1, 13)]

headers = {'content-type' : 'application/json;charset=utf-8'}

result = {'locdate' : [], 'isHoliday' : [], 'dateName' : []}

for year in year_list :
    for month in month_list :
        url = url_format.format(year = year, month = month, api_key = api_key)
        response = requests.get(url, headers = headers, verify = False)
        response_xml = elemTree.fromstring(response.text)
        
        
        for each_dt_ele in response_xml.findall('.//item'):
            for k in result.keys():
                result[k].append(each_dt_ele.find(k).text)
            
result_df = pd.DataFrame(result)
'''


# In[575]:


# 판매 데이터와의 결합 전의 날씨 데이터 저장
weather_rev.to_csv('weather_before_merge.csv', index = False, encoding = 'utf-8')


# ### 판매 데이터들과 결합

# In[581]:


cvs.head()


# In[579]:


weather_rev.info()


# In[589]:


cvs


# In[605]:


# 이제 cvs, hnb 데이터와 merge
cvs['sale_dt'] = pd.to_datetime(cvs['sale_dt'],format='%Y%m%d')
cvs_total = pd.merge(cvs, weather_rev, on = ['sale_dt', 'area_nm'], how = 'left')
cvs_total.drop(['stn_id', 'avg_rhm'], axis = 1, inplace = True)
print(cvs_total.isnull().sum())


# In[606]:


cvs_total.to_csv(path + "cvs_weather.csv", index = False, encoding = 'utf-8')


# In[ ]:




