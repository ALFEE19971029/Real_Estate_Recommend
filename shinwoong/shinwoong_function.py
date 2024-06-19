import pandas as pd
import numpy as np
import sklearn
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def shinwoong_func(lower_limit, upper_limit):
    underrated_df = pd.read_csv(r"C:\Users\kshw1\Desktop\Study\중급프로그래밍\final_project\shinwoong\underrated_data.csv")
    data_df = pd.read_csv(r"C:\Users\kshw1\Desktop\Study\중급프로그래밍\project_functions\shinwoong_data\data_강남2.csv", encoding='utf-8-sig')
    filtered_underrated_df = underrated_df[(underrated_df['최근거래가격'] >= lower_limit) & (underrated_df['최근거래가격'] <= upper_limit)]
    filtered_underrated_df = filtered_underrated_df.sort_values(by='저평가_비율', ascending=False)
    top5_apt = filtered_underrated_df.head(5).reset_index(drop=True)

    result_hat =[]
    result_gubun = []
    result_name = []
    result_beforeprice = []
    result_rate = []
    result_beforeprice_per_area = []
    result_real_beforeprice_per_area = []
    
    for i in range(len(top5_apt)):
        ix = top5_apt.loc[i, 'Gubun']
        #print(i, ix)

        data_df['Price_per_area'] = data_df['GPrice']/data_df['EArea']
        df1 = data_df[data_df['Gubun'] == ix]
        df1['GMonth1'] = pd.to_datetime(df1['GMonth'], format='%Y%m')

        df1 = df1.sort_values(by='GMonth1')

        if df1.empty:
            print(f"No data found for apartment: {ix}")
            continue
        
        # 필요한 데이터만 추출
        df_prophet = pd.DataFrame({
        'ds': df1['GMonth1'],
        'y': df1['Price_per_area']
        })
        
        train = df_prophet[df_prophet['ds'] < pd.Timestamp('2023-01-01')]  # 예시: 'cutoff_date'는 데이터 분할 기준 날짜
        # df_prophet
        
        model = Prophet()
        model.fit(train)
        
        # 미래 데이터 프레임 생성 (2023년 12월까지)
        last_date = pd.Timestamp('2023-12-31')
        months_to_forecast = (last_date.year - train['ds'].max().year) * 12 + (last_date.month - train['ds'].max().month) +1
        future = model.make_future_dataframe(periods=months_to_forecast, freq='M')
        
        forecast = model.predict(future)

        ###############시각화######################(평당가격도 충분한지 확인해보기 위해)
        # 실제 데이터와 예측 데이터 병합
        merged = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds', how='left')
        
        # 시계열 데이터 시각화
        #plt.figure(figsize=(12, 6))
        #plt.plot(merged['ds'], merged['y'], label='Actual Price per Area')
        #plt.plot(merged['ds'], merged['yhat'], label='Predicted Price per Area', linestyle='--')
        #plt.xlabel('Date')
        #plt.ylabel('Price per Area')
        #plt.title('Actual vs Predicted Price per Area')
        #plt.legend()
        #plt.show()
        ###################################################
        
        result_hat.append(float(forecast[forecast['ds'] == '2023-12-31']['yhat']))
        result_gubun.append(ix)
        temp = data_df.copy()
        danji_idx = temp[temp['Gubun'] == ix].index
        result_name.append(temp.loc[danji_idx[0]]['AptName'])
        
        latest_gmonth_before_202401 = data_df[(data_df['Gubun'] == ix) & (data_df['GMonth'] < 202401)]['GMonth'].max()
        latest_gmonth_before_202301 = data_df[(data_df['Gubun'] == ix) & (data_df['GMonth'] < 202301)]['GMonth'].max()

        # realprice_after = df0[(df0['GMonth'] == latest_gmonth_before_202401) & (df0['Gubun'] == ix)]['GPrice'].mean()
        #realprice_before = df0[(df0['GMonth'] == latest_gmonth_before_202301) & (df0['Gubun'] == ix)]['GPrice'].mean()
        realprice_before = data_df[(data_df['GMonth'] == latest_gmonth_before_202301) & (data_df['Gubun'] == ix)].sort_values(by='GDay').iloc[-1]['GPrice']
        price_per_area_before = data_df[(data_df['GMonth'] == latest_gmonth_before_202301) & (data_df['Gubun'] == ix)]['Price_per_area'].max()
        real_price_per_area_before_2024 = data_df[(data_df['GMonth'] == latest_gmonth_before_202401) & (data_df['Gubun'] == ix)]['Price_per_area'].max()

        # result_real_after.append(realprice_after)
        result_beforeprice.append(realprice_before)
        result_beforeprice_per_area.append(price_per_area_before)
        result_real_beforeprice_per_area.append(real_price_per_area_before_2024)
        result_rate.append(float(forecast[forecast['ds'] == '2023-12-31']['yhat'])/price_per_area_before)
        
    result_df = pd.DataFrame({"gubun":result_gubun, 'AptName':result_name, "hat":result_hat, "beforeprice":result_beforeprice, "rate":result_rate, "beforeprice_per_area":result_beforeprice_per_area,"real_2023_price":result_real_beforeprice_per_area})
    result_df = result_df.sort_values(by='rate',ascending=False)
    final_df = pd.DataFrame()
    if len(result_df) == 0:
        final_df = []
    else:
        final_df['Gubun'] = result_df['gubun']
        final_df['AptName_EArea2'] = final_df['AptName_EArea2'] = result_df['AptName'] + "_" + result_df['gubun'].apply(lambda x: x.split('_')[1])
        final_df['202212_real'] = result_df['beforeprice_per_area']
        final_df['202312_predict'] = result_df['hat']
        final_df['202312_real'] = result_df['real_2023_price']
    return final_df