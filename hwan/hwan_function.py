import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

def whan_model(lower_limit, upper_limit):
    df = pd.read_csv(r"C:\Users\kshw1\Desktop\Study\중급프로그래밍\final_project\hwan\진짜최종.csv", index_col = 0)
    df_under = pd.read_csv(r"C:\Users\kshw1\Desktop\Study\중급프로그래밍\final_project\hwan\저평가아파트.csv", index_col = 0)

    def underrated_model(lower_limit, upper_limit, df_under):
        affordable_apartments = df_under[df_under['GPrice'].between(lower_limit, upper_limit, inclusive='both')].sort_values(by='GPrice', ascending = False)
        if affordable_apartments.empty:
            return []
        else:
            return affordable_apartments[:5]['Gubun'].unique().tolist()

    def price_prediction(df):
        apt_list = underrated_model(lower_limit, upper_limit, df_under)
        if not apt_list:
            print("조건에 부합하는 아파트가 없습니다.")
            return pd.DataFrame()

        df = df[df['Gubun'].isin(apt_list)].drop(['Si','Gu','Dong','Bunji','Bonbun','Bubun','Juso','StrName','JusoCode','EArea2','BYear','Floor','maxFloor','percentFloor','Lat','Long','nearStation','distanceStation','avgSeoulPrice','avgGuPrice','avgRentPrice','RentGap'], axis=1)
        df['GDay'] = [f"{day:02d}" for day in df['GDay']]
        df['Date'] = df['GMonth'].astype(str) + df['GDay'].astype(str)
        df['Date'] = df['Date'].astype(int)
        df['202212_real'] = df['GPrice'] / df['EArea']

        def EDA(x):
          x = x.loc[x['GMonth'] <= 202212].sort_values(by="GMonth").groupby(x['GMonth'])['202212_real'].mean().reset_index()
          return x

        def make_month_merge(x):
          # 2010년 1월부터 2023년 11월까지의 월 단위 날짜 범위 생성
          start_date = pd.to_datetime('2010-01-01')
          end_date = pd.to_datetime('2022-12-31')
          monthly_range = pd.date_range(start=start_date, end=end_date, freq='MS')

          # 생성된 월 단위 날짜 범위를 데이터프레임으로 변환
          monthly_df = pd.DataFrame({'GMonth': monthly_range})
          monthly_df['GMonth'] = monthly_df['GMonth'].dt.strftime('%Y-%m').astype(str).apply(lambda x: x.replace("-",'')).astype(int)
          merged_df = pd.merge(monthly_df, x, on='GMonth', how='left')
          return merged_df

        def fill(x):
          # 결측값을 이전 값으로 보간합니다.
          x = x.fillna(method='ffill').fillna(method='bfill')
          return x

        results = []

        for apt in apt_list:
            time1 = df[df['Gubun'] == apt]
            time1 = EDA(time1)
            time1 = make_month_merge(time1)
            time1 = fill(time1)

            data = time1['202212_real'].values
            ts = pd.Series(data)

            model = ExponentialSmoothing(ts, seasonal='add', seasonal_periods=12)
            fit_model = model.fit()

            forecast = fit_model.forecast(steps=12)
            results.append({'Gubun': apt, '202312_predict': forecast.iloc[-1]})

        result_df = pd.DataFrame(results)

        df_underrated = df.copy()
        df_underrated = df_underrated[df_underrated['GMonth'] <= 202212]
        df_underrated = df_underrated[df_underrated['Gubun'].isin(apt_list)]
        df_underrated = df_underrated.sort_values(by=['Gubun','Date'], ascending = False).groupby('Gubun').first().reset_index()

        df_test = df.copy()
        df_test = df_test[df_test['GMonth'] <= 202312]
        df_test = df_test[df_test['Gubun'].isin(apt_list)].sort_values(by=['Gubun','Date'], ascending = False).groupby('Gubun').first().reset_index()
        df_test['202312_real'] = df_test['GPrice'] / df_test['EArea']
        df_test['AptName_EArea2'] = df_test['AptName'] + '_' + df_test['Gubun'].apply(lambda x : x.split("_")[1]).astype(str)
        df_test = df_test[['Gubun','202312_real','AptName_EArea2']]

        final_df = pd.merge(result_df, df_underrated, on='Gubun', how='left')
        final_df['increase_rate'] = (final_df['202312_predict'] - final_df['202212_real']) / final_df['202212_real'] * 100
        final_df = final_df.sort_values(by='increase_rate', ascending=False)[['Gubun','202212_real','202312_predict']]
        final_df = pd.merge(final_df, df_test, on='Gubun', how='left')
        return final_df

    return price_prediction(df)