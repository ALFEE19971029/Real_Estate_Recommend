import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

def jiyeon_model(lower_limit, upper_limit):
    df = pd.read_csv(r'C:\Users\kshw1\Desktop\Study\중급프로그래밍\final_project\jiyeon\추가자료2.csv', index_col = 0)
    df_under = pd.read_csv(r"C:\Users\kshw1\Desktop\Study\중급프로그래밍\final_project\jiyeon\지연저평가아파트.csv", index_col = 0)

    def underrated_model(lower_limit, upper_limit, df_under):
        affordable_apartments = df_under[df_under['GPrice'].between(lower_limit, upper_limit, inclusive='both')].sort_values(by='GPrice', ascending = False)
        if affordable_apartments.empty:
            return []
        else:
            return affordable_apartments[:5]['RealGubun'].unique().tolist()

    def price_prediction(df):
        apt_list = underrated_model(lower_limit, upper_limit, df_under)
        if not apt_list:
            print("조건에 부합하는 아파트가 없습니다.")
            return pd.DataFrame()

        under_apt = df[df['RealGubun'].isin(apt_list)]
        under_apt['PriceArea'] = (under_apt['GPrice'] / under_apt['EArea']).round().astype(int)
        under_apt = under_apt[['GMonth', 'PriceArea', 'RealGubun']].groupby(['GMonth', 'RealGubun']).mean().reset_index()
        under_apt['Date'] = pd.to_datetime(under_apt['GMonth'], format='%Y%m')
        under_apt.set_index('Date', inplace=True)


        results = []

        for apt in apt_list:
            juso_apt = under_apt[under_apt['RealGubun'] == apt].copy()

            train_data = train_data = juso_apt[(juso_apt['GMonth'] >= 201001) & (juso_apt['GMonth'] <= 202212)].copy()

            # 전체 기간의 날짜 인덱스 생성
            full_index = pd.date_range(start="2010-01-01", end="2022-12-31", freq='MS')

            # 전체 기간의 데이터프레임 생성
            train_data = train_data.reindex(full_index)

            # 선형보간법으로 빈 값 채우기
            train_data = train_data.interpolate(method='linear')

            # SARIMA 모델 피팅
            model_sar = SARIMAX(train_data['PriceArea'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_sar_fit = model_sar.fit(disp=False)

            # 예측
            forecast = model_sar_fit.get_forecast(steps=12)
            predicted_value = forecast.predicted_mean.iloc[0]

            results.append({'RealGubun': apt, '202312_predict': predicted_value})

        result_df = pd.DataFrame(results)
        result_df

        df_underrated = df.copy()
        df_underrated = df_underrated[(df_underrated['GMonth'] <= 202212) & df_underrated['RealGubun'].isin(apt_list)]
        df_underrated['202212_real'] = df_underrated['GPrice'] / df_underrated['EArea']
        df_underrated = df_underrated.sort_values(by=['RealGubun','GMonth'], ascending = False).groupby('RealGubun').first().reset_index()

        df_test = df.copy()
        df_test = df_test[(df_test['GMonth'] <= 202312) & df_test['RealGubun'].isin(apt_list)].sort_values(by=['RealGubun','GMonth'], ascending = False).groupby('RealGubun').first().reset_index()
        df_test['202312_real'] = df_test['GPrice'] / df_test['EArea']
        df_test['AptName_EArea2'] = df_test['AptName'] + '_' + df_test['RealGubun'].apply(lambda x : x.split("_")[1]).astype(str)
        df_test = df_test[['RealGubun','202312_real','AptName_EArea2']]
        print(df_test)


        final_df = pd.merge(result_df, df_underrated, on='RealGubun', how='left')
        final_df['increase_rate'] = (final_df['202312_predict'] - final_df['202212_real']) / final_df['202212_real'] * 100
        final_df = final_df.sort_values(by='increase_rate', ascending=False)[['RealGubun','202212_real','202312_predict']]
        final_df = pd.merge(final_df, df_test, on='RealGubun', how='left').rename(columns = {'RealGubun':'Gubun'})

        return final_df

    return price_prediction(df)