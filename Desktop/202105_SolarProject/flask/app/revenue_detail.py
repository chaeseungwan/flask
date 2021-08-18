from flask_restful import Resource, reqparse, request
import json
from db import Database
import pandas as pd
import numpy as np 
import datetime
from datetime import time, timedelta


class Revenue_detail(Resource):

    # 1. 데이터 조회   
    # 1-1 원본 데이터에서 id, sub_id를 통해 데이터 조회    
    def Row_search_DB(self, id, sub_id, dbc):
        sql = "select * from row_solardb where id='%s' AND sub_id=%s "%( id, sub_id)
        result = dbc.executeAll(sql)
        data = pd.DataFrame(result, columns=['Date', 'id', 'sub_id', '발전량', '발전용량', '위도', '경도', '기온', '강수량', '풍속', '풍향', '습도', '전운량'])
        # timestamp를 통해 년도, 월, 일, 시간 추출
        data['Date'] = pd.to_datetime(data['Date'])
        data['년도'] = data['Date'].dt.year
        data['월'] = data['Date'].dt.month
        data['일'] = data['Date'].dt.day
        data['시'] = data['Date'].dt.hour
        
        return data

    # 1-2 단기 예측 DB에서 id, sub_id를 통해 데이터 조회
    def Short_search_DB(self, id, sub_id, dbc):
        sql = "select * from short_solardb where id='%s' AND sub_id=%s "%( id, sub_id)
        result = dbc.executeAll(sql)
        result = pd.DataFrame(result, columns=['id', 'sub_id', 'year', 'month', 'day', 'timestamp' ,'t_0', 't_1', 't_2','t_3','t_4','t_5','t_6','t_7','t_8','t_9','t_10','t_11','t_12','t_13','t_14','t_15','t_16','t_17','t_18','t_19','t_20','t_21','t_22','t_23'])

        return result

    # 1-3 시간대별 요금 DB에서 id, sub_id를 통해 데이터 조회
    def Charge_search_DB(self, id, sub_id, dbc):
        sql = "select * from charge where plantId_subId='%s_%s'"%(id, sub_id)
        result = dbc.executeAll(sql)
        result = pd.DataFrame(result, columns=['plantId_sub_Id', 't_0', 't_1', 't_2','t_3','t_4','t_5','t_6','t_7','t_8','t_9','t_10','t_11','t_12','t_13','t_14','t_15','t_16','t_17','t_18','t_19','t_20','t_21','t_22','t_23'])

        return result

    # 1-4 회원정보 DB에서 id, sub_id를 통해 데이터 조회
    def User_search_DB(self, id, sub_id, dbc):
        sql = "select * from userlist where id='%s_%s'"%(id, sub_id)
        result = dbc.executeAll(sql)
        result = pd.DataFrame(result, columns=['plantId_sub_Id', 'password', 'investment'])
        result = result['investment'].values
        return result


    def get(self):
        try:
            ##요청값을 받아오고 결과값을 주기 위한 로직
            result = {}
            #request를 파싱 하는 부분
            parser = reqparse.RequestParser()
            #type을 key로 한 값에 대해서 string 명시
            parser.add_argument('plantId_subId', type=str)
            #parser.add_argument('sub_id', type=int)
            parser.add_argument('timestamp', type = str)

            
            #data를 key로 한 값에 대해서 list형태 명시
            #파싱하여 args에 할당
            args = parser.parse_args()

            # 2. 데이터 전처리
            # 2-1 입력받은 argument에서 plantid와 sub_id 추출, timestamp에서 년도와 월 추출    
            plant_id = args['plantId_subId'].split('_')[0]
            sub_id = args['plantId_subId'].split('_')[1]
            time_stamp = args['timestamp']          
            time_stamp_p = datetime.datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
            year = time_stamp_p.year
            month = time_stamp_p.month
                 

            dbc = Database()
            
            # 2-2 데이터베이스 조회
            charge_data = self.Charge_search_DB(plant_id, sub_id, dbc)
            row_data = self.Row_search_DB(plant_id, sub_id, dbc)
            short_data = self.Short_search_DB(plant_id, sub_id,dbc)
            userInvestment = self.User_search_DB(plant_id, sub_id, dbc)

            # 2-3 날짜를 기준으로 데이터 정렬
            row_data = row_data.sort_values('Date')
            row_now_data = row_data[row_data['Date'] <= time_stamp_p]
            


            
            # 2-4 수익금 및 누적 수익금 계산
            for i in range(24):
                row_now_data.loc[row_now_data['시'] == i, '수익금'] = row_now_data[row_now_data['시'] == i]['발전량'].values * charge_data['t_'+str(i)].values 

            row_now_all_value = row_now_data['수익금'].sum()
            all_mean = row_now_all_value / len(row_now_data.groupby(['년도','월']))
            row_month_table = row_now_data.groupby(['년도','월'])['수익금'].sum().reset_index()
            row_month_table['누적수익금'] = 0
            c_rev = [] 
            count = 0 
            # 누적 수익금 계산
            for i in row_month_table['수익금']:
                count += i
                c_rev.append(count)

            row_month_table['누적수익금'] = c_rev    

            # 2-5 월별(년월) 실제 누적 수익 계산 
            year_sl = year - 1 
            year_sl2 = year - 2 

            filtering_short_data = short_data[(short_data['year'] == year_sl) & (short_data['month'] == month)]
            # 이번달 예상 수익
            predictedRevenueThisMonth = filtering_short_data.iloc[:,6:]
            predictedRevenueThisMonth = sum(sum(predictedRevenueThisMonth.values * charge_data.iloc[:,1:].values))

            filtering_short_data2 = short_data[(short_data['year'] == year_sl2) & (short_data['month'] == month)]
            
            # 월별 작년대비 예상 수익 비교
            compareLastYearOfMonth = filtering_short_data2.iloc[:,6:]
            compareLastYearOfMonth = predictedRevenueThisMonth - sum(sum(compareLastYearOfMonth.values * charge_data.iloc[:,1:].values))

            # 실제 누적 수익
            actualRevenue = row_now_all_value - userInvestment

            row_month_table['실제누적수익'] = row_month_table['누적수익금'] - userInvestment
            row_month_table['일시'] = row_month_table['년도'].astype(str) + '.' + row_month_table['월'].astype(str)
            row_month_table = row_month_table[['일시','수익금','누적수익금','실제누적수익','년도','월']].iloc[:,:-2]

            li = []
            for i in range(len(row_month_table)):
                isotime = str(row_month_table['일시'][i]) + '_' + '01'
                isotime = datetime.datetime.strptime(isotime, '%Y.%m_%d')
                isotime = isotime.isoformat()
                li.append(isotime)
            row_month_table['일시'] = li



            # 월별 실제 누적 수익
            cumulativeRevenueList = row_month_table.values.tolist()

            pred_month_result = []

            # 날짜 계산
            month_sl = month + 1 

            if month == 12: 
                month = 0
                month_sl = month + 1 
                year = year + 1  

            # 예상 수익 * 시간대별 요금
            for i in range(month_sl, 13):
                filtering_short_data3 = short_data[(short_data['year'] == year_sl) & (short_data['month'] == i)]
                compareLastYearOfMonth2 = filtering_short_data3.iloc[:,6:]
                month_revenue = sum(sum(compareLastYearOfMonth2.values * charge_data.iloc[:,1:].values))
                pre_isotime  = str(year) + str(i) + '_' + '01'
                pre_isotime = datetime.datetime.strptime(pre_isotime, '%Y%m_%d')
                pre_isotime = pre_isotime.isoformat()
                pred_month_result.append([pre_isotime, month_revenue])

            pred_X = []
            pred_Y = [] 

            # 월별, 누적 수익
            for i in range(len(pred_month_result)):
                pred_X.append(pred_month_result[i][0])
                pred_Y.append(pred_month_result[i][1])

            # 2-6 계산된 결과값 return
            # 누적 수익 그래프 값
            accumulatedRevenueGraph = {"Real_X" : row_month_table['일시'].values.tolist(), "Real_Y" : row_month_table['누적수익금'].values.tolist(),"Pred_X" : pred_X, "pred_Y" : pred_Y}

            result['success'] = True

            # 현재 월 예상 수익 (현재 월이 없기 때문에 입력 받은 년도 -1 의 예측 수익 보여줌 )
            result['predictedRevenueThisMonth'] = predictedRevenueThisMonth.tolist()
            #각 월별 수익 평균 비교
            result['compareMonthAverage'] = predictedRevenueThisMonth - all_mean
            #작년과 수익 비교  
            result['compareLastYearOfMonth'] = compareLastYearOfMonth.tolist()
            #원금 + 실제 누적 수익 
            result['totalRevenue'] = row_now_all_value.tolist()
            #투자원금
            result['userInvestment'] = userInvestment.tolist()
            #투자수익 - 원금 
            result['actualRevenue'] = actualRevenue.tolist()
            #누적 수익 그래프 값 
            result['accumulatedRevenueGraph'] = accumulatedRevenueGraph
            #월별 실제 누적수익
            result['cumulativeRevenueList'] = cumulativeRevenueList

            return result
         #variable is None:
            
        # 에러 발생시 예외처리
        except Exception as e:
            return {"error" : str(e)}
