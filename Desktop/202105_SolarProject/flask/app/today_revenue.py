from flask_restful import Resource, reqparse, request
import json
from db import Database
import pandas as pd
import numpy as np 
import datetime

# 당일 수익 현황 조회
class Today_revenue(Resource):

    # 1. DB 조회
    # 1-1 원본 데이터 조회
    def Row_search_DB(self, id, sub_id, dbc):
        sql = "select * from row_solardb where id='%s' AND sub_id=%s "%(id, sub_id)
        result = dbc.executeAll(sql)
        data = pd.DataFrame(result, columns=['Date', 'id', 'sub_id', '발전량', '발전용량', '위도', '경도', '기온', '강수량', '풍속', '풍향', '습도', '전운량'])
        data['Date'] = pd.to_datetime(data['Date'])
        data['년도'] = data['Date'].dt.year
        data['월'] = data['Date'].dt.month
        data['일'] = data['Date'].dt.day
        data['시'] = data['Date'].dt.hour
        
        return data

    # 1-2 단기 예측 테이블 조회
    def Short_search_DB(self, id, sub_id, year, month, day, dbc):
        sql = "select * from short_solardb where id='%s' AND sub_id=%s AND year=%d AND month=%d AND day=%d"%( id, sub_id, year, month, day)
        result = dbc.executeAll(sql)
        result = pd.DataFrame(result, columns=['id', 'sub_id', 'year', 'month', 'day', 'timestamp' ,'t_0', 't_1', 't_2','t_3','t_4','t_5','t_6','t_7','t_8','t_9','t_10','t_11','t_12','t_13','t_14','t_15','t_16','t_17','t_18','t_19','t_20','t_21','t_22','t_23'])

        return result

    # 1-3 시간대별 요금 테이블 조회
    def Charge_search_DB(self, id, sub_id, dbc):
        sql = "select * from charge where plantId_subId='%s_%s'"%(id, sub_id)
        result = dbc.executeAll(sql)
        result = pd.DataFrame(result, columns=['plantId_sub_Id', 't_0', 't_1', 't_2','t_3','t_4','t_5','t_6','t_7','t_8','t_9','t_10','t_11','t_12','t_13','t_14','t_15','t_16','t_17','t_18','t_19','t_20','t_21','t_22','t_23'])

        return result

    # 2. get 통신
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

            # 2-1 데이터 전처리
            # id, sub_id, timestamp 추출
            plant_id = args['plantId_subId'].split('_')[0]
            sub_id = args['plantId_subId'].split('_')[1]
            time_stamp = args['timestamp']
            time_stamp_p = datetime.datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
            year = time_stamp_p.year
            month = time_stamp_p.month
            day =  time_stamp_p.day
            hour = time_stamp_p.hour  
            
            dbc = Database()

            # 시간대별 요금 테이블 조회
            charge_data = self.Charge_search_DB(plant_id, sub_id, dbc)

            # 원본 테이블 조회
            row_data = self.Row_search_DB(plant_id, sub_id, dbc)
            
            filtering_data = row_data.loc[(row_data['년도']==year) & (row_data['월']==month) & (row_data['일']==day)]
            filtering_data = filtering_data.fillna(0)
            filtering_data = filtering_data.loc[filtering_data['시'] <= hour].reset_index(drop=True)


            # 단위 변환
            if (plant_id == 'plant19') | (plant_id == 'plant20') | (plant_id == 'plant21'):
                filtering_data['발전량'] = filtering_data['발전량'] / 1000 
            
            # 실제 수익 구하기 
            hour_result = [] 
            hour_real = [] 

            for i in filtering_data['시']:
                hour_rev = filtering_data['발전량'][i] * charge_data['t_%d'%(filtering_data['시'][i])]
                hour_result.append(hour_rev[0])

                isotime = str(year) + str(month) + '_'+ str(day) + '_' +str(i)
                isotime = datetime.datetime.strptime(isotime, '%Y%m_%d_%H')
                isotime = isotime.isoformat()
                hour_real.append(isotime)

            # 현재까지 수익
            todayRevenue = sum(hour_result) 

            # 실측 수익에 대한 그래프 값
            realGraph = {'X' : hour_real , 'Y' : hour_result} 

            ### 예측 테이블 ###
            filtering_pred_data = self.Short_search_DB(plant_id, sub_id, year, month, day, dbc)

            s_result = [] 
            hour_inx = hour + 1

            # 수익
            for j in filtering_pred_data.iloc[:,6:].columns[hour_inx:]:
                s_res = filtering_pred_data[j][0] * charge_data[j][0]
                s_result.append(s_res)

            # 남은 시간 예측 수익
            todayPredictedRevenue = sum(s_result) #3
            # 예측 수익
            predictedGraph_value = np.array(filtering_pred_data.iloc[:,6:].values * charge_data.iloc[:,1:].values)[0][hour_inx:] #5
            predictedGraph_value = predictedGraph_value.tolist()

            hour_short = [] 

            for k in range(hour_inx,24):
                isotime = str(year) + str(month) + '_'+ str(day) + '_' +str(k)
                isotime = datetime.datetime.strptime(isotime, '%Y%m_%d_%H')
                isotime = isotime.isoformat()
                hour_short.append(isotime)

            # 예측 수익에 대한 그래프 값
            predictedGraph = {'X': hour_short ,'Y' : predictedGraph_value}

           

            result['todayTotalRevenue'] = todayRevenue + todayPredictedRevenue
            #오늘의 시간 입력 받은거 만큼 수익 다 더한 값
            result['todayRevenue'] = todayRevenue
            # - 입력 받은 시간 부터의 예측 값에 수익 다 더한 값 
            result['todayPredictedRevenue'] = todayPredictedRevenue
            # 시간 입력 받은거 까지 각각 시간의 실제 수익
            result['realGraph'] = realGraph
            # 시간 입력 받은거 까지 각각 시간의 예측 수익
            result['predictedGraph'] = predictedGraph

            return result
         #variable is None:
            
        # 에러 예외 처리
        except Exception as e:
            return {"error" : str(e)}
