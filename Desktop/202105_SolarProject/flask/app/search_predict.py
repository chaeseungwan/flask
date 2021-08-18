from datetime import time, timedelta
from flask_restful import Resource, reqparse, request
import json
from db import Database
import pandas as pd
import numpy as np 
import datetime
from dateutil.relativedelta import relativedelta

class Search_predict(Resource):

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

    # 1-3 장기 예측 테이블 조회
    def Long_search_db(self, id, sub_id, year, month, day, dbc):
        sql = "select * from long_solardb where id='%s' AND sub_id=%s AND year=%d AND month=%d AND day=%d"%( id, sub_id, year, month, day)
        result = dbc.executeAll(sql)
        result = pd.DataFrame(result, columns=['id', 'sub_id', 'year', 'month', 'day', 'timestamp' ,'t1_0', 't1_1', 't1_2', 't1_3', 't1_4', 't1_5', 't1_6', 't1_7', 't1_8', 't1_9', 't1_10', 't1_11', 't1_12', 't1_13', 't1_14', 't1_15', 't1_16', 't1_17', 't1_18', 't1_19', 't1_20', 't1_21', 't1_22', 't1_23', 't2_0', 't2_1', 't2_2', 't2_3', 't2_4', 't2_5', 't2_6', 't2_7', 't2_8', 't2_9', 't2_10', 't2_11', 't2_12', 't2_13', 't2_14', 't2_15', 't2_16', 't2_17', 't2_18', 't2_19', 't2_20', 't2_21', 't2_22', 't2_23', 't3_0', 't3_1', 't3_2', 't3_3', 't3_4', 't3_5', 't3_6', 't3_7', 't3_8', 't3_9', 't3_10', 't3_11', 't3_12', 't3_13', 't3_14', 't3_15', 't3_16', 't3_17', 't3_18', 't3_19', 't3_20', 't3_21', 't3_22', 't3_23', 't4_0', 't4_1', 't4_2', 't4_3', 't4_4', 't4_5', 't4_6', 't4_7', 't4_8', 't4_9', 't4_10', 't4_11', 't4_12', 't4_13', 't4_14', 't4_15', 't4_16', 't4_17', 't4_18', 't4_19', 't4_20', 't4_21', 't4_22', 't4_23', 't5_0', 't5_1', 't5_2', 't5_3', 't5_4', 't5_5', 't5_6', 't5_7', 't5_8', 't5_9', 't5_10', 't5_11', 't5_12', 't5_13', 't5_14', 't5_15', 't5_16', 't5_17', 't5_18', 't5_19', 't5_20', 't5_21', 't5_22', 't5_23', 't6_0', 't6_1', 't6_2', 't6_3', 't6_4', 't6_5', 't6_6', 't6_7', 't6_8', 't6_9', 't6_10', 't6_11', 't6_12', 't6_13', 't6_14', 't6_15', 't6_16', 't6_17', 't6_18', 't6_19', 't6_20', 't6_21', 't6_22', 't6_23', 't7_0', 't7_1', 't7_2', 't7_3', 't7_4', 't7_5', 't7_6', 't7_7', 't7_8', 't7_9', 't7_10', 't7_11', 't7_12', 't7_13', 't7_14', 't7_15', 't7_16', 't7_17', 't7_18', 't7_19', 't7_20', 't7_21', 't7_22', 't7_23', 't8_0', 't8_1', 't8_2', 't8_3', 't8_4', 't8_5', 't8_6', 't8_7', 't8_8', 't8_9', 't8_10', 't8_11', 't8_12', 't8_13', 't8_14', 't8_15', 't8_16', 't8_17', 't8_18', 't8_19', 't8_20', 't8_21', 't8_22', 't8_23', 't9_0', 't9_1', 't9_2', 't9_3', 't9_4', 't9_5', 't9_6', 't9_7', 't9_8', 't9_9', 't9_10', 't9_11', 't9_12', 't9_13', 't9_14', 't9_15', 't9_16', 't9_17', 't9_18', 't9_19', 't9_20', 't9_21', 't9_22', 't9_23', 't10_0', 't10_1', 't10_2', 't10_3', 't10_4', 't10_5', 't10_6', 't10_7', 't10_8', 't10_9', 't10_10', 't10_11', 't10_12', 't10_13', 't10_14', 't10_15', 't10_16', 't10_17', 't10_18', 't10_19', 't10_20', 't10_21', 't10_22', 't10_23'])

        return result
    
    # 1-4 시간대별 요금 DB 조회
    def Charge_search_DB(self, id, sub_id, dbc):
        sql = "select * from charge where plantId_subId='%s_%s'"%(id, sub_id)
        result = dbc.executeAll(sql)
        result = pd.DataFrame(result, columns=['plantId_sub_Id', 't_0', 't_1', 't_2','t_3','t_4','t_5','t_6','t_7','t_8','t_9','t_10','t_11','t_12','t_13','t_14','t_15','t_16','t_17','t_18','t_19','t_20','t_21','t_22','t_23'])

        return result

    def get(self):
        try:
            ##요청값을 받아오고 결과값을 주기 위한 로직
            result = {}
            #request를 파싱 하는 부분
            parser = reqparse.RequestParser()
            #type을 key로 한 값에 대해서 string 명시
            parser.add_argument('plantId_subId', type=str)
            parser.add_argument('timestamp', type = str)
            parser.add_argument('periodType', type=str)

            
            #data를 key로 한 값에 대해서 list형태 명시
            #파싱하여 args에 할당
            args = parser.parse_args()

            # 2. 데이터 전처리
            # 2-1 id, sub_id, date, type(일,주,월,년) 추출
            plant_id = args['plantId_subId'].split('_')[0]
            sub_id = args['plantId_subId'].split('_')[1]
            time_stamp = args['timestamp']
            period_type = args['periodType']


            time_stamp_p = datetime.datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
            year = time_stamp_p.year
            month = time_stamp_p.month
            day =  time_stamp_p.day
            hour = time_stamp_p.hour

            dbc = Database()
            
           # 2-2 시간대별 요금 데이터 tolist
            charge_data = self.Charge_search_DB(plant_id, sub_id,dbc)
            charge_data = charge_data.iloc[:,1:].values.reshape(-1)
            charge_data = charge_data.tolist()

            # 2-3 일별 발전량 및 수익 계산
            if period_type == 'day':
                row_data = self.Row_search_DB(plant_id, sub_id, dbc)

                # 단위 잘못된 plant 전처리   
                if (plant_id == 'plant19') | (plant_id == 'plant20') | (plant_id == 'plant21'):
                    row_data['발전량'] = row_data['발전량'] / 1000 

                # 입력 받은 argment에 맞게 원본 데이터 전처리
                row_data = row_data[(row_data['년도'] == year) & (row_data['월'] == month) & (row_data['일'] == day)]
                row_data = row_data.fillna(0)
                row_data = row_data.loc[row_data['시'] <= hour].reset_index(drop=True)
            
                filtering_row = np.array(row_data['발전량'].values).tolist()
       
                # 단기 예측 테이블 조회
                short_solardb = self.Short_search_DB(plant_id, sub_id, year, month, day, dbc)
                
                short_idx = 6 + (hour + 1)
                filtering_short = short_solardb.iloc[:,short_idx:].values.reshape(-1)
                filtering_short = filtering_short.tolist()

                # 시간(하루 24시간, 24개)
                hour_to_result = [] 
                for i in range(24):
                    hour_to_result.append([i])

                # 원본 데이터 및 단기 테이블 통합
                conc_data = filtering_row + filtering_short
                # 발전량 * 요금
                conc_charge = [x*y for x,y in zip(conc_data, charge_data)]
                
                # 발전량
                conc_data_result = []
                for i in conc_data:
                    conc_data_result.append([i])

                # 수익
                conc_charge_result = []
                for i in conc_charge:
                    conc_charge_result.append([i])

                # 누적 수익
                cumulative_revenue = []
                count = 0 
                for i in conc_charge_result:
                    count += i[0]
                    cumulative_revenue.append([count])
               
                # list append
                revenueFromPowerList3 = list(map(list.__add__, conc_data_result, conc_charge_result))
                revenueFromPowerList2 = list(map(list.__add__, hour_to_result ,revenueFromPowerList3))
                revenueFromPowerList = list(map(list.__add__, revenueFromPowerList2, cumulative_revenue))

                # 실제 시간
                real_hour = []
                for i in range(hour+1):
                    isoformat = str(year)+str(month)+str(day)+ '_'+ str(i)
                    isoformat = datetime.datetime.strptime(isoformat, '%Y%m%d_%H')
                    isoformat = isoformat.isoformat()
                    real_hour.append(isoformat)
                
                # 예측 시간
                pred_hour = [] 
                for i in range(hour+1, 24):
                    isoformat = str(year)+str(month)+str(day)+ '_'+ str(i)
                    isoformat = datetime.datetime.strptime(isoformat, '%Y%m%d_%H')
                    isoformat = isoformat.isoformat()
                    pred_hour.append(isoformat)

                
                # 실제 발전량
                realPowerGraph = {'X': real_hour ,'Y':filtering_row}
                # 예측 발전량
                predictedPowerGraph = {'X': pred_hour,'Y':filtering_short}

                result['success'] = True
                result['realPowerGraph'] = realPowerGraph
                result['predictedPowerGraph'] = predictedPowerGraph
                # 발전량 대비 누적 수익(시간 단위)
                result['revenueFromPowerList'] = revenueFromPowerList


            # 2-4 주별 발전량 및 수익 계산
            elif period_type == 'week':

                # 원본 DB 조회
                row_data = self.Row_search_DB(plant_id, sub_id, dbc)
                
                # 단위 잘못된 plant 전처리
                if (plant_id == 'plant19') | (plant_id == 'plant20') | (plant_id == 'plant21'):
                    row_data['발전량'] = row_data['발전량'] / 1000 

                row_data = row_data[(row_data['년도'] == year) & (row_data['월'] == month) & (row_data['일'] == day)]
                row_data = row_data.fillna(0)
                row_data = row_data.loc[row_data['시'] <= hour].reset_index(drop=True)

                filtering_row = np.array(row_data['발전량'].values).tolist()

                idx = 6 + (hour + 1) 
                # 장기 예측 테이블 조회
                long_data = self.Long_search_db(plant_id, sub_id, year, month, day, dbc)
            
                filtering_long = long_data.iloc[:,idx:idx + 168 -len(row_data)].values
                filtering_long = filtering_long.reshape(-1)
                filtering_long = filtering_long.tolist()

                # # 일(day), 시간
                hour_to_result = [] 
                for j in range(1, 8):
                    for i in range(24):
                        hour_to_result.append([i])

                # 원본 데이터 + 장기 테이블
                conc_data = filtering_row + filtering_long


                # 시간대별 요금 DB 조회
                charge_data = self.Charge_search_DB(plant_id, sub_id, dbc)
                charge_data = charge_data.iloc[:,1:].values.reshape(-1)
                charge_data = charge_data.tolist()

                # 수익금 계산
                conc_charge = [x*y for x,y in zip(conc_data, charge_data * 7)]
 
                # 발전량
                conc_data_result = []
                for i in conc_data:
                    conc_data_result.append([i])

                # 수익
                conc_charge_result = []
                for i in conc_charge:
                    conc_charge_result.append([i])

                # 누적 수익
                cumulative_revenue = []
                count = 0
                for i in conc_charge_result:
                    count += i[0]
                    cumulative_revenue.append([count])

                # list append
                revenueFromPowerList3 = list(map(list.__add__, conc_data_result, conc_charge_result))
                revenueFromPowerList2 = list(map(list.__add__, hour_to_result ,revenueFromPowerList3))
                revenueFromPowerList = list(map(list.__add__, revenueFromPowerList2, cumulative_revenue))


                hour_result = []
                for j in range(1, 8):
                    for i in range(24):
                        isoformat = str(year)+str(month)+str(day)
                        
                        if j >= 2:
                            isoformat_data = datetime.datetime.strptime(isoformat, '%Y%m%d') + timedelta(days=j-1)
                            isoformat_data = isoformat_data.isoformat()
                            hour_result.append(isoformat_data)
                        else:
                            isoformat_data = datetime.datetime.strptime(isoformat, '%Y%m%d')
                            isoformat_data = isoformat_data.isoformat()
                            hour_result.append(isoformat_data)


                # 실제 시간
                real_hour = hour_result[:len(filtering_row)]

                # 예측 시간
                pred_hour = hour_result[len(filtering_row):]

                df = pd.DataFrame(list(zip(pred_hour, filtering_long)), columns = ['시간','수익'])
                df_group = df.groupby('시간')['수익'].sum().reset_index()

        
                
                realPowerGraph = {'X': real_hour[0] ,'Y':[sum(filtering_row)]}
                predictedPowerGraph = {'X': df_group['시간'].tolist(), 'Y':df_group['수익'].tolist()}

                result['success'] = True

                # 실제 발전량
                result['realPowerGraph'] = realPowerGraph
                # 예측 발전량
                result['predictedPowerGraph'] = predictedPowerGraph
                # 발전량 대비 누적 수익(시간, 발전량, 수익, 누적수익)
                result['revenueFromPowerList'] = revenueFromPowerList


            # 2-5 월별 발전량 및 수익 계산
            elif period_type == 'month':

                # 원본 데이터 조회
                row_data = self.Row_search_DB(plant_id, sub_id, dbc)
                
                # 단위 잘못된 plant 전처리
                if (plant_id == 'plant19') | (plant_id == 'plant20') | (plant_id == 'plant21'):
                    row_data['발전량'] = row_data['발전량'] / 1000 


                

                row_data = row_data.sort_values(by=['년도','월', '일'])

                row_data = row_data[(row_data['년도'] == year) & (row_data['월'] == month) & (row_data['일'] == day)]
                row_data = row_data.fillna(0)
                row_data = row_data.loc[row_data['시'] <= hour].reset_index(drop=True)

            
                filtering_row = np.array(row_data['발전량'].values).tolist()

                idx = 6 + (hour + 1)
                # 장기 예측 테이블 조회
                long_data = self.Long_search_db(plant_id, sub_id, year, month, day, dbc)
                filtering_long = long_data.iloc[:,idx:].values
                filtering_long = filtering_long.reshape(-1)
                filtering_long = filtering_long.tolist()

                # 날짜 전처리
                time_stamp = time_stamp.split(' ')[0]
                time_stamp = datetime.datetime.strptime(time_stamp, '%Y-%m-%d')
                ago_timestamp = time_stamp - timedelta(days=365)
                next_timestamp = ago_timestamp + relativedelta(months=1)

                # 원본 데이터 조회
                row_data = self.Row_search_DB(plant_id, sub_id, dbc)

                row_data = row_data.sort_values(by=['년도','월', '일']) 
                row_data = row_data.loc[(ago_timestamp <= row_data['Date']) & (row_data['Date'] < next_timestamp)].reset_index(drop=True)
                remind_data = row_data.copy()
                remind_data = remind_data['발전량'].iloc[240:]
                remind_data = remind_data.values
                remind_data = remind_data.tolist()

                # 시간 
                hour_to_result = [] 
                count_day = int((len(filtering_row) + len(filtering_long) + len(remind_data)) / 24)
                for j in range(0, count_day):
                    for i in range(24):
                        hour_to_result.append([i])

                # 원본 데이터 + 장기 예측 데이터
                conc_data = filtering_row + filtering_long
                
                # 원본 데이터 + 장기 예측 데이터 + 작년 지난 달 데이터(11일~)
                conc_data_merge = conc_data + remind_data

                # 시간대별 요금
                conc_charge = [x*y for x,y in zip(conc_data_merge, charge_data * count_day)]
                    
                # 발전량   
                conc_data_merge_result = []
                for i in conc_data_merge:
                    conc_data_merge_result.append([i])

                # 수익
                conc_charge_result = []
                for i in conc_charge:
                    conc_charge_result.append([i])

                # 누적 수익
                cumulative_revenue = []
                count = 0
                for i in conc_charge_result:
                    count += i[0]
                    cumulative_revenue.append([count])

                # list append
                revenueFromPowerList3 = list(map(list.__add__, conc_data_merge_result, conc_charge_result))
                revenueFromPowerList2 = list(map(list.__add__, hour_to_result, revenueFromPowerList3))
                revenueFromPowerList = list(map(list.__add__, revenueFromPowerList2, cumulative_revenue))

                hour_result = []
                for j in range(1, count_day+1):
                     for i in range(24):
                        a = str(year) + str(month) + str(j) 
                        isotime = datetime.datetime.strptime(a, '%Y%m%d')
                        isotime = isotime.isoformat()
                        hour_result.append(isotime)

                

                # 실제 시간
                real_hour = hour_result[:len(filtering_row)]
                # 예측 시간
                pred_hour = hour_result[len(filtering_row):]
          
                # 예측 발전량
                predictedPowerGraph_value = filtering_long + remind_data

                df = pd.DataFrame(list(zip(pred_hour, predictedPowerGraph_value)), columns = ['시간','수익'])
                df_group = df.groupby('시간')['수익'].sum().reset_index()

                # json 형태로 return
                realPowerGraph = {'X': [real_hour[0]] ,'Y':[sum(filtering_row)]}
                predictedPowerGraph = {'X': df_group['시간'].tolist(),'Y':df_group['수익'].tolist()}

                result['success'] = True
                # 실제 발전량
                result['realPowerGraph'] = realPowerGraph
                # 예측 발전량
                result['predictedPowerGraph'] = predictedPowerGraph
                # 발전량 대비 누적 수익(시간, 발전량, 수익, 누적 수익)
                result['revenueFromPowerList'] = revenueFromPowerList


            # 2-6 월별 발전량 및 수익 계산
            elif period_type == 'year':
                
                # 원본 데이터 조회
                data = self.Row_search_DB(plant_id, sub_id, dbc)
               
                # 단위 잘못된 plant 전처리
                if (plant_id == 'plant19') | (plant_id == 'plant20') | (plant_id == 'plant21'):
                    data['발전량'] = data['발전량'] / 1000 

                data = data.sort_values(by=['년도','월', '일']).reset_index(drop=True)
                data = data.fillna(0)

                data = data.loc[data['년도'] == year-1]
                
                
                filtering_row = data['발전량'].values.tolist()

                # 시간(1년)
                hour_to_result = [] 
                for j in range(1, 366):
                    for i in range(24):
                        hour_to_result.append([i])

                # 수익 계산
                conc_charge = [x*y for x,y in zip(filtering_row, charge_data * 365)]

                # 작년 데이터
                conv_data = []
                for i in filtering_row:
                    conv_data.append([i])
                
                # 수익
                conc_charge_result = []
                for i in conc_charge:
                    conc_charge_result.append([i])

                # 누적 수익
                cumulative_revenue = []
                count = 0
                for i in conc_charge_result:
                    count += i[0]
                    cumulative_revenue.append([count])

                # list append
                revenueFromPowerList3 = list(map(list.__add__, conv_data, conc_charge_result))
                revenueFromPowerList2 = list(map(list.__add__, hour_to_result, revenueFromPowerList3))
                revenueFromPowerList = list(map(list.__add__, revenueFromPowerList2, cumulative_revenue))

                hour_result = [] 
                for j in range(1, 366):
                    for i in range(24):
                        a = str(year) + str(month) + '_' + str(day)
                        if j >= 2:
                            isotime = datetime.datetime.strptime(a, '%Y%m_%d') + timedelta(days = j - 1)
                            isotime = isotime.isoformat()
                            isotime = isotime[:7]
                            hour_result.append(isotime)
                        else:
                            isotime = datetime.datetime.strptime(a, '%Y%m_%d')
                            isotime = isotime.isoformat()
                            isotime = isotime[:7]
                            hour_result.append(isotime)

                df = pd.DataFrame(list(zip(hour_result, filtering_row)), columns = ['날짜','수익'])
                df_group = df.groupby('날짜')['수익'].sum().reset_index()

                time_li = []
                for i in range(len(df_group)):
                    isotime = datetime.datetime.strptime(df_group['날짜'][i], '%Y-%m')
                    isotime = isotime.isoformat()
                    time_li.append(isotime)
                
                df_group['날짜'] = time_li

          
                # 실제 발전량
                realPowerGraph = {'X': df_group['날짜'].tolist() ,'Y':df_group['수익'].tolist()}
                # 예측 발전량
                predictedPowerGraph = {'X': df_group['날짜'].tolist(),'Y':df_group['수익'].tolist()}

                result['success'] = True
                result['realPowerGraph'] = realPowerGraph
                result['predictedPowerGraph'] = predictedPowerGraph
                # 발전량 대비 누적 수익(시간, 발전량, 수익, 누적 수익)
                result['revenueFromPowerList'] = revenueFromPowerList


            return result

        # 에러 발생시 예외처리
        except Exception as e:
            return {"error" : str(e)}
