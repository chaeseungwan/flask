from db import Database
import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
import datetime
from flask_restful import Resource, reqparse, request
import psycopg2
import joblib

# 1. DB 조회
# 1-1 DB에서 원본 데이터 조회
def LoadDB():
    sql = "select * from row_solardb"
    dbc = Database()
    result = dbc.executeAll(sql)
    data = pd.DataFrame(result, columns=['Date', 'id', 'sub_id', '발전량', '발전용량', '위도', '경도', '기온', '강수량', '풍속', '풍향', '습도', '전운량'])
    # datetime에서 년,월,일,시 추출
    data['Date'] = pd.to_datetime(data['Date'])
    data['년도'] = data['Date'].dt.year 
    data['월'] = data['Date'].dt.month 
    data['일'] = data['Date'].dt.day 
    data['시'] = data['Date'].dt.hour

    df = pd.DataFrame(columns=data.columns)


    # 각 plant, sub_id별 마지막 행 16개 가져오기
    for ide in data['id'].unique():
        for sub_id in data.loc[data['id']==ide]['sub_id'].unique():
            choice = data.loc[(data['id']== ide) & (data['sub_id']==sub_id)].iloc[-1]
            year = choice['년도']
            month = choice['월']
            day = choice['일']
            
            plant = data.loc[(data['id']==ide) & (data['sub_id']==sub_id) & (data['년도']==year) & (data['월']==month) & (data['일']==day)]      
            df = pd.concat([df, plant], axis=0)
        
    df = df.reset_index(drop=True)

    return df

# 2. 데이터 전처리
def Processing():
    # db load, merge 및 결측치처리
    data = LoadDB()        
    # 단위 변환(해당 plant 단위가 잘못되어 있어 단위 통일)
    train_true = data.loc[(data['id']!='plant19') & (data['id']!='plant20') & (data['id']!='plant21')]
    train_false = data.loc[(data['id']=='plant19') | (data['id']=='plant20') | (data['id']=='plant21')]
    train_false['발전량'] = train_false['발전량'] / 1000

    # merge(단위가 잘못되어 있는 데이터 수정 후 데이터 통합)
    data = pd.concat([train_true, train_false]).reset_index(drop=True)


    # 시간 설정 및 결측치 처리(발전량이 있는 시간대로 시간 제한)
    data = data[(data['시'] > 3) & (data['시'] < 20 )].reset_index(drop=True)
    data.fillna(0,inplace = True)


    return data


# 2-1 LSTM 모델에 들어가기 위한 데이터 형식 만듬
# target : 발전량, start_index:시작 index, end_index:종료 index, history_size-target_size:과거 관측치 만큼 샘플링, step:데이터 간격
def multivariate_data(dataset, target, start_index, end_index, history_size,
                    target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index #+ history_size
    if end_index is None:
        end_index = len(dataset) #- target_size

    for i in range(start_index, end_index, step):
        indices = range(i-history_size, i)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data)


# 3. 단기 모델 정의
def Short_Model():
    short_all = pd.DataFrame(columns = ['4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','id','sub_id','time'])

    
    time_list = [] 
    
    
    data = Processing()
    
    # 3-1 단기모델을 위한 데이터 전처리
    # plant, sub_id의 unique한 값을 가지고 옴
    for i in data['id'].unique():
        print(i)
        for j in data[data['id'] == i]['sub_id'].unique():
            print(j)
            if i == 'plant19' : 
                print('plant19 is None')
            else: 
                te_data = data[(data['id'] == i) & (data['sub_id'] == j)].reset_index(drop=True)
        
                time_list = te_data['Date'].reset_index(drop = True)            
                time_list = [(datetime.datetime.strptime(str(time_list.iloc[-1]), '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')]


                # 발전용량으로 발전량을 나눔(정규화)
                capacity = te_data['발전용량'].iloc[0] 
                te_data['value_nom'] = te_data['발전량'] / capacity

                x_te_data = te_data[['기온','강수량','습도','풍속', '풍향', '시']] # 독립변수
                y_te_data = te_data[['value_nom']] # 종속변수


                # 독립변수(기온, 강수량, 습도 등 0~1사이로 min-max 정규화)
                scalerfile = './minmax_pkl/%s_%d.pkl' %(i, j)
                scaler = joblib.load(scalerfile) 
                x_te_data = scaler.transform(x_te_data)

                # 단기 모델의 들어간 데이터 형태 맞춤
                x_te = multivariate_data(x_te_data, y_te_data, 0, None, 16,16,16)  
                

                # 저장된 모델을 불러와서 예측
                short_model = tf.keras.models.load_model('Short_model/' + str(i) +'_' + str(j) + '.h5')
                short_y_pred = short_model.predict(x_te)

                short_y_pred = short_y_pred * capacity
                

                # 발전량이 발전용량의 10% 미만의 경우 0으로 처리
                for k in range(len(short_y_pred)):
                    for h in range(len(short_y_pred[k])):
                        if short_y_pred[k][h] < capacity * 0.1 :
                            short_y_pred[k][h] = 0 

                
                # 예측 결과를 dataframe으로 변환
                short_pred_table = pd.DataFrame(short_y_pred, columns = ['4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']) 
                short_pred_table['id'] = i
                short_pred_table['sub_id'] = j
                short_pred_table['time'] = time_list

                short_all = pd.concat([short_all, short_pred_table])
                

                # 해당 모델의 경우 4~19시까지 에측하므로 나머지 시간에 대해서는 발전량을 0으로 처리
                short_all['0'] = 0
                short_all['1'] = 0
                short_all['2'] = 0
                short_all['3'] = 0
                short_all['20'] = 0
                short_all['21'] = 0
                short_all['22'] = 0
                short_all['23'] = 0
                
                # 년도, 월, 일 추가
                short_all['time'] = pd.to_datetime(short_all['time'])
                short_all['year'] = short_all['time'].dt.year
                short_all['month'] = short_all['time'].dt.month
                short_all['day'] = short_all['time'].dt.day

                short_all = short_all[[ 'id', 'sub_id', 'year', 'month', 'day','time','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                            '18', '19', '20', '21', '22', '23']]
    
    return short_all

# 4. 장기 모델 정의
# 장기 모델이며, 16시간을 보고 160시간을 예측
def Long_Model():
    long_all = pd.DataFrame(columns = ['1_4','1_5','1_6','1_7','1_8','1_9','1_10','1_11','1_12','1_13','1_14','1_15','1_16','1_17','1_18','1_19','2_4','2_5','2_6','2_7','2_8','2_9','2_10','2_11','2_12','2_13','2_14','2_15','2_16','2_17','2_18','2_19','3_4','3_5','3_6','3_7','3_8','3_9','3_10','3_11','3_12','3_13','3_14','3_15','3_16','3_17','3_18','3_19','4_4','4_5','4_6','4_7','4_8','4_9','4_10','4_11','4_12','4_13','4_14','4_15','4_16','4_17','4_18','4_19','5_4','5_5','5_6','5_7','5_8','5_9','5_10','5_11','5_12','5_13','5_14','5_15','5_16','5_17','5_18','5_19','6_4','6_5','6_6','6_7','6_8','6_9','6_10','6_11','6_12','6_13','6_14','6_15','6_16','6_17','6_18','6_19','7_4','7_5','7_6','7_7','7_8','7_9','7_10','7_11','7_12','7_13','7_14','7_15','7_16','7_17','7_18','7_19','8_4','8_5','8_6','8_7','8_8','8_9','8_10','8_11','8_12','8_13','8_14','8_15','8_16','8_17','8_18','8_19','9_4','9_5','9_6','9_7','9_8','9_9','9_10','9_11','9_12','9_13','9_14','9_15','9_16','9_17','9_18','9_19','10_4','10_5','10_6','10_7','10_8','10_9','10_10','10_11','10_12','10_13','10_14','10_15','10_16','10_17','10_18','10_19','id','sub_id','time'])


    time_list = [] 
    
    
    data = Processing()
    

    # 4-1 장기 모델 형식에 맞도록 데이터 전처리
    # plant, sub_id의 unique한 값을 가지고 옴
    for i in data['id'].unique():
        print(i)
        for j in data[data['id'] == i]['sub_id'].unique():
            print(j)
            if i == 'plant19' : 
                print('plant19 is None')
            else: 
                te_data = data[(data['id'] == i) & (data['sub_id'] == j)].reset_index(drop=True)
            
                time_list = te_data['Date'].reset_index(drop = True)            
                time_list = [(datetime.datetime.strptime(str(time_list.iloc[-1]), '%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')]

                    # 발전용량으로 발전량을 나눔(정규화)
                capacity = te_data['발전용량'].iloc[0]
                te_data['value_nom'] = te_data['발전량'] / capacity

                x_te_data = te_data[['기온','강수량','습도','풍속', '풍향', '시']] # 독립변수
                y_te_data = te_data[['value_nom']] # 종속변수


                # 독립변수(기온, 강수량, 습도 등 0~1사이로 min-max 정규화)
                scalerfile = './minmax_pkl/%s_%d.pkl' %(i, j)
                scaler = joblib.load(scalerfile) 
                x_te_data = scaler.transform(x_te_data)


                # 장기 모델의 들어간 데이터 형태 맞춤
                x_te = multivariate_data(x_te_data, y_te_data, 0, None, 16,16,16)  

                # 장기모델을 불러와서 예측
                long_model = tf.keras.models.load_model('long_model/' + str(i) +'_' + str(j) + '.h5')
                long_y_pred = long_model.predict(x_te)

                long_y_pred = long_y_pred * capacity
                
                # 발전량이 발전용량의 10%미만 0으로 처리
                for k in range(len(long_y_pred)):
                    for h in range(len(long_y_pred[k])):
                        if long_y_pred[k][h] < capacity * 0.1 :
                            long_y_pred[k][h] = 0 


                # 모델에서 예측된 값을 dataframe으로 변환
                long_pred_table = pd.DataFrame(long_y_pred, columns = ['1_4','1_5','1_6','1_7','1_8','1_9','1_10','1_11','1_12','1_13','1_14','1_15','1_16','1_17','1_18','1_19','2_4','2_5','2_6','2_7','2_8','2_9','2_10','2_11','2_12','2_13','2_14','2_15','2_16','2_17','2_18','2_19','3_4','3_5','3_6','3_7','3_8','3_9','3_10','3_11','3_12','3_13','3_14','3_15','3_16','3_17','3_18','3_19','4_4','4_5','4_6','4_7','4_8','4_9','4_10','4_11','4_12','4_13','4_14','4_15','4_16','4_17','4_18','4_19','5_4','5_5','5_6','5_7','5_8','5_9','5_10','5_11','5_12','5_13','5_14','5_15','5_16','5_17','5_18','5_19','6_4','6_5','6_6','6_7','6_8','6_9','6_10','6_11','6_12','6_13','6_14','6_15','6_16','6_17','6_18','6_19','7_4','7_5','7_6','7_7','7_8','7_9','7_10','7_11','7_12','7_13','7_14','7_15','7_16','7_17','7_18','7_19','8_4','8_5','8_6','8_7','8_8','8_9','8_10','8_11','8_12','8_13','8_14','8_15','8_16','8_17','8_18','8_19','9_4','9_5','9_6','9_7','9_8','9_9','9_10','9_11','9_12','9_13','9_14','9_15','9_16','9_17','9_18','9_19','10_4','10_5','10_6','10_7','10_8','10_9','10_10','10_11','10_12','10_13','10_14','10_15','10_16','10_17','10_18','10_19']) 
                long_pred_table['id'] = i
                long_pred_table['sub_id'] = j
                long_pred_table['time'] = time_list

                long_all = pd.concat([long_all, long_pred_table])
                
                # 장기 모델의 경우 4~19시까지 예측되므로 나머지 시간에 대해서는 0으로 처리
                long_all['1_0'] = 0
                long_all['1_1'] = 0
                long_all['1_2'] = 0
                long_all['1_3'] = 0
                long_all['1_20'] = 0
                long_all['1_21'] = 0
                long_all['1_22'] = 0
                long_all['1_23'] = 0
                long_all['2_0'] = 0
                long_all['2_1'] = 0
                long_all['2_2'] = 0
                long_all['2_3'] = 0
                long_all['2_20'] = 0
                long_all['2_21'] = 0
                long_all['2_22'] = 0
                long_all['2_23'] = 0
                long_all['3_0'] = 0
                long_all['3_1'] = 0
                long_all['3_2'] = 0
                long_all['3_3'] = 0
                long_all['3_20'] = 0
                long_all['3_21'] = 0
                long_all['3_22'] = 0
                long_all['3_23'] = 0
                long_all['4_0'] = 0
                long_all['4_1'] = 0
                long_all['4_2'] = 0
                long_all['4_3'] = 0
                long_all['4_20'] = 0
                long_all['4_21'] = 0
                long_all['4_22'] = 0
                long_all['4_23'] = 0
                long_all['5_0'] = 0
                long_all['5_1'] = 0
                long_all['5_2'] = 0
                long_all['5_3'] = 0
                long_all['5_20'] = 0
                long_all['5_21'] = 0
                long_all['5_22'] = 0
                long_all['5_23'] = 0
                long_all['6_0'] = 0
                long_all['6_1'] = 0
                long_all['6_2'] = 0
                long_all['6_3'] = 0
                long_all['6_20'] = 0
                long_all['6_21'] = 0
                long_all['6_22'] = 0
                long_all['6_23'] = 0
                long_all['7_0'] = 0
                long_all['7_1'] = 0
                long_all['7_2'] = 0
                long_all['7_3'] = 0
                long_all['7_20'] = 0
                long_all['7_21'] = 0
                long_all['7_22'] = 0
                long_all['7_23'] = 0
                long_all['8_0'] = 0
                long_all['8_1'] = 0
                long_all['8_2'] = 0
                long_all['8_3'] = 0
                long_all['8_20'] = 0
                long_all['8_21'] = 0
                long_all['8_22'] = 0
                long_all['8_23'] = 0
                long_all['9_0'] = 0
                long_all['9_1'] = 0
                long_all['9_2'] = 0
                long_all['9_3'] = 0
                long_all['9_20'] = 0
                long_all['9_21'] = 0
                long_all['9_22'] = 0
                long_all['9_23'] = 0
                long_all['10_0'] = 0
                long_all['10_1'] = 0
                long_all['10_2'] = 0
                long_all['10_3'] = 0
                long_all['10_20'] = 0
                long_all['10_21'] = 0
                long_all['10_22'] = 0
                long_all['10_23'] = 0
                
                # 년도, 월, 일을 예측 테이블에 추가
                long_all['time'] = pd.to_datetime(long_all['time'])
                long_all['year'] = long_all['time'].dt.year
                long_all['month'] = long_all['time'].dt.month
                long_all['day'] = long_all['time'].dt.day

                long_all = long_all[['id','sub_id', 'year', 'month', 'day', 'time', '1_0', '1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '1_7', '1_8', '1_9', '1_10', '1_11', '1_12', '1_13', '1_14', '1_15', '1_16', '1_17', '1_18', '1_19', '1_20', '1_21', '1_22', '1_23', '2_0', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '2_10', '2_11', '2_12', '2_13', '2_14', '2_15', '2_16', '2_17', '2_18', '2_19', '2_20', '2_21', '2_22', '2_23', '3_0', '3_1', '3_2', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '3_10', '3_11', '3_12', '3_13', '3_14', '3_15', '3_16', '3_17', '3_18', '3_19', '3_20', '3_21', '3_22', '3_23', '4_0', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8', '4_9', '4_10', '4_11', '4_12', '4_13', '4_14', '4_15', '4_16', '4_17', '4_18', '4_19', '4_20', '4_21', '4_22', '4_23', '5_0', '5_1', '5_2', '5_3', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9', '5_10', '5_11', '5_12', '5_13', '5_14', '5_15', '5_16', '5_17', '5_18', '5_19', '5_20', '5_21', '5_22', '5_23', '6_0', '6_1', '6_2', '6_3', '6_4', '6_5', '6_6', '6_7', '6_8', '6_9', '6_10', '6_11', '6_12', '6_13', '6_14', '6_15', '6_16', '6_17', '6_18', '6_19', '6_20', '6_21', '6_22', '6_23', '7_0', '7_1', '7_2', '7_3', '7_4', '7_5', '7_6', '7_7', '7_8', '7_9', '7_10', '7_11', '7_12', '7_13', '7_14', '7_15', '7_16', '7_17', '7_18', '7_19', '7_20', '7_21', '7_22', '7_23', '8_0', '8_1', '8_2', '8_3', '8_4', '8_5', '8_6', '8_7', '8_8', '8_9', '8_10', '8_11', '8_12', '8_13', '8_14', '8_15', '8_16', '8_17', '8_18', '8_19', '8_20', '8_21', '8_22', '8_23', '9_0', '9_1', '9_2', '9_3', '9_4', '9_5', '9_6', '9_7', '9_8', '9_9', '9_10', '9_11', '9_12', '9_13', '9_14', '9_15', '9_16', '9_17', '9_18', '9_19', '9_20', '9_21', '9_22', '9_23', '10_0', '10_1', '10_2', '10_3', '10_4', '10_5', '10_6', '10_7', '10_8', '10_9', '10_10', '10_11', '10_12', '10_13', '10_14', '10_15', '10_16', '10_17', '10_18', '10_19', '10_20', '10_21', '10_22', '10_23']]
                
    return long_all



# 5. api 호출 시 예측 결과를 DB에 저장
def run():
    #connect to the database
    conn = psycopg2.connect(host='172.28.19.02',dbname='postgres',user='postgres',password='postgres',port=5432)  

    conn.autocommit = True 

    cur = conn.cursor()

    # 5-1 예측 데이틀 변환
    # 단기 예측 테이블 numpy 자료형으로 변환
    short_all = Short_Model()
    row = [tuple(x) for x in short_all.to_numpy()]
    print(row)

    print("short done")

    # 장기 예측 테이블 numpy 자료형으로 변환
    long_all = Long_Model()
    long_row = [tuple(x) for x in long_all.to_numpy()]
    print("long done")

    # 5-2 변환한 데이터를 DB에 추가
    # 단기 예측 결과를 단기 예측 테이블에 추가
    for i in range(len(row)):
        insert_query = "INSERT INTO short_solardb(id, sub_id, year, month, day, timestamp, t_0, t_1, t_2, t_3, t_4, t_5, t_6, \
        t_7, t_8, t_9,t_10,t_11, t_12, t_13, t_14, t_15, t_16, t_17, t_18, t_19, t_20, t_21, t_22, t_23) VALUES ('%s', %d, %d, %d, %d, '%s', %d, \
        %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d);"%(row[i][0], row[i][1], row[i][2], row[i][3], row[i][4], row[i][5], row[i][6], row[i][7], row[i][8], row[i][9], row[i][10], row[i][11], row[i][12], row[i][13], row[i][14], row[i][15], row[i][16], row[i][17], row[i][18], row[i][19], row[i][20], row[i][21], row[i][22], row[i][23], row[i][24], row[i][25],row[i][26],row[i][27],row[i][28],row[i][29])

        try:
            cur.execute(insert_query)
            
            conn.commit()

        except:
            conn.rollback()


    # 장기 예측 결과를 장기 예측 테이블에 추가
    for i in range(len(long_row)):
        long_insert_query = "INSERT INTO long_solardb(id, sub_id, year, month, day, timestamp, t1_0, t1_1, t1_2, t1_3, t1_4, t1_5, t1_6, t1_7, t1_8, t1_9, t1_10, t1_11, t1_12, t1_13, t1_14, t1_15, t1_16, t1_17, t1_18, t1_19, t1_20, t1_21, t1_22, t1_23, t2_0, t2_1, t2_2, t2_3, t2_4, t2_5, t2_6, t2_7, t2_8, t2_9, t2_10, t2_11, t2_12, t2_13, t2_14, t2_15, t2_16, t2_17, t2_18, t2_19, t2_20, t2_21, t2_22, t2_23, t3_0, t3_1, t3_2, t3_3, t3_4, t3_5, t3_6, t3_7, t3_8, t3_9, t3_10, t3_11, t3_12, t3_13, t3_14, t3_15, t3_16, t3_17, t3_18, t3_19, t3_20, t3_21, t3_22, t3_23, t4_0, t4_1, t4_2, t4_3, t4_4, t4_5, t4_6, t4_7, t4_8, t4_9, t4_10, t4_11, t4_12, t4_13, t4_14, t4_15, t4_16, t4_17, t4_18, t4_19, t4_20, t4_21, t4_22, t4_23, t5_0, t5_1, t5_2, t5_3, t5_4, t5_5, t5_6, t5_7, t5_8, t5_9, t5_10, t5_11, t5_12, t5_13, t5_14, t5_15, t5_16, t5_17, t5_18, t5_19, t5_20, t5_21, t5_22, t5_23, t6_0, t6_1, t6_2, t6_3, t6_4, t6_5, t6_6, t6_7, t6_8, t6_9, t6_10, t6_11, t6_12, t6_13, t6_14, t6_15, t6_16, t6_17, t6_18, t6_19, t6_20, t6_21, t6_22, t6_23, t7_0, t7_1, t7_2, t7_3, t7_4, t7_5, t7_6, t7_7, t7_8, t7_9, t7_10, t7_11, t7_12, t7_13, t7_14, t7_15, t7_16, t7_17, t7_18, t7_19, t7_20, t7_21, t7_22, t7_23, t8_0, t8_1, t8_2, t8_3, t8_4, t8_5, t8_6, t8_7, t8_8, t8_9, t8_10, t8_11, t8_12, t8_13, t8_14, t8_15, t8_16, t8_17, t8_18, t8_19, t8_20, t8_21, t8_22, t8_23, t9_0, t9_1, t9_2, t9_3, t9_4, t9_5, t9_6, t9_7, t9_8, t9_9, t9_10, t9_11, t9_12, t9_13, t9_14, t9_15, t9_16, t9_17, t9_18, t9_19, t9_20, t9_21, t9_22, t9_23, t10_0, t10_1, t10_2, t10_3, t10_4, t10_5, t10_6, t10_7, t10_8, t10_9, t10_10, t10_11, t10_12, t10_13, t10_14, t10_15, t10_16, t10_17, t10_18, t10_19, t10_20, t10_21, t10_22, t10_23) VALUES ('%s', %d, %d, %d, %d, '%s', %d, \
        %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d);"%(long_row[i][0], long_row[i][1], long_row[i][2], long_row[i][3], long_row[i][4], long_row[i][5], long_row[i][6], long_row[i][7], long_row[i][8], long_row[i][9], long_row[i][10], long_row[i][11], long_row[i][12], long_row[i][13], long_row[i][14], long_row[i][15], long_row[i][16], long_row[i][17], long_row[i][18], long_row[i][19], long_row[i][20], long_row[i][21], long_row[i][22], long_row[i][23], long_row[i][24], long_row[i][25], long_row[i][26], long_row[i][27], long_row[i][28], long_row[i][29], long_row[i][30], long_row[i][31], long_row[i][32], long_row[i][33], long_row[i][34], long_row[i][35], long_row[i][36], long_row[i][37], long_row[i][38], long_row[i][39], long_row[i][40], long_row[i][41], long_row[i][42], long_row[i][43], long_row[i][44], long_row[i][45], long_row[i][46], long_row[i][47], long_row[i][48], long_row[i][49], long_row[i][50], long_row[i][51], long_row[i][52], long_row[i][53], long_row[i][54], long_row[i][55], long_row[i][56], long_row[i][57], long_row[i][58], long_row[i][59], long_row[i][60], long_row[i][61], long_row[i][62], long_row[i][63], long_row[i][64], long_row[i][65], long_row[i][66], long_row[i][67], long_row[i][68], long_row[i][69], long_row[i][70], long_row[i][71], long_row[i][72], long_row[i][73], long_row[i][74], long_row[i][75], long_row[i][76], long_row[i][77], long_row[i][78], long_row[i][79], long_row[i][80], long_row[i][81], long_row[i][82], long_row[i][83], long_row[i][84], long_row[i][85], long_row[i][86], long_row[i][87], long_row[i][88], long_row[i][89], long_row[i][90], long_row[i][91], long_row[i][92], long_row[i][93], long_row[i][94], long_row[i][95], long_row[i][96], long_row[i][97], long_row[i][98], long_row[i][99], long_row[i][100], long_row[i][101], long_row[i][102], long_row[i][103], long_row[i][104], long_row[i][105], long_row[i][106], long_row[i][107], long_row[i][108], long_row[i][109], long_row[i][110], long_row[i][111], long_row[i][112], long_row[i][113], long_row[i][114], long_row[i][115], long_row[i][116], long_row[i][117], long_row[i][118], long_row[i][119], long_row[i][120], long_row[i][121], long_row[i][122], long_row[i][123], long_row[i][124], long_row[i][125], long_row[i][126], long_row[i][127], long_row[i][128], long_row[i][129], long_row[i][130], long_row[i][131], long_row[i][132], long_row[i][133], long_row[i][134], long_row[i][135], long_row[i][136], long_row[i][137], long_row[i][138], long_row[i][139], long_row[i][140], long_row[i][141], long_row[i][142], long_row[i][143], long_row[i][144], long_row[i][145], long_row[i][146], long_row[i][147], long_row[i][148], long_row[i][149], long_row[i][150], long_row[i][151], long_row[i][152], long_row[i][153], long_row[i][154], long_row[i][155], long_row[i][156], long_row[i][157], long_row[i][158], long_row[i][159], long_row[i][160], long_row[i][161], long_row[i][162], long_row[i][163], long_row[i][164], long_row[i][165], long_row[i][166], long_row[i][167], long_row[i][168], long_row[i][169], long_row[i][170], long_row[i][171], long_row[i][172], long_row[i][173], long_row[i][174], long_row[i][175], long_row[i][176], long_row[i][177], long_row[i][178], long_row[i][179], long_row[i][180], long_row[i][181], long_row[i][182], long_row[i][183], long_row[i][184], long_row[i][185], long_row[i][186], long_row[i][187], long_row[i][188], long_row[i][189], long_row[i][190], long_row[i][191], long_row[i][192], long_row[i][193], long_row[i][194], long_row[i][195], long_row[i][196], long_row[i][197], long_row[i][198], long_row[i][199], long_row[i][200], long_row[i][201], long_row[i][202], long_row[i][203], long_row[i][204], long_row[i][205], long_row[i][206], long_row[i][207], long_row[i][208], long_row[i][209], long_row[i][210], long_row[i][211], long_row[i][212], long_row[i][213], long_row[i][214], long_row[i][215], long_row[i][216], long_row[i][217], long_row[i][218], long_row[i][219], long_row[i][220], long_row[i][221], long_row[i][222], long_row[i][223], long_row[i][224], long_row[i][225], long_row[i][226], long_row[i][227], long_row[i][228], long_row[i][229], long_row[i][230], long_row[i][231], long_row[i][232], long_row[i][233], long_row[i][234], long_row[i][235], long_row[i][236], long_row[i][237], long_row[i][238], long_row[i][239], long_row[i][240], long_row[i][241], long_row[i][242], long_row[i][243], long_row[i][244], long_row[i][245])

        try:
            
            cur.execute(long_insert_query)
            conn.commit()

        except:
            conn.rollback()
            
    conn.close()

    
if __name__=="__main__":
    run()
