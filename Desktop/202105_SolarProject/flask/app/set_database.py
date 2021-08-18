from flask_restful import Resource, reqparse, request
import json
from db import Database
import psycopg2 #import the Postgres library

# DB 테이블 생성 및 데이터 입력
class Set_database(Resource):

    # 1. 테이블 생성
    def set_relations(self, cur):
        re = []
        # 1.1 원본 데이터 테이블 생성
        try:
            cur.execute("CREATE TABLE row_solardb(timestamp TEXT, id TEXT, sub_id INT, value REAL, capacity REAL, lat REAL, longitude REAL, ta REAL,rn REAL, ws REAL, wd REAL, hm REAL, dc10Tca REAL, PRIMARY KEY (timestamp, id, sub_id));")
        except Exception as e:
            re.append(str(e))
        # 1.2 단기 예측 테이블 생성
        try:
            cur.execute("CREATE TABLE short_solardb(id TEXT, sub_id INT, year INT, month INT, day INT, timestamp TEXT, t_0 REAL, t_1 REAL, t_2 REAL, t_3 REAL, t_4 REAL, t_5 REAL, t_6 REAL, t_7 REAL, t_8 REAL, t_9 REAL, t_10 REAL, t_11 REAL, t_12 REAL, t_13 REAL, t_14 REAL, t_15 REAL, t_16 REAL, t_17 REAL, t_18 REAL, t_19 REAL, t_20 REAL, t_21 REAL, t_22 REAL, t_23 REAL, PRIMARY KEY (id, sub_id, timestamp));")
        except Exception as e:
            re.append(str(e))
        # 1.3 장기 예측 테이블 생성
        try:
            cur.execute("CREATE TABLE long_solardb(id TEXT, sub_id INT, year INT, month INT, day INT, timestamp TEXT, t1_0 REAL, t1_1 REAL, t1_2 REAL, t1_3 REAL, t1_4 REAL, t1_5 REAL, t1_6 REAL, t1_7 REAL, t1_8 REAL, t1_9 REAL, t1_10 REAL, t1_11 REAL, t1_12 REAL, t1_13 REAL, t1_14 REAL, t1_15 REAL, t1_16 REAL, t1_17 REAL, t1_18 REAL, t1_19 REAL, t1_20 REAL, t1_21 REAL, t1_22 REAL, t1_23 REAL,  t2_0 REAL, t2_1 REAL, t2_2 REAL, t2_3 REAL, t2_4 REAL, t2_5 REAL, t2_6 REAL, t2_7 REAL, t2_8 REAL, t2_9 REAL, t2_10 REAL, t2_11 REAL, t2_12 REAL, t2_13 REAL, t2_14 REAL, t2_15 REAL, t2_16 REAL, t2_17 REAL, t2_18 REAL, t2_19 REAL, t2_20 REAL, t2_21 REAL, t2_22 REAL, t2_23 REAL, t3_0 REAL, t3_1 REAL, t3_2 REAL, t3_3 REAL, t3_4 REAL, t3_5 REAL, t3_6 REAL, t3_7 REAL, t3_8 REAL, t3_9 REAL, t3_10 REAL, t3_11 REAL, t3_12 REAL, t3_13 REAL, t3_14 REAL, t3_15 REAL, t3_16 REAL, t3_17 REAL, t3_18 REAL, t3_19 REAL, t3_20 REAL, t3_21 REAL, t3_22 REAL, t3_23 REAL, t4_0 REAL, t4_1 REAL, t4_2 REAL, t4_3 REAL, t4_4 REAL, t4_5 REAL, t4_6 REAL, t4_7 REAL, t4_8 REAL, t4_9 REAL, t4_10 REAL, t4_11 REAL, t4_12 REAL, t4_13 REAL, t4_14 REAL, t4_15 REAL, t4_16 REAL, t4_17 REAL, t4_18 REAL, t4_19 REAL, t4_20 REAL, t4_21 REAL, t4_22 REAL, t4_23 REAL, t5_0 REAL, t5_1 REAL, t5_2 REAL, t5_3 REAL, t5_4 REAL, t5_5 REAL, t5_6 REAL, t5_7 REAL, t5_8 REAL, t5_9 REAL, t5_10 REAL, t5_11 REAL, t5_12 REAL, t5_13 REAL, t5_14 REAL, t5_15 REAL, t5_16 REAL, t5_17 REAL, t5_18 REAL, t5_19 REAL, t5_20 REAL, t5_21 REAL, t5_22 REAL, t5_23 REAL, t6_0 REAL, t6_1 REAL, t6_2 REAL, t6_3 REAL, t6_4 REAL, t6_5 REAL, t6_6 REAL, t6_7 REAL, t6_8 REAL, t6_9 REAL, t6_10 REAL, t6_11 REAL, t6_12 REAL, t6_13 REAL, t6_14 REAL, t6_15 REAL, t6_16 REAL, t6_17 REAL, t6_18 REAL, t6_19 REAL, t6_20 REAL, t6_21 REAL, t6_22 REAL, t6_23 REAL, t7_0 REAL, t7_1 REAL, t7_2 REAL, t7_3 REAL, t7_4 REAL, t7_5 REAL, t7_6 REAL, t7_7 REAL, t7_8 REAL, t7_9 REAL, t7_10 REAL, t7_11 REAL, t7_12 REAL, t7_13 REAL, t7_14 REAL, t7_15 REAL, t7_16 REAL, t7_17 REAL, t7_18 REAL, t7_19 REAL, t7_20 REAL, t7_21 REAL, t7_22 REAL, t7_23 REAL, t8_0 REAL, t8_1 REAL, t8_2 REAL, t8_3 REAL, t8_4 REAL, t8_5 REAL, t8_6 REAL, t8_7 REAL, t8_8 REAL, t8_9 REAL, t8_10 REAL, t8_11 REAL, t8_12 REAL, t8_13 REAL, t8_14 REAL, t8_15 REAL, t8_16 REAL, t8_17 REAL, t8_18 REAL, t8_19 REAL, t8_20 REAL, t8_21 REAL, t8_22 REAL, t8_23 REAL, t9_0 REAL, t9_1 REAL, t9_2 REAL, t9_3 REAL, t9_4 REAL, t9_5 REAL, t9_6 REAL, t9_7 REAL, t9_8 REAL, t9_9 REAL, t9_10 REAL, t9_11 REAL, t9_12 REAL, t9_13 REAL, t9_14 REAL, t9_15 REAL, t9_16 REAL, t9_17 REAL, t9_18 REAL, t9_19 REAL, t9_20 REAL, t9_21 REAL, t9_22 REAL, t9_23 REAL, t10_0 REAL, t10_1 REAL, t10_2 REAL, t10_3 REAL, t10_4 REAL, t10_5 REAL, t10_6 REAL, t10_7 REAL, t10_8 REAL, t10_9 REAL, t10_10 REAL, t10_11 REAL, t10_12 REAL, t10_13 REAL, t10_14 REAL, t10_15 REAL, t10_16 REAL, t10_17 REAL, t10_18 REAL, t10_19 REAL, t10_20 REAL, t10_21 REAL, t10_22 REAL, t10_23 REAL, PRIMARY KEY (id, sub_id, timestamp));")
        except Exception as e:
            re.append(str(e))
        # 1.4 시간대별 요금 테이블 생성
        try:
            cur.execute("CREATE TABLE charge(plantId_subId TEXT PRIMARY KEY, charge_0 REAL, charge_1 REAL, charge_2 REAL, charge_3 REAL, charge_4 REAL, \
            charge_5 REAL, charge_6 REAL, charge_7 REAL, charge_8 REAL, charge_9 REAL, charge_10 REAL, charge_11 REAL, charge_12 REAL, charge_13 REAL, charge_14 REAL, charge_15 REAL, charge_16 REAL, \
                charge_17 REAL, charge_18 REAL, charge_19 REAL, charge_20 REAL, charge_21 REAL, charge_22 REAL, charge_23 REAL);")
        except Exception as e:
            re.append(str(e))
        # 1.5 회원정보 테이블 생성
        try:
            cur.execute("CREATE TABLE userlist(id TEXT PRIMARY KEY, password TEXT, investment REAL);")
        except Exception as e:
            re.append(str(e))
        return re

    # 2. DB에 데이터 넣기
    # 2-1 원본데이터 DB에 추가
    def set_row_solardb(self, conn, cur):
        with open('./data/train_data.csv','r', encoding='cp949') as f:
            next(f)
            cur.copy_from(f, 'row_solardb', sep=',', null='')

            conn.commit()
            conn.close()
        f.close()

    # 2-2 단기예측 데이터 DB에 추가
    def set_short_solardb(self, conn, cur):

        #create table with same headers as csv file

        with open('./data/Short_model_result.csv','r', encoding='cp949') as f:
            next(f)
            cur.copy_from(f, 'short_solardb', sep=',', null='')

            conn.commit()
        f.close()

    # 2-3 장기 예측 데이터 DB에 추가
    def set_long_solardb(self, conn, cur):

        #create table with same headers as csv file


        with open('./data/long_model_resutl.csv','r', encoding='cp949') as f:
            next(f)
            cur.copy_from(f, 'long_solardb', sep=',', null='')

            conn.commit()
        f.close()

    # 2-4 시간대별 요금 데이터 DB에 추가
    def set_charge(self, conn, cur):

        with open('./data/charge.csv','r', encoding='cp949') as f:
            next(f)
            cur.copy_from(f, 'charge', sep=',', null='')

            conn.commit()
        f.close()

    # 2-5 회원정보 데이터 DB에 추가    
    def set_userlist(self, conn, cur):


        with open('./data/userlist.csv','r', encoding='cp949') as f:
            next(f)
            cur.copy_from(f, 'userlist', sep=',', null='')

            conn.commit()
        f.close()

    # 3. get 통신
    def get(self):
        try:
            result={}
            #connect to the database
            conn = psycopg2.connect(host='172.28.19.02',
                                dbname='postgres',
                                user='postgres',
                                password='postgres',
                                port=5432)  
            #create a cursor object 
            #cursor object is used to interact with the database
            conn.autocommit = True 
            cur = conn.cursor()
            result["set_relations"] = self.set_relations(cur)
  
            # 원본 데이터
            try:
                cur = conn.cursor()

                self.set_row_solardb(conn, cur)
                result["set_row_solardb"] = "success"
            except Exception as e:
                result["set_row_solardb"] = str(e)
            
            # 단기 예측 데이터
            try:
                cur = conn.cursor()

                self.set_short_solardb(conn, cur)
                result["set_short_solardb"] = "success"
            except Exception as e:
                result["set_short_solardb"] = str(e)
            
            # 장기 예측 데이터
            try:
                cur = conn.cursor()

                self.set_long_solardb(conn, cur)
                result["set_long_solardb"] = "success"
            except Exception as e:
                result["set_long_solardb"] = str(e)
            
            # 시간대별 요금
            try:
                cur = conn.cursor()

                self.set_charge(conn, cur)
                result["set_charge"] = "success"
            except Exception as e:
                result["set_charge"] = str(e)
            
            # 회원정보
            try:
                cur = conn.cursor()

                self.set_userlist(conn, cur)
                result["set_userlist"] = "success"
            except Exception as e:
                result["set_userlist"] = str(e)
              
            return {"result" : result}
            
            
        except Exception as e:
            return {"error" : str(e)}
