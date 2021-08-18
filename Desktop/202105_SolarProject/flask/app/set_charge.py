from flask_restful import Resource, reqparse, request
import json
from db import Database


class Set_charge(Resource):
    # 1. DB 
    # 1.1 DB update(시간대별 요금)
    def Update_charge(self, charge, user_id):
        dbc = Database()
        for i in range(len(charge)):
            sql = "UPDATE charge SET charge_%d='%d' WHERE plantId_subId='%s'"%(i, charge[i], user_id)
            dbc.execute(sql)

    # 2. get 통신
    def put(self):
        try:
            ##요청값을 받아오고 결과값을 주기 위한 로직
            #request를 파싱 하는 부분
            result = {}
            #type을 key로 한 값에 대해서 string 명시
            parser = reqparse.RequestParser()
            
            parser.add_argument('plantId_subId', type=str)
            parser.add_argument('charge', type=int , action='append')
            
            args = parser.parse_args()
            # 2-1 argment에서 입력받은 plantId_subId, charege
            user_id = args['plantId_subId']
            charge = args["charge"]
            
            
            # DB update
            self.Update_charge(charge, user_id)

            # 요금을 입력시 True, 아닐 시 False return    
            if len(charge) >= 0:
                result['Success'] = True
                result['data'] = charge
            else:
                result['isSuccess'] = False
                
            return result
            # 에러 발생시 예외처리
        except Exception as e:
            return {"error" : str(e)}
