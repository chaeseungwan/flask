from flask_restful import Resource, reqparse, request
import json
from db import Database

# 시간대별 요금 설정
class Set_investment(Resource):
    # 1. DB update
    def Update_investment(self,investment,user_id):
        dbc = Database()
        sql = "UPDATE userlist SET investment=%d WHERE id='%s'"%(investment,user_id)
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
            parser.add_argument('investment', type=int)
            
            args = parser.parse_args()

            # argment에서 입력받은 id, investment 추출
            user_id = args['plantId_subId']
            investment = args["investment"]

            # 시간대별 요금 설정시 DB에 update
            if investment >= 0:
                result['Success'] = True
                self.Update_investment(investment, user_id)
                result['data'] = investment

            else: 
                result['isSuccess'] = False

            return result
            
            
        except Exception as e:
            return {"error" : str(e)}
