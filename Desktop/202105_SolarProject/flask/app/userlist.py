from flask_restful import Resource, reqparse, request
import json
from db import Database


# 회원정보 목록 조회
# 1. UserlistDB 조회
class UserList(Resource):
    def DB_load_id_list(self, dbc):
        sql = "select id from userlist"
        result = dbc.executeAll(sql)
        return result

# 2. get 통신
    def get(self):
        try:
            ##요청값을 받아오고 결과값을 주기 위한 로직
            result = {}

            dbc = Database()
            users_load = self.DB_load_id_list(dbc)

            user_list = [] 

            for i in range(len(users_load)):
                user_list.append(users_load[i][0])
            
            result['success'] = True
            result['subUserList'] = user_list
            
            
            return result
         #variable is None:
            
        except Exception as e:
            return {"error" : str(e)}
