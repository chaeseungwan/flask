from flask_restful import Resource, reqparse, request
from flask import Flask, jsonify
import json
from db import Database
import bcrypt


class Login(Resource):
    # 1. 로그인 설계
    # 1-1 회원정보 DB 조회
    def userlist_search(self, id_subid, dbc):
        sql = "select * from userlist where id='%s'"%(id_subid)
        result = dbc.executeOne(sql)
        return result

    # 2. get 통신
    def post(self):
        try:
            ##요청값을 받아오고 결과값을 주기 위한 로직
            result = {}
            #request를 파싱 하는 부분
            parser = reqparse.RequestParser()
            #type을 key로 한 값에 대해서 string 명시
            parser.add_argument('id', type=str)
            parser.add_argument('passwd', type=str)
            
            #data를 key로 한 값에 대해서 list형태 명시
            #파싱하여 args에 할당
            args = parser.parse_args()

            # 2-1 입력받은 argment의 id, passwd 추출
            input_id = args['id']
            input_passwd = args["passwd"]
            
            dbc = Database()
            
            
            data = self.userlist_search(input_id, dbc)
            

            # 2-2 : userlist DB에서 userID와 password 추출
            db_id = data[0] #ID
            db_passwd = data[1]  #password
        
            # 2-3 입력받은 id,pw와 DB정보와 매칭 후 일치여부 전달
            if (input_id == db_id) & (bcrypt.checkpw(input_passwd.encode('utf-8'), db_passwd.encode('utf-8'))):
                result['success'] = True
            else:
                result['success'] = False

            return result
        
        # 에러 발생시 예외처리
        except Exception as e:
            return {'error' : str(e)}    
        
          
            