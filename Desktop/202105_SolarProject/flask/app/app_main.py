# About Flask and Flask_restful Library
from flask import Flask, render_template, request
from flask_restful import Api
from predict import Update_Predict
from login import Login 
from userlist import UserList
from today_revenue import Today_revenue
from set_investment import Set_investment
from set_charge import Set_charge
from search_predict import Search_predict
from revenue_detail import Revenue_detail
from set_database import Set_database

app = Flask(__name__)
api = Api(app)


api.add_resource(Update_Predict, '/update') # 서비스명 : 새로운 데이터가 들어오면 예측 값 DB 저장
api.add_resource(Login, '/login') # 서비스명 : 로그인
api.add_resource(UserList, '/user/list') # 서비스명 : 사용자 조회
api.add_resource(Today_revenue, '/dashboard') # 서비스명 : 당일 수익현황 조회
api.add_resource(Set_investment, '/investment') # 서비스명 : 태양광 설치 투자금액 설정
api.add_resource(Set_charge, '/charge') # 서비스명 : 시간대별 요금 설정
api.add_resource(Search_predict, '/detail') # 서비스명 : 발전량 및 수익예측현황 조회
api.add_resource(Revenue_detail, '/report') # 서비스명 : 수익현황 상세 조회
api.add_resource(Set_database, '/set/database') # 서비스명 : DB 업로드 API


@app.route("/")
def hello():
    return render_template('/test.html')

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
