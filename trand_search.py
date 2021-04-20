import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import ssl
import json
from datetime import datetime
from datetime import timedelta

def open_api():
    client_id = 'I84U56_OOhxYkILn0k0d'
    client_secret = 'D_FNX4feD7'
    url = 'https://openapi.naver.com/v1/datalab/search'
    return [client_id, client_secret, url]


def date():
    endDate = datetime.today().strftime("%Y-%m-%d")
    startDate = datetime.today() - timedelta(90)
    startDate = startDate.strftime("%Y-%m-%d")
    return [str(startDate), str(endDate)]


def setting(keyword):
    data = date()
    startDate = data[0]
    endDate = data[1]

    body = {"startDate": startDate,
            "endDate": endDate,
            "timeUnit": "date",
            "keywordGroups": [{"groupName": keyword,
                               "keywords": [keyword]}]}

    body = json.dumps(body, )

    id_pw = open_api()
    client_id = id_pw[0]
    client_secret = id_pw[1]
    url = id_pw[2]


    request = urllib.request.Request(url)  # 이건 HTTP Header 변경시에 사용하는 라인, 필요없으면 바로 urllib.request.urlopen()으로 ㄱㄱ
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    request.add_header("Content-Type", "application/json")

    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(request, data=body.encode("utf-8"), context=context)
    rescode = response.getcode()

    if (rescode == 200):
        response_body = response.read()
        scraped = response_body.decode('utf-8')
    else:
        print("Error Code:" + rescode)

    result = json.loads(scraped)

    return result


def parsing(word):
    result = setting(word)
    data = result['results'][0]['data']
    time = [i['period'] for i in data]
    value = [i['ratio'] for i in data]
    data_frame = pd.DataFrame({'Date': time, 'search_count': value})
    return data_frame

