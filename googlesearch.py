from pytrends.request import TrendReq
import datetime


def google_parsing(word):
    keyword = word
    period = 'today 3-m'

    trend_obj = TrendReq()
    trend_obj.build_payload(kw_list=[keyword], timeframe=period)

    trend_df = trend_obj.interest_over_time()
    trend_df = trend_df.iloc[:, :1]
    trend_df = trend_df.reset_index()
    str_date = []

    for i in range(len(trend_df)):
        timestamp = trend_df['date'][i]
        ymd = datetime.datetime.strptime(str(timestamp), '%Y-%m-%d %H:%M:%S').date()
        str_ymd = ymd.strftime('%Y-%m-%d')
        str_date.append(str_ymd)

    trend_df['date'] = str_date

    return trend_df

