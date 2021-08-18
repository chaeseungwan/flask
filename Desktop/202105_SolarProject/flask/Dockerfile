FROM python:3.8
#DockerHub에서 python:3.8 이미지 가져옴

WORKDIR /app
#docker conatainer 내의 작업 경로 설정
ADD ./flask/app /app
#COPY ./flask/app /app
#현재 디렉터리 /flask/app을 python:3.8 도커 컨테이너 내의 /app으로 복사



# apt-get update
RUN apt-get update

# cron install
RUN apt-get install -y cron

RUN touch /usr/sbin/entrypoint.sh
RUN chmod 777 /usr/sbin/entrypoint.sh


RUN pip3 install -r /app/requirements.txt


RUN apt-get install -y --no-install-recommends apt-utils

ENTRYPOINT ["/usr/sbin/entrypoint.sh"]


# entry point
###############################################
#RUN touch /usr/sbin/entrypoint.sh
#RUN chmod 777 /usr/sbin/entrypoint.sh

#ENTRYPOINT ["/usr/sbin/entrypoint.sh"]

#/app/requirements.txt에 정의된 패키지 설치
#CMD ["uwsgi","uwsgi.ini"]
#CMD ["python3","app_main.py"]



