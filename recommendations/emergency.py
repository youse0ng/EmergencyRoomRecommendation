import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any, Text
from datetime import datetime
import os

# 0. load key file------------------\
def LoadOpenAIClient(filepath):
    with open(filepath + 'api_key.txt', 'r') as file:
        openai.api_key = file.readline().strip()
    os.environ['OPENAI_API_KEY'] = openai.api_key
    c = OpenAI()
    return c

# 1-1 audio2text--------------------
# OpenAI API 불러온 후, 음성 인식한 텍스트를 요약해서 리턴하는 함수
def AudioToText(audio_path, filename, Client):
    c = Client
    audio_file = open(audio_path + filename, "rb")
    transcript = c.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        language="ko",
        response_format="text",
        temperature=0.0
    )
    return transcript

# 1-2 text2summary------------------
def TextToSummary(input_text, c):
   # OpenAI 클라이언트 생성
  client = c

  # 시스템 역할과 응답 형식 지정
  system_role = '''당신은 응급전화로부터 핵심을 요약하고 응급상황인지 아닌지 판단하는 어시스턴트입니다.
  요약은 최대 20자 이내로 작성하세요. 그리고 증상을 중심으로 요약하세요.
  응답은 다음의 형식을 지켜주세요
  출력예시: {화상 사고로 인해 피부가 심하게 손상되고 통증으로 의식잃은 상태}
  {"summary": \"텍스트 요약\"}
  '''

  # 입력데이터를 GPT-3.5-turbo에 전달하고 답변 받아오기
  response = client.chat.completions.create(model="gpt-3.5-turbo",
                                            messages=[{"role": "system",
                                                       "content": system_role},
                                                      {"role": "user",
                                                       "content": input_text} ])

  # 응답 받기
  answer = response.choices[0].message.content


  # 응답형식을 정리하고 return
  answer = json.loads(answer)
  return answer['summary']

  df['summary'] = df[text].apply(answer)


# 2. model prediction------------------
# 데이터 예측 함수
def predict(text ,path ,device='cuda')->int:
    '''
    이 함수는 응급 전화로부터 요약된 문장을 입력받고 이미 학습된 모델이 1,2,3,4,5 등급으로 응급도를 예측하는 모델입니다.
    
    Args:
        - text(Text:str): 요약된 응급 문장
        - model(Any): HuggingFace의 모델
        - tokenizer(Any): HuggingFace의 모델을 따라가는 Tokenizer
        - save_directory(str): 이미 학습된 가중치가 들어있는 pt 파일 경로
        - device (default='cuda') = 'cuda' or 'cpu'
    return:
        - pred(int): 응급 등급 확인 
    '''

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # 입력 문장 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 입력 텐서를 GPU로 이동
    model = AutoModelForSequenceClassification.from_pretrained(path)

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)
    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)
    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities, dim=-1).item()
    return pred


# 3-1. get_distance------------------
def GetDistance(start_lat, start_lng, dest_lat, dest_lng):
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    c_id = "5rpillllni"
    c_key = "j7dsmQvmp2W5Q9OQ9oB9JUOF6CTLw6mSzjEtdDOH"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": c_id,
        "X-NCP-APIGW-API-KEY": c_key,
    }
    params = {
        "start": f"{start_lng},{start_lat}",  # 출발지 (경도, 위도)
        "goal": f"{dest_lng},{dest_lat}",     # 목적지 (경도, 위도)
        "option": "trafast"                   # 실시간 빠른 길 옵션
    }

    # 요청하고, 답변 받아오기
    response = requests.get(url, params=params, headers=headers)

    # 성공적으로 받아왔다면
    if (response.status_code == 200):
        response_data = response.json()  # JSON 데이터를 파싱
        if response_data["code"] == 0:
            # 거리
            dist = response_data['route']['trafast'][0]['summary']['distance'] / 1000   # km
            # 소요 시간
            time = response_data['route']['trafast'][0]['summary']['duration'] / 60000  # minute
            return round(dist,2), round(time,2)
        else:
            print(f"에러 : {response_data['code']}, {response_data['message']}")
            return None
    else:
        print(f"에러 : {response.status_code}, {response.text}")
        return None


# 3-2. recommendation------------------
def RecommandHospital(my_location, df):
    my_lat, my_lon = my_location
    alpha = 0
    df_temp = pd.DataFrame()

    while len(df_temp) < 3:
        alpha += 0.1
        lat_min, lat_max = my_lat - alpha, my_lat + alpha
        lon_min, lon_max = my_lon - alpha, my_lon + alpha
        df_temp = df[(df['위도'] >= lat_min) & (df['위도'] <= lat_max) &
                     (df['경도'] >= lon_min) & (df['경도'] <= lon_max)].copy()

    distances = []
    times = []
    for index in range(len(df_temp)):
        hospital_lat = df_temp.iloc[index]['위도']
        hospital_lon = df_temp.iloc[index]['경도']

        # `GetDistance`로 도로 거리 및 소요 시간 계산
        distance, time = GetDistance(my_lat, my_lon, hospital_lat, hospital_lon)
        distances.append(distance)
        times.append(time)

    # 계산한 도로 거리 및 소요 시간 추가
    df_temp['거리(km)'] = distances
    df_temp['소요시간(분)'] = times

    # 거리 기준으로 정렬
    df_temp = df_temp.sort_values(by='거리(km)', ascending=True).reset_index(drop=True)

    # 가장 가까운 병원 3곳 반환
    return df_temp[:3]

from datetime import datetime
import os
dt = datetime.now()
dt = dt.strftime("%Y/%m/%d, %H:%M:%S")

path = r'C:/Users/hyssk/Desktop/EmergencyRecommendation/emergency_service/recommendations/' # 개인에 맞게 설정정
df_emergency = pd.read_csv(path + '응급실 정보.csv')

def pipeline(path, audio, location):
  OpenAIClient = LoadOpenAIClient(path)
  audio_path = path + 'audio/'
  audioName = audio + '.mp3'
  a = AudioToText(audio_path, audioName, OpenAIClient)
  sumText = TextToSummary(a, OpenAIClient)
  model_path = path + 'fine_tuned_bert'
  result = predict(sumText,  model_path, 'cpu')
  if(result < 3):
    print('응급상황인것으로 판단됩니다. 응급실로 이동하세요')
  else:
    print('응급상황이 아닌 것으로 판단됩니다. 증상이 악화된다면 응급실로 이동하세요')
  return RecommandHospital(location, df_emergency),result, sumText

def pipeline2(latitude,longitude,text):
    OpenAIClient = LoadOpenAIClient(path)
    sumText = TextToSummary(text, OpenAIClient)
    model_path = path + 'fine_tuned_bert'
    result = predict(sumText,  model_path, 'cpu')
    if(result < 3):
        print('응급상황인것으로 판단됩니다. 응급실로 이동하세요')
    else:
        print('응급상황이 아닌 것으로 판단됩니다. 증상이 악화된다면 응급실로 이동하세요')
    return RecommandHospital((latitude,longitude), df_emergency),result, sumText

if __name__ == '__main__':
    my_location = (36.339073,127.966817)
    print(pipeline2(36.339073,127.966817,'살려주세요요'))
    