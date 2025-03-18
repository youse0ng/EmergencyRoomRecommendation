from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import openai
import torch
import json
import pandas as pd
import requests
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
path = './'

# 0. load key file------------------
def load_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return file.readline().strip()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="API key file not found.")



# 1-2 text2summary------------------
def text_summary(input_text):
    # OpenAI 클라이언트 생성
    client = OpenAI()

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
    answer = response.choices[0].message.content
    parsed_answer = json.loads(answer)

    summary = parsed_answer["summary"]

    return summary
 
 # 2. model prediction------------------
def predict(text, model, tokenizer):
    # 입력 문장 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value for key, value in inputs.items()}  # 각 텐서를 GPU로 이동

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)

    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities, dim=-1).item()

    return pred, probabilities

# 3-1. get_distance------------------
def get_dist(start_lat, start_lng, dest_lat, dest_lng, c_id, c_key):
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
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
            #print(f"에러 : {response_data['code']}, {response_data['message']}")
            return None
    else:
        print(f"에러 : {response.status_code}, {response.text}")
        return None


# 3-2. recommendation------------------


def recommend_hospital3(start_lat, start_lng, emergency, c_id, c_key,hospital_num):
    """
    응급 환자의 현재 위치를 기반으로 가장 가까운 병원 3곳을 추천하는 함수.

    Args:
        - start_lat (float): 출발지 위도
        - start_lng (float): 출발지 경도
        - emergency (DataFrame): 병원 정보 데이터프레임
        - c_id (str): 네이버 API 클라이언트 ID
        - c_key (str): 네이버 API 클라이언트 키

    Returns:
        - DataFrame: 추천 병원 3곳과 거리 및 소요 시간 정보
    """

    # 초기 검색 반경 설정
    alpha = 0.05
    df_temp = pd.DataFrame()

    # 3곳 이상의 병원이 나올 때까지 반경을 증가시키며 병원 검색
    while len(df_temp) < 3:
        lat_min, lat_max = start_lat - alpha, start_lat + alpha
        lon_min, lon_max = start_lng - alpha, start_lng + alpha

        # 반경 내 병원 필터링
        df_temp = emergency.loc[
            emergency['위도'].between(lat_min, lat_max) &
            emergency['경도'].between(lon_min, lon_max)
        ].copy()

        # 반경을 점진적으로 확대
        alpha += 0.05

    # 도로 거리 및 소요 시간 계산
    distances = []
    times = []
    for _, row in df_temp.iterrows():
        hospital_lat = row['위도']
        hospital_lon = row['경도']

        # get_dist 함수로 거리 및 소요 시간 계산
        result = get_dist(start_lat, start_lng, hospital_lat, hospital_lon, c_id, c_key)

        if result:
            distance, time = result
        else:
            distance, time = float('inf'), float('inf')  # 오류 발생 시 무한대 값 처리

        distances.append(distance)
        times.append(time)

    # 계산된 거리 및 소요 시간 추가
    df_temp['거리(km)'] = distances
    df_temp['소요시간(분)'] = times

    # 거리 기준으로 정렬
    df_temp = df_temp.sort_values(by='거리(km)').reset_index(drop=True)

    # 가장 가까운 병원 동적으로로 반환
    return [
        {
            "병원이름": row["병원이름"],
            "주소": row["주소"],
            "응급의료기관 종류": row["응급의료기관 종류"],
            "전화번호 1": row["전화번호 1"],
            "전화번호 2": row["전화번호 3"],
            "위도": row["위도"],
            "경도": row["경도"],
            "거리(km)": row["거리(km)"],
            "소요시간(분)": row["소요시간(분)"]
        }
        for _, row in df_temp.head(hospital_num).iterrows()
    ]


#5. FastAPI 엔드포인트
@app.get('/')
def read_root():
    return {"message":"Welcome to the Hospital Recommendation API!"}

@app.get("/recommend-hospitals")
def get_hospitals(request:str,latitude:float,longitude:float,hospital_num:int):
    # OpenAI API 키 설정
    openai.api_key = load_file(path + 'api_key.txt')
    api_key = load_file(path+'api_key.txt')
    os.environ['OPENAI_API_KEY'] = openai.api_key
    
    #지도 API 키
    map_key = load_file(path + 'map_key.txt')
    map_key = json.loads(map_key)
    c_id, c_key = map_key['c_id'], map_key['c_key']
    
    #CSV 불러오기기
    emergency = pd.read_csv(path + '응급실 정보.csv')

    # 모델, 토크나이저 로드
    save_directory = path + "fine_tuned_bert2"
    model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
            
    #1. 요약
    summary=text_summary(request)
    #2. 요약된것,코드 분류
    predicted_class, _ = predict(summary, model, tokenizer)
    if predicted_class<=2:
        message="응급상황입니다. 빠른 시간안에 응급실로 이동하세요"
    else:
        message="응급상황이 아닌것으로 판단됩니다. 증상이 악화되면 응급실로 이동하세요"
    
    # 병원 추천
    result = recommend_hospital3(latitude, longitude, emergency, c_id, c_key,hospital_num)
    
    return {"요약된 응급상황": summary,"현재 응급등급":predicted_class+1,"응급상황 메시지":message,"주변 응급실": result}


