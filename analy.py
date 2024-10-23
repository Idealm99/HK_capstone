from pymongo import MongoClient
import gridfs
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
from collections import Counter, defaultdict
import collections as c
import cv2
import mediapipe as mp
import torch
from torchvision import transforms
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from math import tanh


# MongoDB 연결
client = MongoClient('mongodb+srv://sparta:test@cluster0.50ukg43.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['dbsparta']
fs = gridfs.GridFS(db)
preferences_collection = db['preferences']  # 여기서 preferences_collection 정의

def get_user_preferences():
    """가장 최근의 사용자 선호도 데이터를 MongoDB에서 가져옴."""
    latest_preferences = list(preferences_collection.find().sort('uploadDate', -1).limit(1))  # 최신순으로 정렬

    if len(latest_preferences) > 0:
        print("Preferences loaded from DB:", latest_preferences[0])  # 최신 데이터를 출력
        return latest_preferences[0]  # 최신 데이터 반환
    else:
        print("No preferences found in DB.")  # 선호도 데이터가 없을 경우
        return None  # 데이터가 없을 경우 None 반환



    
def get_latest_image():
    analysis_collection = db['analysis_results']
    analyzed_file_ids = analysis_collection.distinct('file_id')
    image = fs.find({'_id': {'$nin': analyzed_file_ids}}).sort([('uploadDate', -1)]).limit(1)
    
    if image.count() > 0:
        grid_out = image[0]
        image_data = grid_out.read()
        image = Image.open(BytesIO(image_data))
        print(f"Image loaded from DB with file ID: {grid_out._id}")  # 이미지 데이터 출력
        return image, grid_out._id
    else:
        print("No new images found in DB.")  # 이미지 데이터가 없을 경우
        return None, None



# 얼굴 형태 및 턱 분석

def chin_shape(image_path):
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)

    class ChinShapeModel(nn.Module):
        def __init__(self):
            super(ChinShapeModel, self).__init__()
            self.model = EfficientNet.from_name('efficientnet-b4')
            self.model._fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.BatchNorm1d(self.model._fc.in_features),
                nn.Linear(self.model._fc.in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 3)
            )

        def forward(self, x):
            return self.model(x)

    # 가중치 파일 경로
    model_path = 'weights/best_model(92).pth'
    model = ChinShapeModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()

    # 이미지 파일 열기 (추가된 부분 - RGB 변환)
    image = Image.open(image_path).convert('RGB')

    # 이미지 전처리
    input_image = preprocess_image(image)
    with torch.no_grad():
        output = model(input_image)
    class_labels = ['세모형', '둥근형', '각진형']
    predicted_class_index = torch.argmax(output, dim=1)
    return class_labels[predicted_class_index.item()]


def process_image(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    # 결과 저장 사전  # 이마비율, 코비율, 턱 비율,
    result={}
    # 이미지 불러오기
    scale_factor=1
    image = cv2.imread(image_path)

    # 이미지 크기 확대
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # FaceMesh 객체 생성
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        # 이미지를 RGB로 변환하고 FaceMesh 모델에 입력
        results = face_mesh.process(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

        # 얼굴 랜드마크가 감지되었는지 확인
        if results.multi_face_landmarks:
            # 랜드마크 점과 번호 그리기
            annotated_image = resized_image.copy()
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), circle_radius=1)
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * annotated_image.shape[1])
                    y = int(landmark.y * annotated_image.shape[0])
                    cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                landmark_list = face_landmarks.landmark

                # 10번과 9번, 9번과 2번, 2번과 152번 사이의 거리 계산
                landmarks_10 = landmark_list[10]
                landmarks_9 = landmark_list[9]
                landmarks_2 = landmark_list[2]
                landmarks_152 = landmark_list[152]
                distance_10_9 = math.sqrt((landmarks_10.x - landmarks_9.x)**2 + (landmarks_10.y - landmarks_9.y)**2) * 1.6
                distance_9_2 = math.sqrt((landmarks_9.x - landmarks_2.x)**2 + (landmarks_9.y - landmarks_2.y)**2)
                distance_2_152 = math.sqrt((landmarks_2.x - landmarks_152.x)**2 + (landmarks_2.y - landmarks_152.y)**2)


                total_distance = distance_10_9 + distance_9_2 + distance_2_152
                ratio_1 = distance_10_9 / total_distance
                ratio_2 = distance_9_2 / total_distance
                ratio_3 = distance_2_152 / total_distance

                result['forehead']=round(ratio_1, 3)
                result['nose']= round(ratio_2, 3)
                result['chin']= round(ratio_3, 3)

                # print("첫 번째 선의 비율:", round(ratio_1, 3))
                # print("두 번째 선의 비율:", round(ratio_2, 3))
                # print("세 번째 선의 비율:", round(ratio_3, 3))

                # 127, 130, 112, 463, 263, 356 랜드마크 점들 사이의 거리 계산
                distances = []
                pairs = [(127, 130), (130, 155), (155, 463), (463, 263), (263, 356)]
                for pair in pairs:
                    p1 = landmark_list[pair[0]]
                    p2 = landmark_list[pair[1]]
                    distance = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    distances.append(distance)

                # 거리 비율 계산
                total_distance = sum(distances)
                ratios = [d / total_distance for d in distances]
                result['eye_mid']=round(ratios[2],3)

                #광대는 일단 보류
                #result['cheek']=[round(ratios[0],3),round(ratios[-1],3)]
                # print("거리 비율:", [round(r, 3) for r in ratios])

                # 랜드마크 점들에서 수직선 그리기
                for pair in pairs:
                    x = int(landmark_list[pair[0]].x * annotated_image.shape[1])
                    y1 = 0
                    y2 = annotated_image.shape[0]
                    cv2.line(annotated_image, (x, y1), (x, y2), (255, 255, 255), 2)
                             # 세로 10-152, 가로 127-264의 거리 계산




                # 세로 10-152, 가로 127-264의 거리 계산
                landmarks_264 = landmark_list[264]
                landmarks_127 = landmark_list[127]
                distance_10_152 = math.sqrt((landmarks_10.x - landmarks_152.x)**2 + (landmarks_10.y - landmarks_152.y)**2)
                distance_127_264 = math.sqrt((landmarks_127.x - landmarks_264.x)**2 + (landmarks_127.y - landmarks_264.y)**2)

                # 세로 10-152, 가로 127-264의 비율 계산
                total_distance = distance_10_152 + distance_127_264
                ratio_vertical = distance_10_152 / total_distance
                #ratio_horizontal = distance_127_264 / total_distance

                result['vertical']=round(ratio_vertical, 3)
                #result['horizontal']=round(ratio_horizontal, 3)
                return result


                # print("세로의 비율:", round(ratio_vertical, 3))
                # print("가로의 비율:", round(ratio_horizontal, 3))


        else:
            print("No face detected in the image.")

def labeling(result):

  # 데이터 프레임 생성

  standard = {
    'index': ['forehead', 'nose', 'chin', 'eye_mid', 'vertical'],
    'Q1': [0.276, 0.355, 0.333, 0.233, 0.476],
    'mean': [0.2862963148032478, 0.3648178638351054, 0.3488998126171169, 0.24161561524047606, 0.5098495940037496],
    'Q3': [0.299, 0.375, 0.363, 0.251, 0.542]
              }

  face_standard = pd.DataFrame(standard)
  face_standard.rename(columns={'index': 'feature'}, inplace=True)
  face_standard.set_index('feature', inplace=True)

  features_dict= {'forehead' : '이마','nose': '코', 'chin': '턱', 'eye_mid': '미간', 'vertical': '얼굴'}
  # 라벨링
  labels = []


  for i in result.keys():
    if result[i] < face_standard.loc[i, 'Q1']:
      labels.append('짧은 '+features_dict[str(i)])
    elif result[i] > face_standard.loc[i, 'Q3']:
      labels.append('긴 '+features_dict[str(i)])


  return labels


def face_analyze(image_path):
  result=process_image(image_path)
  li=[]
  max=0
  for i in range(10):
    li.append(chin_shape(image_path))



  print(c.Counter(li))

  return [c.Counter(li).most_common(1)[0][0],c.Counter(li).most_common(1)[0][0],c.Counter(li).most_common(2)[1][0]]+labeling(result)

features_df = {

    "짧은 미간": ["올린 머리", "가르마 머리"],
    "긴 미간": ["내린 머리", "위볼륨","앞머리볼륨"],
    "긴 코": ["내린 머리", "옆볼륨", "컬","short" ,"mid"],
    "긴 이마": ["내린 머리", "mid","비대칭"],
    "광대가 돌출된 얼굴": ["mid","short", "long","컬"],
    "긴 턱": [ "mid","long", "옆볼륨", "컬"],
    "짧은 턱": ["long", "mid", "위볼륨"],
    "짧은 이마": ["올린 머리", "위볼륨","다운펌"],
    "짧은 코": ["올린 머리", "위볼륨","다운펌"],
    "긴 얼굴": ["가르마 머리", "내린 머리","옆볼륨","mid", "long","웨이브","컬","비대칭","뒷머리"],
    "짧은 얼굴" : ['short','내린 머리'] ,# 세로 대비 가로가 넓은 얼굴 형
    "각진형":["위볼륨","올린 머리","가르마 머리","다운펌","short","mid"], # 옆볼륨x longx
    "세모형":["앞머리볼륨","옆볼륨","mid","long","컬","뒷머리"], # 짧은건 마이너
    "둥근형":["mid","다운펌","위볼륨","올린 머리","가르마 머리"]

    }


    # 턱 형태에 따른 라벨 추가하기

''' 
남자 헤어 기장감의 기준
short 눈썹 위
mid 눈썰 ~ 코
long  코 ~
'''

# 다운폄 = 옆에 볼륨을 없앤 헤어스타일
male_hairstyles = {

    "댄디컬x": [
        "mid",
        "내린 머리",
        "앞머리볼륨",
        "다운펌",
    ],
    "댄디컬o": [
        "mid",
        "내린 머리",
        "앞머리볼륨",
        "컬"
    ],
    # "시스루댄디": [
    #     "위볼륨",
    #     "mid",
    #     "내린머리",
    #     "앞머리볼륨"
    # ],

    "슬릭댄디": [
        "mid",
        "내린 머리",
        "다운펌",
    ],
    "울프": [
        "내린 머리",
         "long",
        "앞머리볼륨",
        "뒷머리",
        "컬"
    ],
    "리젠트": [
        "short",
        "올린 머리",
        "다운펌"
    ],
    "가일": [
        "mid",
        "가르마 머리",
        "앞머리볼륨",
        "비대칭",
    ],
    "드롭": [
        "short",
        "내린 머리",
        "앞머리볼륨",
        "다운펌"
    ],
    "포마드": [
        "short","mid",
        "올린 머리",
        "다운펌"
    ],
    "롱리프": [
        "long",
        "가르마 머리",
        "앞머리볼륨",
        "옆볼륨"
        ,"컬",
        "뒷머리"

    ],
    "세미리프": [
        "mid ","long",
        "가르마 머리",
        "앞머리볼륨",
        "옆볼륨",
        "컬"
    ],
    "쉐도우펌": [
        "mid",
        "내린 머리",
        "앞머리볼륨",
        "옆볼륨",
        "컬"
    ],
    "아이비리그": [
        "short ",
        "올린 머리",
        "앞머리볼륨",
        "다운펌"
    ],
    "5대5가르마펌o": [
        "mid",
        "가르마 머리",
        "앞머리볼륨",
        "컬"
    ],
    "5대5가르마펌x": [
        "mid",
        "가르마 머리",
        "앞머리볼륨",
        "다운펌"
    ],
    "6대4가르마펌o": [
        "비대칭",
        "mid",
        "가르마 머리",
        "앞머리볼륨",
        "컬"
    ],
    "6대4가르마펌x": [
        "비대칭",
        "mid",
        "가르마 머리",
        "앞머리볼륨",
        "다운펌"
    ],
    "히피펌": [
        "long",
        "내린 머리",
        "앞머리볼륨",
        "옆볼륨",
        "컬"
    ],
    "플랫": [
        "mid",
        "올린 머리",
        "앞머리볼륨",
        "비대칭",
        "다운펌"
    ]
    ,
    "슬립백" : [
        "올린 머리",
        "mid",
        "다운펌"
    ]
}
# "straight" "비대칭","뒷머리"   

Girl_features_df = {
    "index": [0, 1, 2, 3],
    "짧은 미간": ["묶음머리", "가르마머리","레이어드","위볼륨"],
    "긴 미간": ["내린머리", "위볼륨","앞머리볼륨","사이드"],
    "긴 코": ["가르마머리", "위볼륨", "short","컬","사이드"],
    "긴 이마": ["내린머리", "mid","레이어드","위볼륨"],
    "광대가 돌출된 얼굴": ["mid", "long","레이어드","컬","가르마머리","사이드"],
    "긴 턱": ["short","레이어드","옆볼륨", "컬","내린머리"],
    "짧은 턱": ["long", "mid", "위볼륨","가르마머리","컬","레이어드"],
    "짧은 이마": ["묶음머리", "위볼륨","long","스트레이트"],
    "짧은 코": ["묶음머리", "위볼륨","long","컬","사이드","가르마머리"],
    "긴 얼굴": ["내린머리","mid", "long","컬","레이어드","사이드","옆볼륨"],
    "둥근형":["위볼륨","묶음머리","가르마 머리","short","mid"],
    "세모형":["앞머리볼륨","옆볼륨","mid","long","컬"] ,
    "각진형":["mid","위볼륨","묶음머리","가르마 머리"]
    }

girl_hairstyles = {
    "앞o단발허쉬컷": [
        "short",
        "내린머리",
        "앞머리볼륨",
        "옆볼륨",
        "레이어드"
    ],

    "앞x단발허쉬컷": [
        "short",
        "가르마머리",
        "옆볼륨",
        "레이어드"
    ],

    "보브컷": [
        "short",
        "가르마",
        "옆볼륨",
        "위볼륨"
    ],

    "앞o장발히피펌": [
        "long",
        "내린머리",
        "앞머리볼륨",
        "옆볼륨",
        "위볼륨",
        "컬"
    ],

    "앞x장발히피펌": [
        "long",
        "가르마머리",
        "옆볼륨",
        "위볼륨",
        "컬"
    ],

    "앞o테슬컷": [
        "short",
        "내린머리",
        "스트레이트"
    ],

    "앞x테슬컷": [
        "short",
        "가르마머리",
        "스트레이트"
    ],

    "앞o히메컷": [
        "long",
        "내린머리",
        "스트레이트",
        "단차" #사이드
    ],

    "앞x히메컷": [
        "long",
        "가르마머리",
        "스트레이트",
        "단차" #사이드
    ],

    "앞o슬릭컷": [
        "long",
        "내린머리",
        "스트레이트"
    ],

    "앞x슬릭컷": [
        "long",
        "가르마 머리",
        "스트레이트"
    ],

    "앞o단발레이어드c컬": [
        "short",
        "내린머리",
        "앞머리볼륨",
        "옆볼륨",
        "위볼륨",
        "레이어드",
        "컬"
    ],

    "앞x단발레이어드c컬": [
        "short",
        "가르마머리",
        "옆볼륨",
        "위볼륨",
        "레이어드",
        "컬"
    ],

    "앞o단발히피펌": [
        "short",
        "내린머리",
        "앞머리볼륨",
        "옆볼륨",
        "위볼륨",
        "컬"
    ],

    "앞x단발히피펌": [
        "short",
        "가르마 머리",
        "앞머리볼륨",
        "옆볼륨",
        "위볼륨",
        "컬"
    ],

    "앞o장발레이어드c컬": [
        "long",
        "내린머리",
        "앞머리볼륨",
        "옆볼륨",
        "위볼륨",
        "컬",
        "레이어드"
    ],

    "앞x장발레이어드c컬": [
        "long",
        "가르마머리",
        "옆볼륨",
        "위볼륨",
        "레이어드",
        "컬"
    ],

    "앞o숏컷": [
        "short",
        "내린머리",
        "앞머리볼륨",
        "옆볼륨",
        "위볼륨"
    ],

    "앞x숏컷": [
        "short",
        "가르마머리",
        "앞머리볼륨",
        "옆볼륨",
        "위볼륨"
    ],

    "앞o빌드펌": [
        "long",
        "내린머리",
        "앞머리볼륨",
        "옆볼륨",
        "컬"
    ],

    "앞x빌드펌": [
        "long",
        "가르마머리",
        "옆볼륨",
        "컬"
    ],

    "앞o윈드펌": [
        "short",
        "내린 머리",
        "앞머리볼륨",
        "옆볼륨",
        "레이어드"
    ],

    "앞x윈드펌": [
        "short",
        "가르마 머리",
        "옆볼륨"
        "위볼륨"
    ],

    "긴머리사이드뱅": [
        "long",
        "가르마 머리",
        "앞머리볼륨",
        "위볼륨",
        "옆볼륨",
        "컬"
    ],

    "앞o젤리펌": [
        "mid to long",
        "내린 머리",
        "앞머리볼륨",
        "옆볼륨",
        "위볼륨",
        "컬"
    ],

    "앞x포니테일": [
        "가르마 머리"
        "long",
        "묶음머리"

     ]
    ,
    "앞o포니테일": [
        "내린머리",
        "long",
        "묶음머리",
        "앞머리볼륨"
    ]



    ,"앞x젤리펌": [
        "mid","long",
        "가르마머리",
        "앞머리볼륨",
        "옆볼륨",
        "위볼륨",
        "컬"
    ]

}

def log_scale(count):
    """빈도수를 로그 스케일로 변환합니다"""
    return math.log(count + 1)  # 1을 더해 log(0)을 방지합니다


def custom_weight(feature, preferences):
    """사용자 선호도에 따라 가중치를 동적으로 설정."""

    # fringe가 문자열인지 딕셔너리인지 확인하여 처리
    if isinstance(preferences['fringe'], dict):
        fringe_weights = {
            "올린 머리": preferences['fringe'].get('up', 1.0),
            "가르마 머리": preferences['fringe'].get('side', 1.0),
            "내린 머리": preferences['fringe'].get('down', 1.0)
        }
    else:
        # 문자열일 경우 기본 가중치 설정
        fringe_weights = {
            "올린 머리": 1.0,
            "가르마 머리": 1.0 if preferences['fringe'] == '가르마' else 0.5,
            "내린 머리": 0.5 if preferences['fringe'] == '내린 머리' else 1.0
        }

    # length가 문자열인지 딕셔너리인지 확인하여 처리
    if isinstance(preferences['length'], dict):
        length_weights = {
            "short": preferences['length'].get('short', 1.0),
            "mid": preferences['length'].get('medium', 1.0),
            "long": preferences['length'].get('long', 1.0)
        }
    else:
        # 문자열일 경우 기본 가중치 설정
        length_weights = {
            "short": 0.5 if preferences['length'] == '짧은' else 1.0,
            "mid": 1.0,
            "long": 1.0 if preferences['length'] == '긴' else 0.5
        }

    # curl도 문자열인지 딕셔너리인지 확인하여 처리
    curl_weight = preferences.get('curl', 1.0) if isinstance(preferences.get('curl'), (int, float)) else 1.0

    # 특징에 맞는 가중치 반환
    weights = {
        **fringe_weights,
        "앞머리볼륨": 1.0,  # 이 부분은 기본값 그대로
        "웨이브": 1.0,
        "위볼륨": 1.0,
        "옆볼륨": 1.0,
        **length_weights,
        "컬": curl_weight
    }

    return weights.get(feature, 1.0)  # 정의되지 않은 특징은 기본값 1.0을 사용




# 얼굴 특징들에 어울리는 헤어 특징들을 count하는 함수
# 중복되는 헤어특징들이 무엇인지 확인
def features_counter(features_list):
    a = []
    for i in features_list:
        a += features_df.get(i, [])
    return dict(Counter(a))


# 모든 weighted_count 함수들 정의
def weighted_count0(features_count, feature, preferences):
    return log_scale((1 + (0.1 * features_count.get(feature, 0))) * custom_weight(feature, preferences))

# 변형 1: 로그 스케일 제거
def weighted_count21(features_count, feature, preferences):
    return (1 + (0.1 * features_count.get(feature, 0))) * custom_weight(feature,preferences)

# 변형 2: 가중치 팩터 파라미터화
def weighted_count22(features_count, feature, preferences, weight_factor=0.1):
    return log_scale((1 + (weight_factor * features_count.get(feature, 0))) * custom_weight(feature,preferences))

# 변형 3: 지수 함수 사용
from math import exp
def weighted_count23(features_count, feature,preferences):
    return exp((0.1 * features_count.get(feature, 0)) * custom_weight(feature,preferences))



# 변형 6: 두 개의 가중치 사용
def weighted_count24(features_count, feature,preferences, weight_factor1=0.1, weight_factor2=0.5):
    return log_scale((1 + (weight_factor1 * features_count.get(feature, 0))) *
                     (1 + (weight_factor2 * custom_weight(feature,preferences))))

# # 변형 7: 최대값으로 정규화
# def weighted_count25(features_count, feature, max_count):
#     normalized_count = features_count.get(feature, 0) / max_count
#     return log_scale((1 + normalized_count) * custom_weight(feature))

# 변형 8: 비선형 변환 후 로그 스케일
def weighted_count26(features_count, feature,preferences):
    x = (0.1 * features_count.get(feature, 0)) * custom_weight(feature,preferences)
    return log_scale(1 + x**2)

# 변형 9: 조화 평균 사용
def weighted_count27(features_count, feature, preferences):
    count = features_count.get(feature, 0)
    weight = custom_weight(feature,preferences)
    return 2 / ((1/count) + (1/weight)) if count != 0 and weight != 0 else 0




# 1. 로그 대신 제곱근 사용
def weighted_count1(features_count, feature,preferences):
    return math.sqrt((1 + features_count.get(feature, 0)) * custom_weight(feature,preferences))
def weighted_count2(features_count, feature,preferences):
    x = (0.1 * features_count.get(feature, 0)) * custom_weight(feature,preferences)
    return tanh(x)


# 3. 쌍곡선 사인 함수 사용
def weighted_count3(features_count, feature,preferences):
    return math.sinh(0.1 * features_count.get(feature, 0) * custom_weight(feature,preferences))

# 4. 아크탄젠트 함수 사용
def weighted_count4(features_count, feature,preferences):
    return math.atan(features_count.get(feature, 0) * custom_weight(feature,preferences))

# 5. 베타 함수를 이용한 정규화
def weighted_count5(features_count, feature,preferences, alpha=2, beta=2):
    x = features_count.get(feature, 0) / max(features_count.values())
    return math.pow(x, alpha - 1) * math.pow(1 - x, beta - 1)

# 6. 가우시안 함수 사용
def weighted_count6(features_count, feature,preferences, mu=5, sigma=2):
    x = features_count.get(feature, 0)
    return math.exp(-((x - mu)**2) / (2 * sigma**2)) * custom_weight(feature,preferences)

# 7. 로지스틱 함수 사용
def weighted_count7(features_count, feature,preferences,k=1):
    x = features_count.get(feature, 0)
    return 1 / (1 + math.exp(-k * x)) * custom_weight(feature,preferences)

# 8. 멱함수 사용
def weighted_count8(features_count, feature,preferences,power=0.5):
    return math.pow(1 + features_count.get(feature, 0), power) * custom_weight(feature,preferences)

# 9. 이항 계수를 이용한 조합
def weighted_count9(features_count, feature,preferences, n=10):
    k = min(features_count.get(feature, 0), n)
    return math.comb(n, k) * custom_weight(feature,preferences)

# 10. 조화수 시리즈 사용
def weighted_count10(features_count, feature,preferences):
    n = features_count.get(feature, 0)
    return sum(1/i for i in range(1, n+1)) * custom_weight(feature,preferences)

# 11. 소수 기반 가중치
def weighted_count11(features_count, feature,preferences):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    n = min(features_count.get(feature, 0), len(primes)-1)
    return primes[n] * custom_weight(feature,preferences)

# 12. 피보나치 수열 기반 가중치
def weighted_count12(features_count, feature,preferences):
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    n = min(features_count.get(feature, 0), len(fib)-1)
    return fib[n] * custom_weight(feature,preferences)

# 13. 쿼드라틱 함수 사용
def weighted_count13(features_count, feature,preferences, a=0.1, b=0.5, c=1):
    x = features_count.get(feature, 0)
    return (a * x**2 + b * x + c) * custom_weight(feature,preferences)

# 14. 로그와 지수 함수의 혼합
def weighted_count14(features_count, feature,preferences):
    x = features_count.get(feature, 0)
    return (math.log(1 + x) + math.exp(0.1 * x)) * custom_weight(feature,preferences)

# 15. 삼각함수 조합
# def weighted_count15(features_count, feature):
#     x = features_count.get(feature, 0)
#     return (math.sin(x) + math.cos(x)) * custom_weight(feature)

# 16. 역수 함수 사용
def weighted_count16(features_count, feature,preferences):
    x = features_count.get(feature, 0)
    return (1 / (1 + x)) * custom_weight(feature,preferences)

# 17. 로그 시그모이드 함수
def weighted_count17(features_count, feature,preferences):
    x = features_count.get(feature, 0)
    return 1 / (1 + math.exp(-log_scale(x))) * custom_weight(feature,preferences)

# 18. 쌍곡선 탄젠트와 시그모이드의 조합
def weighted_count18(features_count, feature,preferences):
    x = features_count.get(feature, 0)
    return (math.tanh(x) + 1 / (1 + math.exp(-x))) * 0.5 * custom_weight(feature,preferences)

# 19. 루트-로그 조합
def weighted_count19(features_count, feature,preferences):
    x = features_count.get(feature, 0)
    return math.sqrt(1 + math.log(1 + x)) * custom_weight(feature,preferences)

# 20. 지수화된 시그모이드
def weighted_count20(features_count, feature,preferences):
    x = features_count.get(feature, 0)
    return math.exp(1 / (1 + math.exp(-x))) * custom_weight(feature,preferences)

# 유사도 계산 함수
def calculate_similarity(features_count, attributes, weighted_count_func, preferences):
    set1 = set(features_count.keys())
    intersection = sum(weighted_count_func(features_count, item, preferences) for item in set1 for item2 in attributes if item == item2)
    union = sum(weighted_count_func(features_count, item, preferences) for item in set1.union(attributes))
    return intersection / union if union != 0 else 0


# features_coun은 필요한 헤어특징: 중복횟수  attributes는 한 헤어스타일의 특징들
def improved_jaccard_similarity(features_count , attributes):
    set1=set(features_count.keys())

    '''
    자카드는 교집합/ 합집합 이고
    코사인 유사도는 두 벡터 사이의 사잇각(내적)으로 유사도 측정하는 것
    '''

    def weighted_count0(feature):
        # 중복되는 헤어특징의 갯수가 n이라면 (1.n) * (사용자의 선호도)
        return log_scale((1+(0.1*features_count.get(feature, 0)))  * custom_weight(feature))



    intersection = sum(weighted_count0(item) for item in set1 for item2 in attributes if item == item2)
    union = sum(weighted_count0(item) for item in set1.union(attributes))

    jaccard_similarity = intersection / union if union != 0 else 0



    return jaccard_similarity

# 성별에 따른 헤어스타일 추천
def recommend_hairstyles(user_face_features, male_hairstyles, girl_hairstyles, gender, top_n=3):
    """성별에 따라 얼굴 특징과의 유사도를 기반으로 헤어스타일을 추천합니다."""
    
    preferences = get_user_preferences()
    if preferences is None:
        print("사용자 선호도 데이터를 찾을 수 없습니다.")
        return []
    
    similarities = defaultdict(float)
    user_face_feature_set = user_face_features
    features_count = features_counter(user_face_feature_set)

    # 성별에 따라 다른 헤어스타일을 사용
    if gender == '남성':
        hairstyles = male_hairstyles
    else:
        hairstyles = girl_hairstyles

    # 모든 weighted_count 함수 테스트
    weighted_count_functions = [weighted_count0, weighted_count1, weighted_count2,
                                weighted_count3, weighted_count4, weighted_count5,
                                weighted_count6, weighted_count7, weighted_count8,
                                weighted_count9, weighted_count10, weighted_count11,
                                weighted_count12, weighted_count13, weighted_count14,
                                weighted_count16, weighted_count17,
                                weighted_count18, weighted_count19, weighted_count20,
                                weighted_count21, weighted_count22, weighted_count23,
                                weighted_count24, weighted_count26,
                                weighted_count27]

    print("얼굴 특징", user_face_feature_set)
    print("헤어 특징", features_count)

    results = {}

    # 각 weighted_count 함수에 대해 모든 헤어스타일의 유사도 계산
    for i, func in enumerate(weighted_count_functions):
        similarities = {}
        for style, attributes in hairstyles.items():
            similarity = calculate_similarity(features_count, set(attributes), func, preferences)
            similarities[style] = similarity
        results[f'weighted_count{i}'] = similarities

    # 데이터프레임 생성
    df = pd.DataFrame(results).T
    df.mean().sort_values(ascending=False).round(4)

    print(similarities.items())
    
    # 유사도 점수를 기준으로 헤어스타일을 내림차순으로 정렬합니다
    sorted_styles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # 상위 N개의 추천 결과를 반환합니다
    return sorted_styles[:top_n]

# 메인 실행 부분
# 메인 실행 부분
if __name__ == "__main__":
    # MongoDB에서 가장 최근 이미지를 가져오기
    image, file_id = get_latest_image()

    if image is not None:
        # MongoDB에서 가져온 이미지를 사용해 분석
        user_face_features = face_analyze(image)
        recommendations = recommend_hairstyles(user_face_features, male_hairstyles)

        print("추천 헤어스타일:")
        for style, similarity in recommendations:
            print(f"{style}: {similarity}")
    else:
        print("분석할 새로운 이미지가 없습니다.")



