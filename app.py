from flask import Flask, render_template, request, jsonify, url_for
from pymongo import MongoClient
import requests
import gridfs
from werkzeug.utils import secure_filename
import os
import time  # 분석 상태 시뮬레이션을 위해
from PIL import Image
import io
from datetime import datetime
from analy import face_analyze, recommend_hairstyles, male_hairstyles, girl_hairstyles, get_user_preferences
from flask import send_from_directory
from collections import Counter  # 여기에 추가

app = Flask(__name__)






# 파일 크기 제한 추가
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
# MongoDB 연결 설정
client = MongoClient('mongodb+srv://sparta:test@cluster0.50ukg43.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['dbsparta']
preferences_collection = db['preferences']
fs = gridfs.GridFS(db)  # GridFS 사용해 파일 저장

# 전역 상태 변수
analyzing_progress = 0


@app.route('/apply_hair_synthesis', methods=['POST'])
def apply_hair_synthesis():
    try:
        image_data = request.files['image']
        hair_style = request.form['hair_style']
        
        api_key = 'none'
        url = 'https://www.ailabapi.com/api/portrait/effects/hairstyle-editor-pro'
        headers = {'ailabapi-api-key': api_key}
        
        files = [('image', (secure_filename(image_data.filename), image_data, 'application/octet-stream'))]
        data = {'hair_style': hair_style, 'color': 'brown', 'task_type': 'async'}

        response = requests.post(url, headers=headers, files=files, data=data)
        response_json = response.json()

        if response.status_code == 200:
            task_id = response_json.get('task_id')
            image_url = get_async_result(task_id)

            if image_url:
                return render_template('next_page.html', image_url=image_url)
            else:
                return jsonify({'error': 'Hair synthesis result not available yet'}), 500
        else:
            return jsonify({'error': 'Hair synthesis failed', 'details': response_json}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    

    
def get_async_result(task_id): 
    url = "https://www.ailabapi.com/api/common/query-async-task-result"
    headers = {
        'ailabapi-api-key': 'none' 
    }
    params = {'task_id': task_id}  # task_id를 쿼리 파라미터로 전달

    for attempt in range(20):
        response = requests.get(url, headers=headers, params=params)
        print(f"Attempt {attempt + 1}: Get async result response: {response.status_code}")

        if response.status_code == 200:
            try:
                response_json = response.json()
                print(f"Response JSON: {response_json}")

                task_status = response_json.get('task_status')
                if task_status == 2:
                    return response_json.get('data', {}).get('images', [None])[0]
                elif task_status in [0, 1]:
                    print("Waiting for the task to complete...")
            except ValueError:
                print("Invalid JSON response received.")
                return None
        else:
            print(f"Error retrieving async result: {response.status_code}")
        
        time.sleep(10)

    return None




    
# 홈 페이지 (index.html) 서빙
@app.route('/')
def index():
    return render_template('index.html')

# 카메라 페이지 (camera.html) 서빙
@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/custom')
def custom_hair_synthesis():
    return render_template('custom.html')

@app.route('/next_page', methods=['GET'])
def next_page():
    return render_template('next_page.html')

from flask import send_from_directory

# hairfit 폴더에서 이미지를 서빙하는 엔드포인트 추가
@app.route('/hairfit/<path:filename>')
def serve_image(filename):
    return send_from_directory('C:/Users/USER/Desktop/hairfit', filename)




    
def process_image_from_db(file_id):
    """GridFS에서 이미지 가져와서 처리하는 함수"""
    try:
        # MongoDB에서 이미지 가져오기
        grid_out = fs.get(file_id)
        image_data = grid_out.read()

        # 디버깅: 이미지 데이터를 확인해봄
        print(f"Image data length: {len(image_data)} bytes")  # 이미지 데이터 길이 출력

        # 바이트 데이터를 NumPy 배열로 변환하여 OpenCV로 처리
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # 디버깅: OpenCV로 읽은 이미지 확인
        if image is None:
            print(f"Error: 이미지가 로드되지 않았습니다. 파일 ID: {file_id}")
            return None

        # 이미지 크기 조정
        scale_factor = 1  # 크기 조정 인자
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        return resized_image

    except Exception as e:
        print(f"Error during image processing: {e}")
        return None


@app.route('/submit_preferences', methods=['POST'])
def submit_preferences():
    try:
        data = request.json
        gender = data.get('gender')
        
        # 선호도 데이터
        preferences_data = {
            'gender': gender,
            'fringe': {
                'up': float(data['fringe']['up']),
                'side': float(data['fringe']['side']),
                'down': float(data['fringe']['down']),
            },
            'length': {
                'long': float(data['length']['long']),
                'medium': float(data['length']['medium']),
                'short': float(data['length']['short']),
            },
            'curl': float(data['curl']),
            'uploadDate': datetime.utcnow()  # 타임스탬프 추가
        }

        # MongoDB에 데이터 삽입
        preferences_collection.insert_one(preferences_data)

        return jsonify({'message': 'Preferences saved successfully!'}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to save preferences'}), 500

    
# 이미지 업로드 API
@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        print(f"파일 이름: {file.filename}")

        # 파일을 PIL Image로 열기
        image = Image.open(file)

        # 이미지 크기 줄이기 (예: 1024x1024로 조정)
        image = image.resize((1024, 1024))

        # 이미지 데이터를 바이트로 변환
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # GridFS에 저장
        file_id = fs.put(img_byte_arr, filename=secure_filename(file.filename), content_type='image/jpeg')
        print(f"파일이 성공적으로 저장되었습니다. 파일 ID: {file_id}")

        return jsonify({'message': 'Image uploaded successfully!', 'file_id': str(file_id)}), 200
    except Exception as e:
        print(f"파일 저장 오류: {e}")
        return jsonify({'error': 'Failed to upload image', 'details': str(e)}), 500


@app.route('/chin_shape_result')
def chin_shape_result():
    latest_image = list(fs.find().sort([('uploadDate', -1)]).limit(1))  # 커서를 리스트로 변환
    if len(latest_image) > 0:
        for grid_out in latest_image:
            image_data = grid_out.read()
            image_path = "latest_image.jpg"
            with open(image_path, 'wb') as img_file:
                img_file.write(image_data)

            user_face_features = face_analyze(image_path)
            chin_shape_counter = Counter(user_face_features[:3])
            total = sum(chin_shape_counter.values())
            percentages = {shape: round((count / total) * 100, 1) for shape, count in chin_shape_counter.items()}
            dominant_shape = max(chin_shape_counter, key=chin_shape_counter.get)

            result = {"percentages": percentages, "dominant_shape": dominant_shape}
    
    return render_template('chin_shape_result.html', result=result)

from collections import Counter

@app.route('/facial_features')
def facial_features():
    # 이미지 경로 설정
    # 
    image_path = "latest_image.jpg"  # 상위 폴더 경로 지정

    # 얼굴 분석 수행
    user_face_features = face_analyze(image_path)

    # 얼굴형만 따로 처리 (가장 많이 나온 얼굴형을 남기고 중복 제거)
    face_shapes = [feature for feature in user_face_features if feature in ['둥근형', '각진형', '세모형']]
    if face_shapes:
        # 가장 많이 나온 얼굴형만 남기기
        most_common_shape = Counter(face_shapes).most_common(1)[0][0]
        # 나머지 얼굴형 삭제하고, 얼굴 특징에는 가장 많이 나온 얼굴형만 추가
        unique_features = [feature for feature in user_face_features if feature not in face_shapes]
        unique_features.append(most_common_shape)
    else:
        unique_features = list(set(user_face_features))

    # 설명 데이터
    feature_descriptions = {
        "짧은 미간": "미간이 짧아 보입니다. 올린 머리나 가르마 머리가 어울립니다.",
        "긴 미간": "미간이 길어 보입니다. 내린 머리나 앞머리 볼륨을 주는 스타일이 어울립니다.",
        "긴 코": "코가 길어 보입니다. 옆 볼륨이나 컬 스타일로 얼굴의 밸런스를 맞추는 것이 좋습니다.",
        "긴 이마": "이마가 길어 보입니다. 중간 기장의 비대칭 스타일이 어울립니다.",
        "광대가 돌출된 얼굴": "광대가 돌출되어 보입니다. 중간 기장의 컬 스타일이 좋습니다.",
        "긴 턱": "턱이 길어 보입니다. 중간 또는 긴 기장의 옆 볼륨을 주는 스타일이 어울립니다.",
        "짧은 턱": "턱이 짧아 보입니다. 긴 기장이나 위 볼륨을 주는 스타일이 좋습니다.",
        "짧은 이마": "이마가 짧아 보입니다. 올린 머리나 위 볼륨 스타일이 이마를 보완해줍니다.",
        "짧은 코": "코가 짧아 보입니다. 올린 머리나 위 볼륨 스타일로 코를 강조할 수 있습니다.",
        "긴 얼굴": "얼굴이 길어 보입니다. 가르마 머리나 웨이브, 컬이 어울리며, 비대칭 스타일로 균형을 맞출 수 있습니다.",
        "짧은 얼굴": "얼굴이 짧아 보입니다. 짧은 머리나 내린 머리가 얼굴의 균형을 맞추는 데 도움이 됩니다.",
        "각진형": "각진 얼굴형입니다. 위 볼륨이나 올린 머리가 어울리며, 가르마 머리도 잘 어울립니다.",
        "세모형": "세모형 얼굴입니다. 앞머리 볼륨을 주거나 옆 볼륨 스타일이 어울립니다.",
        "둥근형": "둥근형 얼굴입니다. 중간 기장의 머리나 위 볼륨 스타일이 어울리며, 가르마 머리도 추천됩니다."
    }

    # 얼굴 사진 경로
    image_path = '/hairfit/latest_image.jpg'

    return render_template('facial_features.html', features=unique_features, descriptions=feature_descriptions, image_path=image_path)




@app.route('/hairstyle_recommendations')
def hairstyle_recommendations():
    try:
        # DB에서 가장 최근 이미지를 사용해 얼굴 분석 결과 가져오기
        latest_image = fs.find().sort([('uploadDate', -1)]).limit(1)

        if latest_image:
            for grid_out in latest_image:
                image_data = grid_out.read()
                image_path = "latest_image.jpg"
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_data)

                # 얼굴 분석 실행
                user_face_features = face_analyze(image_path)
                preferences = get_user_preferences()

                if preferences and 'gender' in preferences:
                    gender = preferences['gender']
                else:
                    gender = '남성'  # 기본 성별 설정

                # 추천 헤어스타일 생성
                recommendations = recommend_hairstyles(user_face_features, male_hairstyles, girl_hairstyles, gender)
                
                # 이미지와 스타일 이름 반환
                result = [(style, f'{style}.jpg') for style, _ in recommendations]
                return render_template('hairstyle_recommendations.html', recommendations=result)

        else:
            return jsonify({"error": "No image found"}), 404
    except Exception as e:
        print(f"Error generating hairstyle recommendations: {e}")
        return jsonify({"error": "Failed to generate recommendations"}), 500

# 분석 진행 상태 API
@app.route('/analyze_progress')
def analyze_progress():
    global analyzing_progress
    if analyzing_progress < 100:
        analyzing_progress += 20  # 매 호출마다 진행 상태 증가 (시뮬레이션)
    return jsonify({'percent': analyzing_progress}), 200

# 분석 결과 API (진행 후 실제 결과 반환)
@app.route('/get_analysis_result')
def get_analysis_result():
    global analyzing_progress

    # DB에서 최근 이미지 가져오기
    latest_image = fs.find().sort([('uploadDate', -1)]).limit(1)

    result = None

    for grid_out in latest_image:
        image_data = grid_out.read()  # 이미지 데이터를 가져옴
        image_path = "latest_image.jpg"  # 로컬 임시 파일로 저장
        with open(image_path, 'wb') as img_file:
            img_file.write(image_data)

        # 분석 로직 실행
        user_face_features = face_analyze(image_path)  # analyze.py의 분석 함수 호출
        preferences = get_user_preferences()

        if preferences and 'gender' in preferences:
            gender = preferences['gender']
        else:
            gender = '남성'  # 기본값

        print(f"Analyzing image for user with gender: {gender}")  # 성별 출력
        recommendations = recommend_hairstyles(user_face_features, male_hairstyles, girl_hairstyles, gender)

        print(f"Recommendations generated: {recommendations}")  # 추천 결과 출력

        chin_shape = user_face_features[0]  # 가장 많이 나온 턱 모양
        face_labels = user_face_features[3:]  # 얼굴 특징 라벨 리스트

        # 헤어스타일 이미지 파일명 결정 (static/images 폴더에 저장된 파일명 사용)
        best_hairstyle_image = f'{recommendations[0][0]}.jpg'  # 가장 유사한 헤어스타일의 이미지 파일명

        result = {
            "best_hairstyle": recommendations[0][0],  # 가장 유사한 헤어스타일
            "best_hairstyle_image": best_hairstyle_image,  # 이미지 파일명
            "other_recommendations": [style for style, _ in recommendations[1:3]],  # 상위 3개의 추천 헤어스타일
            "chin_shape": chin_shape,  # 가장 높은 빈도의 턱 모양
            "face_labels": face_labels  # 얼굴 특징 라벨
        }

    if result is None:
        print("No image or analysis result found.")  # 오류 발생 시 출력
        result = {"error": "No image found or analysis failed"}

    # analyzing_progress 초기화
    analyzing_progress = 0  

    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True)
