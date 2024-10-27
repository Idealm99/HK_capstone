# HK_capstone - Hair_StyleFit

# 1. 프로젝트 개요

> 사람들이 자신의 얼굴 형에 대한 정보를 정확하게 알지 못하고 무슨 헤어스타일이 잘 어울리는지 직접 해보지 못하면 알 수 없다. 그래서 저희는 그 사람의 얼굴 형태와 이목구비를 분석하여 가장 잘 어울리는 헤어스타일을 추천하는 어플리케이션을 만들고싶었습니다.

# 2. 시스템 구조

# 3. 기술 스택

* python
* pytourch
* MYSQL
* AWS
* pandas
* computer vision

# 4. 데이터셋 
* 기존 faceshape 데이터를 전처리 및 데이터를 추가적으로 수집하여 턱 데이터 셋을 직접 만들었습니다.
턱 데이터셋 : https://www.kaggle.com/datasets/idealm99/chinshape2
얼굴 데이터셋 : https://www.kaggle.com/datasets/niten19/face-shape-dataset
추가한 얼굴 데이터셋 : https://www.kaggle.com/datasets/idealm99/newfaceshape7

opencv를 이용해서 얼굴 부분만 잘라서 저장

추가된 얼굴 데이터셋에 턱에 대한 분류

미디어파이프를 이용해 얼굴을 인식하고 이목구비 비율 분석

8000장 정도의 사진의 표본을 통해 비율 분포를 분석




# 5. 모델 학습 및 평가

# 6. 시스템 구현

# 7. 결과 및 시연

# 8. 참고문헌 

1. https://www.gqkorea.co.kr/2017/04/26/%EC%96%BC%EA%B5%B4%ED%98%95%EC%97%90-%EB%A7%9E%EB%8A%94-%ED%97%A4%EC%96%B4-%EC%8A%A4%ED%83%80%EC%9D%BC%EB%A7%81/
2. https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=86bf56a212ae9d9cffe0bdc3ef48d419
3. https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=f617cde073fb0ac4ffe0bdc3ef48d419
4. 
