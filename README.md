  # Hair_StyleFit
 
  <br>
  <br>

  ## 1. 📖 프로젝트 개요


  - 보통, 사람들은 자신의 얼굴형에 대한 정보를 정확하게 알지 못하고, 어떤 헤어스타일이 어울릴지 알기는 쉽지가 않다
    
  - 이 프로젝트는 개인의 얼굴 형태와 이목구비를 분석하여, 그에 적합한 헤어스타일을 추천해주는 시스템
  
  - 이를 통해 사용자에게 얼굴형에 대한 정보를 전달하고, **개인적인 헤어스타일 선호도**를 고려하여 스타일 제안

  - 또한, **합성 기능**을 이용하여 자신에게 어울리는 헤어스타일을 시각적으로 확인할 수 있음.

  <br>
  <br>

  ## 2. 🏗 시스템 구조


  <img src="https://github.com/user-attachments/assets/27404326-41a9-46fe-939c-0efc0e36e4cb" alt="image" width="500">


  <br>
  <br>
  <br>



  ## 3. 🛠 기술 스택

  - **프로그래밍 언어**: Python, JavaScript, html, css
  - **프레임워크 및 라이브러리**: MediaPipe, OpenCV, Pytorch
  - **데이터베이스**: MongoDB
  - **배포 환경**: Google Colab, Flask, Streamlit

    <br>
    <br>
    
  ## 4. 📁 데이터셋 
  * 기존 faceshape 데이터를 전처리 및 데이터를 추가적으로 수집하여 턱 데이터 셋을 직접 만들었다.
  * 턱 데이터셋 : https://www.kaggle.com/datasets/idealm99/chinshape2
  * 얼굴 데이터셋 : https://www.kaggle.com/datasets/niten19/face-shape-dataset
  * 추가한 얼굴 데이터셋 : https://www.kaggle.com/datasets/idealm99/newfaceshape7

  * opencv를 이용해서 얼굴 부분만 잘라서 저장

  * 추가된 얼굴 데이터셋에 턱에 대한 분류

  * 미디어파이프를 이용해 얼굴을 인식하고 이목구비 비율 분석

  * 8000장 정도의 사진의 표본을 통해 비율 분포를 분석



    <br>
    <br>


<div style="white-space: nowrap; overflow-x: auto; padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd;">
  <img src="https://github.com/user-attachments/assets/f4ca6bec-bb53-4139-85ca-adf170189f30" alt="image1" width="350" style="display: inline-block; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/fb6c7b19-9da6-43db-85ab-800dbb729500" alt="image2" width="350" style="display: inline-block; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/f7213892-99eb-43cd-8162-5d3cd8e3cbb5" alt="image3" width="350" style="display: inline-block; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/3b924b80-336e-4dca-8537-a0220a1782ce" alt="image4" width="350" style="display: inline-block;">
</div>






  <br>
  <br>



  ## 🤖 5. 모델 학습 및 평가



  - 모델의 학습은 Colab + 환경에서 
  
  - GPU: NVIDIA Tesla V100, P100, T4 GPU를 이용해 학습을 진행하였다. 
  
  - efficientnet B4 모델을 이용해 학습을 시켜 정확도를 높였고, 최종 정확도는 92~94%였다.

  - 가중치를 반영한 자카드 유사도를 이용하여 헤어스타일 추천 
  
  <br>
  <br>

  ## ⚙️ 6. 시스템 구현




  1. **서버 구축**: Flask 프레임워크를 기반으로 서버를 구축하고, MongoDB를 통해 사용자 데이터를 관리
  2. **프론트엔드 개발**: html, css, javascript
  3. **얼굴분석**: MediaPipe와 OpenCV를 사용하여 얼굴 특징을 추출하고 분석
  4. **헤어스타일 추천**: 자카드 알고리즘을 이용한 추천 시스템 개발 
  5. **API 연동**: AILabTools API를 통해 사용자 얼굴과 추천된 헤어스타일의 합성 결과


  <br>
  <br>

  ## 📊 7. 결과 및 시연



  <img src="https://github.com/user-attachments/assets/081ab4cd-b5db-44ec-9ccf-f70e3e9d349e" alt="image1" width="350">
  <img src="https://github.com/user-attachments/assets/61f35c7f-eee5-42bb-96f9-0dfc08dc93f8" alt="image2" width="327">
  <img src="https://github.com/user-attachments/assets/3bc726f9-b9f1-4e1e-9fb9-1607ddab375a" alt="image3" width="347">
  <img src="https://github.com/user-attachments/assets/826f0f62-9958-4384-9f62-d92bada318c1" alt="image4" width="335">

  <br>
  <br>


  ## 📚 참고 자료

  

  - [MediaPipe Documentation](https://google.github.io/mediapipe/)
  - [TensorFlow Guide](https://www.tensorflow.org/guide)
  - https://www.gqkorea.co.kr/2017/04/26/%EC%96%BC%EA%B5%B4%ED%98%95%EC%97%90-%EB%A7%9E%EB%8A%94-%ED%97%A4%EC%96%B4-%EC%8A%A4%ED%83%80%EC%9D%BC%EB%A7%81/
  - https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=86bf56a212ae9d9cffe0bdc3ef48d419
  - https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=f617cde073fb0ac4ffe0bdc3ef48d419


  <br>
  <br>




  ## 🖥️ 역할분담


  - 팀장 이상민
    - 자카드 알고리즘을 이용한 추천시스템 개발
    - 턱모양 구별 모델 훈련 
    - 얼굴형 비율 분석을 위한 미디어파이프 구상
    - faceshape 데이터를 전처리 및 데이터를 추가적으로 수집하여 턱 데이터 셋을 직접 생성
    - 기획 



  - 팀원 정재형
    - DB 구현 및 턱모양 구별 모델 훈련
    - front 구현 
    - README.md 작성 
    - api 연동
    - 기획



