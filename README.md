# HK_capstone - Hair_StyleFit

# 1. 프로젝트 개요

> 보통 사람들은 자신의 얼굴형에 대한 정보를 정확하게 알지 못하고 어떤 헤어스타일이 자신에게 잘 어울리는지 알기는 쉽지가 않다.
> 
> 그래서 개인의 얼굴 형태와 이목구비를 분석하여 얼굴형태와 이목구비에 적합한 헤어스타일을 추천을 해주는 시스템을 만들면 흥미롭겠다라는 생각에 접근을 했다.
> 
>  해당 프로젝트는 개인의 얼굴형에 대한 정보를 전달하고, 사용자 얼굴형의 특징과 개인적인 헤어스타일에 대한 선호도를 고려해서 헤어스타일을 추천 받을 수 있다.
> 
> 또한 합성 기능을 이용해 시각적으로 자신에게 어울리는 헤어스타일을 직접 고를 수 있는 기능을 제공한다.


<br>

# 2. 시스템 구조


<img src="https://github.com/user-attachments/assets/27404326-41a9-46fe-939c-0efc0e36e4cb" alt="image" width="500">


<br>
<br>

# 3. 기술 스택

![icons8-파이썬-48](https://github.com/user-attachments/assets/db6f37ec-2a10-4eb9-984d-5ca155dfa8e5) ![icons8-파이토치-48](https://github.com/user-attachments/assets/96c3fa48-f014-49bf-b382-4ef5f584ada7) 
![icons8-mysql의-50](https://github.com/user-attachments/assets/12b8dc6c-bd6a-4329-9c2d-4e35bde09dce) 
![icons8-aws-48](https://github.com/user-attachments/assets/993f4a08-6700-45dc-973c-b6bc15b24c0a) ![icons8-html-50](https://github.com/user-attachments/assets/87e8a477-2e53-4429-ab83-5a7331b26ca8) ![icons8-css-50](https://github.com/user-attachments/assets/607be868-c7af-43f0-ab1a-a2ef44993d4e) ![icons8-js-50](https://github.com/user-attachments/assets/d67575c1-b050-4709-a6d8-0d6ecbc77d1c)







* python
* pytourch
* MYSQL
* AWS
* pandas
* computer vision

<br>
<br>

# 4. 데이터셋 
* 기존 faceshape 데이터를 전처리 및 데이터를 추가적으로 수집하여 턱 데이터 셋을 직접 만들었다.
* 턱 데이터셋 : https://www.kaggle.com/datasets/idealm99/chinshape2
* 얼굴 데이터셋 : https://www.kaggle.com/datasets/niten19/face-shape-dataset
* 추가한 얼굴 데이터셋 : https://www.kaggle.com/datasets/idealm99/newfaceshape7

* opencv를 이용해서 얼굴 부분만 잘라서 저장

* 추가된 얼굴 데이터셋에 턱에 대한 분류

* 미디어파이프를 이용해 얼굴을 인식하고 이목구비 비율 분석

* 8000장 정도의 사진의 표본을 통해 비율 분포를 분석





<img src="https://github.com/user-attachments/assets/f4ca6bec-bb53-4139-85ca-adf170189f30" alt="image" width="400"> <img src="https://github.com/user-attachments/assets/fb6c7b19-9da6-43db-85ab-800dbb729500" alt="image" width="400">
<img src="https://github.com/user-attachments/assets/f7213892-99eb-43cd-8162-5d3cd8e3cbb5" alt="image1" width="400"> <img src="https://github.com/user-attachments/assets/3b924b80-336e-4dca-8537-a0220a1782ce" alt="image2" width="400">




<br>
<br>



# 5. 모델 학습 및 평가

// 턱 모양 분류 학습 내용과 정확도 
// 대충 어떤 모델을 이용해 학습했고, 앙상블 학습 ... 이런거 쓰면 좋을듯 
// 코랩 + 환경에서 ~ 

모델의 학습은 Colab + 환경에서 GPU를 이용해 학습을 진행하였다. 
~모델과 ~모델을 앙상블 학습을 시켜 정확도를 높였고, 최종 정확도는 ~였다.
<br>
<br>

# 6. 시스템 구현
// 어떤 내용을 넣으면 좋을까? 

<br>
<br>

# 7. 결과 및 시연

// 사진 및 영상 따로 넣어주면 좋을듯 
// 이거는 내가 나중에 한 번에 작업해서 올릴게 


<img src="https://github.com/user-attachments/assets/081ab4cd-b5db-44ec-9ccf-f70e3e9d349e" alt="image1" width="370">
<img src="https://github.com/user-attachments/assets/61f35c7f-eee5-42bb-96f9-0dfc08dc93f8" alt="image2" width="350">
<img src="https://github.com/user-attachments/assets/3bc726f9-b9f1-4e1e-9fb9-1607ddab375a" alt="image3" width="370">
<img src="https://github.com/user-attachments/assets/826f0f62-9958-4384-9f62-d92bada318c1" alt="image4" width="370">

<br>
<br>

# 8. 참고문헌 

1. https://www.gqkorea.co.kr/2017/04/26/%EC%96%BC%EA%B5%B4%ED%98%95%EC%97%90-%EB%A7%9E%EB%8A%94-%ED%97%A4%EC%96%B4-%EC%8A%A4%ED%83%80%EC%9D%BC%EB%A7%81/
2. https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=86bf56a212ae9d9cffe0bdc3ef48d419
3. https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=f617cde073fb0ac4ffe0bdc3ef48d419
4. 

<br>
<br>
# 9. 역할분담 


//+ // 추가적으로 우리 준비했던 ppt 내용도 같이 올리면 좋을듯. 어떤 방향으로 진행되었는지 ..참고 
