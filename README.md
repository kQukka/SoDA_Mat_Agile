
# SoDA_Mat_Agile

Aiffel Hackathon <2022.04.25 ~ 2022.06.07>

팀 구성


| 이름   |  구성   |                      역할                  |
| :----: | :----: |  :---------------------------------------: | 
| 이정현  |  팀장   | 프로젝트 관리 및 설계, 통합 개발 및 테스트   | 
| 고용석  |  팀원   | 물류센터 가상 환경 생성, 데이터 전송 테스트   | 
| 김영태  |  팀원   | 물류센터 가상 환경 생성, 데이터 전송 테스트 | 
| 고세영  |  팀원   | 최적화 경로 추출 모듈 개발, 물류센터 가상 환경, 최적화 경로 추출 연동  | 
| 김현빈  |  팀원   | 최적화 경로 추출 모듈 개발, 물류센터 가상 환경, 최적화 경로 추출 연동  |  
# 
### 1. 개요
<br/>

### 주제
- 강화학습을 활용한 최단경로 아이템회수 에이전트 생성
<br/>

### 문제정의
<br/>

![image](https://user-images.githubusercontent.com/80939966/172124579-69a71d65-4448-4db1-ad51-465e9b1d4227.png)

- 9 x 10의 가상환경에서 주문된 아이템 리스트를 입력 받아 모두 회수해 최단거리로 되돌아오는 에이전트 생성
- 에이전트는 시작지점에서 출발하여 아이템을 모두 회수한 후 시작지점으로 되돌아 옴
<br/>

### 프로젝트 목표
- 최단거리가 되는 루트로 주문된 아이템을 찾아 다시 돌아오는 agent를 생성해, 물류센터에 직접 적용 시킬수있는 알고리즘을 개발하고 비대면 온라인 소비 증가에 따른 물류센터의 운영비용 과 업무오류를 감소시키고자 함

<br/>

### 전체 일정

<br/>

![image](https://user-images.githubusercontent.com/80939966/172230038-84107b19-e175-4429-aa13-efd6b46fefb6.png)

#

### 2. 알고리즘 구현
<br/>

![image](https://user-images.githubusercontent.com/80939966/172190703-a89d873e-6bea-4c95-a158-fda1d9970045.png)

<br/>


### Q-Learning
<br/>

![image](https://user-images.githubusercontent.com/80939966/172215358-a06a3ca2-a626-4bfb-8dea-8a2a6e00c1c2.png)

Q-Learning은 4x4의 환경에서 매우높은 성공률과 안정성을 확인함.

<br/>

### Q-NetWork
<br/>

![image](https://user-images.githubusercontent.com/80939966/172215620-862833b9-d2c7-460e-a93b-b2ae88566956.png)
Q-NetWork는 4x4의 환경에서 괜찮은 성능을 보이지만 Q-Learning보단 부족함을 보임.

<br/>

### DQN
<br/>

![image](https://user-images.githubusercontent.com/80939966/172220161-fe335861-13f3-40cf-aaa2-9ec8cae74beb.png)

DQN에서는 본 과제의 가상환경을 적용한 여러가지 테스트를 하였지만 몇몇 문제점 때문에 만족스러운 결과를 내지못하였다.

<br/>

### DQN모델의 문제점
- 같은 자리에서 계속 맴도는 에이전트
- 에이전트의 탐험에 대한 문제
- 목표지점에 도달했지만 불필요한 스텝을 많이 밟는 경우
- 목표지점에 적은 스텝수로 도달했지만 Q-map이 좋지 못한 경우
- 목표 아이템이 붙어있는 경우에 아이템 회수후 다음스텝에서 바로 종료되는 문제

<br/>

따라서 이러한 문제점을 해결하기 위해 다음과 같은 시도를 하였다.
<br/>

#

### 3. 성능향상을 위한 시도(DQN)
<br/>

#### 하이퍼파라미터 튜닝

- hidden size test

- discount factor test

- learning rate test

- sampling test

- q-map optimization test

- 1 q-value optimzation test

- move reward test

- target update cycle test

<br/>

여러가지 하이퍼파라미터를 셋팅하고 테스트한 결과 가장높은 평균 성공률과 q-map을 보이는 하이퍼파라미터 값

![image](https://user-images.githubusercontent.com/80939966/172229470-8d78e7cb-c881-496e-be78-1113d07e7355.png)



<br/>

#### GOAL 버퍼의 사용
- 기존버퍼의 랜덤 샘플링에는 성공 케이스가 운에 따라 포함되었지만 goal buffer를 만들어 줌으로써 유의미한 학습이 이루어지게 하여 성능을 개선함. 또한, 성공케이스만 포함 하는것 보단 랜덤 샘플링을 적절히 섞어주는 것이 더 좋은 성능을 보임.

<br/>

#### REWARD의 재설정

- 목표 아이템이 붙어있는 경우에 아이템 회수후 다른 아이템을 장애물로 처리하게 하면서 바로 에피소드가 종료되는 문제 해결

#

### 4. agent 평가
<br/>

#

### 5. 결과
<br/>

#

### Competition

- http://www.k-digitalhackathon.kr/
 
![image](https://user-images.githubusercontent.com/80939966/172231094-e91494b3-ba7b-4d84-a62a-499080e9bf50.png)

- 최종지원서 https://www.notion.so/2022-K-9a61f5c8d6bc4394969a33c52a2e6c31
- 예선진출 및 진행중
