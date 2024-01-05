# Signal

<img src="https://user-images.githubusercontent.com/79898245/154721052-6da910b3-aba1-4766-b4e8-c71841a43b94.png" width="1200">

---

<img src="https://user-images.githubusercontent.com/79898245/154711596-6ae4f6bf-05b3-4dc8-b55b-683a448bb4f9.gif">

<img src="https://user-images.githubusercontent.com/79898245/154716004-ca134dc9-2640-491b-952c-b9b9531e05b9.gif">
   
---

## **[ About ]**
> ### **청각장애인을 위한 핸드 시그널 커뮤니케이션 서비스**
본 프로젝트는 한양여자대학교 소프트웨어융합과 졸업작품 & ICT 이브와 멘토링 공모전 작품으로 진행되었습니다.

청각장애인의 수화 동작을 실시간 영상 처리로 인식하고 자막을 통해 비장애인이 인지할 수 있는 통역 서비스를 지원합니다.<br>
청각장애인은 수화를 이용하고 비장애인은 청각과 음성 언어를 이용해 의사소통 가능합니다.

### Intention
1. 청각장애인과의 원활한 의사소통 

2. 수화통역사 부족 해결

### Benefit
* 사회적 고립감 최소화

* 수어 통역사 인력 대체

* 업무 수행 지원

### Development period
20.06.10 ~ 20.10.30

---

## **[ Technical Skills ]**

*   Programming Language
    *   Python, YOLO v3, OpenCV, flask, MySQL
*   Tools
    *   Pycharm

---

## **[ Tech Flow ]**
<img src="https://user-images.githubusercontent.com/79898245/154717487-c7e67b54-3e4d-4c0e-8e11-d99f773e9eea.png">

- **MTT (Motion To Text)** : 청각장애인이 행하는 수화 동작을 인식하여 텍스트로 번역한다.
- **TTS (Text To Speech)** : 텍스트로 번역된 수화 동작을 비장애인이 음성으로 들을 수 있다. 
- **STT (Speech To Text)** : 비장애인의 음성 신호를 텍스트화 하여 청각장애인에게 제공될 수 있도록 한다.

## **[ Role ]**
* 수화 데이터 학습
: 높은 정확도를 위해 여러장의 이미지를 라벨링하여 데이터를 학습

* 지/수화 학습 가중치 파일 제작
: 학습된 수화 동작 데이터 가중치 파일 생성

* 수화 인식
: 학습된 가중치 파일을 기반으로 영상처리를 이용해 텍스트로 변환

* 자연스러운 단어 조합, 문장 제작
: 인식된 자음, 모음 단어 조합 및 문장 제작

## **[ Feature Implementation ]**
<img src="https://user-images.githubusercontent.com/79898245/154719527-417de104-6a5e-4dc7-8a82-54fcaac1bbb5.png" width="450" align="right">
<img src="https://user-images.githubusercontent.com/79898245/154721691-8c6755f6-dfaf-4c13-a17c-2b10ec70b13e.png" width="350">

<br>

프로젝트 구현시 사용할 10개의 지/수화를 다른 각도와 다른 배경에서 영상을 촬영해 이를 100 프레임으로 잘라내어 `Dataset`을 확보하였다.

---

<br>

<img src="https://user-images.githubusercontent.com/79898245/154724518-e215043e-f530-4518-b3ea-d6233a1a6218.png">

확보한 Dataset을 한 장씩 손 영역만큼 지정하여 직접 `labeling` 하였다.

---

<img src="https://user-images.githubusercontent.com/79898245/154719694-4f6c0cc1-26b9-41fa-bf94-886a63b8f369.png" align="right" width="500">
<img src="https://user-images.githubusercontent.com/79898245/154719411-79d58a5d-d79a-466c-8260-1c8e57784a69.png" width="300">

<br>

가중치 파일 제작시 학습 현황을 알려주는 그래프로 `x축은 학습 횟수`, `Y축은 손실율`을 나타내며 0에 수렴할수록 정확도가 높아지게 된다. <br>
정확도를 높이기 위해 0에 수렴할 때까지 학습을 시켜주었으며 <br>
`10000장의 Dataset을 24000번의 학습`을 통해 얻은 가중치 파일의 정확도는 평균적으로 **92%** 에 달하는 것을 확인할 수 있었다.

---

## **[ How To Use ]**
<img src="https://user-images.githubusercontent.com/79898245/154726206-7781454f-375d-4c6f-8245-5014eaea8135.png">

시그널 웹의 메인 페이지, 그룹 채팅 / 번역 / 시그널 서비스를 선택할 수 있다.

---

<br>

<img src="https://user-images.githubusercontent.com/79898245/154726322-1eb0c8a5-ffd5-44b7-9c6c-83e35160fb3b.png">

시그널 서비스를 눌렀을 때 이동하는 페이지로, 청각장애인이 업무를 수행하는 데 있어 가장 어려워하는 곳 4개를 선정하였다. <br>
해당 기관 선택시 이용에 필요한 응대 절차 서비스를 제공받을 수 있다. 

---

<br>

<img src="https://user-images.githubusercontent.com/79898245/154726399-1fa156e8-0919-4ace-969f-8a253eecce3b.png">

병원 서비스를 선택하였을 때의 모습이다. 진료를 통해 병원 접수 서비스를 제공받을 수 있다.

---

<br>

<img src="https://user-images.githubusercontent.com/79898245/154726457-6f980601-be7c-4857-a547-b44452b9b268.png">

왼쪽은 **카메라**를 통해 수화를 하는 나의 모습을 확인할 수 있으며, 오른쪽은 병원과의 소통을 **채팅 형식**으로 이루어지도록 제작하였다. <br>
왼쪽에 나타나는 말풍선은 병원측에서 제공하는 **응대 질문**이 전송되며, 청각장애인이 **수화로 답변**한 내용은 오른쪽 말풍성으로 전달되게 한다. <br>

<br>

카메라 화면 아래쪽의 가운데 `녹색 버튼`을 눌러 **수화**를 시작할 수 있다. <br>
버튼이 눌리면 색과 아이콘이 변경되며 수화를 인식할 수 있는 상태임을 알 수 있다. <br>
청각 장애인이 수화를 하는 것에 대해 오른쪽 화면 아래 입력창에서 학습된 가중치 파일을 기반으로 수화를 인식하여 텍스트로 바꾸는 `MTT(Motion To Text)` 결과값을 확인할 수 있고 <br>
인식이 올바르게 되지 않았을 경우 `삭제 버튼`을 눌러 다시 수화를 시작할 수 있다. <br>
수화를 마치고 다시 한번 가운데 버튼을 누르면 인식된 결과값들이 **단어** 또는 **문장 형식**으로 만들어진다. <br>

<br>

전송 버튼을 누르면 대화창으로 **답변이 전송**되며, 이떄 비장애인이 청각장애인의 응답을 들을 수 있도록 `TTS(Text To Speech)` 기능이 실행되어 응답을 소리로 전달한다. <br>
정해진 응대 매뉴얼 외에 추가적인 질문사항에 대해 카메라 화면 첫 번째 파란색 버튼을 눌러 `STT(Speech To Text)` 기능을 사용할 수 있다. <br>
사용자의 음성을 인식하고 텍스트로 변환하여 청각장애인에게 전달할 수 있다. <br>

<br>

마지막으로 `빨간 버튼`을 눌러 해당 매뉴얼 서비스를 종료할 수 있다. <br>

---

## **[ Problem & Solution ]**

### 초기 프로젝트 진행 방향 - 립모션을 통한 구현

<img src="https://user-images.githubusercontent.com/79898245/154730743-269c6ab0-1d88-45d0-90d3-64320c968070.png">

우리의 수화를 텍스트로 번역하는 초기 구현 방법은 **립모션** 기기를 이용한 것이었다. <br>
립모션을 이용해 수화를 비교하고 인식하기 위해서 손에서의 어떤 요소들을 비교 / 분석하면 될지에 대해 공부하였고 <br>
`손끝의 기울기 (방향)`, `접힌 손가락의 개수` 등을 알아낼 수 있었고, 수화에 대한 데이터를 뽑아내 실시간 `손의 움직임 좌표`와 비교하여 결과를 출력해낼 생각이었다.

<br>

<img src="https://user-images.githubusercontent.com/79898245/154730651-f1f9ac50-fa90-4815-8a36-9d73e9f5e990.png">

하지만 실사용해 본 결과 수화를 행하기엔 립모션의 인식 한계가 존재하였고 지화만을 구현하기로 하여도 손의 움직임이 **정확하지 않고 일관적이지 않다**는 문제점이 존재하였다.
<br>
이 결과로 인해 **립모션**에서 **영상 처리**를 이용한 구현 방법을 채택해 새로이 도전해보게 되었다.

---

### 실시간 카메라의 프레임 속도와 끊김 현상
객체를 인식하고 검출하는 데 있어 소요되는 시간으로 인해 실시간으로 보이는 카메라의 프레임이 맞지 않았다. <br>
이를 해결하기 위해 `스레드 분리`법을 공부하였으며 `멀티 스레드` 방식을 적용시켜 작업을 원활히 할 수 있도록 해주었다. <br>
카메라를 담아내는 화면에서 끊김 현상없이 실시간 모습을 자연스럽게 담아낼 수 있었고 객체를 검출하는 과정에서 스레드 분리를 통해 업무를 나누어 원활히 실행될 수 있었다.

---

## **[ Key Code ]**

```
def myCamera(camera):
    if not camera.isOpened():
        raise RuntimeError('연결된 카메라가 있는지 확인 요함.')

    while True:
        _, frame = camera.read()
        yield (b'--mycam\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))[1].tobytes()
               + b'\r\n')

def read_cam(camera):
    if not camera.isOpened():
        raise RuntimeError('연결된 카메라가 있는지 확인 요함.')

    # YOLO 가중치 파일과 CFG 파일 로드
    YOLO_net = cv2.dnn.readNet("yolov3_last.weights", "yolov3.cfg")
    # YOLO NETWORK 재구성
    classes = []
    with open("classes.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]

    while True:
        _, frame = camera.read()
        h, w, c = frame.shape

        # YOLO 입력
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        YOLO_net.setInput(blob)
        outs = YOLO_net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)
                    # Rectangle coordinate
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]

                # 경계상자와 클래스 정보 이미지에 입력
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5,
                            (255, 255, 255), 1)

                # 결과값 겹치지 않게 하나씩 출력
                global pResult

                if pResult != label:
                    pResult = label
                    print("1" + pResult)

        # 윈도우창, 여기서 라벨링 박스 확인
        cv2.imshow("YOLOv3", frame)

        if cv2.waitKey(100) > 0:
            break



@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)

    cam2 = threading.Thread(target=read_cam, args=(camera,))

    cam2.start()

    return Response(myCamera(camera), mimetype='multipart/x-mixed-replace; boundary=mycam')
```

## **[ Youtube ]**
https://youtu.be/1MKZ8jkfqYs
