from __future__ import unicode_literals
from __future__ import division

import cv2 # 카메라 열어서 이미지 처리
import numpy as np
import cx_Oracle
import hgtk
import time
import threading

from flask import Flask, render_template, Response, request, redirect, url_for

# Oracle 서버와 연결(Connection 맺기)
conn = cx_Oracle.connect('system', '111111', 'localhost:1521/xe')
# Connection 확인
print(conn.version)

app = Flask(__name__)

result = "null"
pResult = "null"
kResult = "null"
answer = ""
DECOMPOSED = ""
bClick = 0
ans = []
bear = []
i = 0
finger = 0
frame2 = ""
isNum = 0
sum = 0

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


@app.route('/sendData')
def sendData():
    global result
    global kResult
    global pResult
    global answer
    global ans
    global i

    if pResult != result:
        result = pResult

        #DataBase
        # Oracle DB의 test_member 테이블 검색(select)
        cursor = conn.cursor()  # cursor 객체 얻어오기
        cursor.execute(f"select KOR from trans where ENG like '{result}'")  # SQL 문장 실행
        row = cursor.fetchone()
        kResult = row[0]
        print("2 " + kResult)
        # Oracle 서버와 연결 끊기
        conn.close

        answer += " " + kResult
        ans.insert(i, kResult)
        i = i + 1

    return answer

@app.route('/comb')
def comb():
    global ans
    global DECOMPOSED
    global finger
    global isNum
    global sum
    ansN = []
    plusN = []
    #ans = ['ㅅ', 'ㅡ', 'ㅇ', 'ㅇ', 'ㅕ', 'ㄴ']
    consonant = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ' 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    vowel = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    bear = []
    bearM = 'ᴥ'
    k = 0

    ansL = len(ans)

    # 자음 1, 모음 0
    for n in range(0, ansL):
        isCon = 0
        for c in consonant:
            if ans[n] in c:
                isCon = 1
                finger = 1
                ansN.insert(n, 1)
                break
        if isCon == 0:
            for v in vowel:
                if ans[n] in v:
                    finger = 1
                    ansN.insert(n, 0)
                    break

        if ans[n].isdigit(): #숫자인지 검사
            isNum = 1
            sum += int(ans[n])

    if ansL < 4:
        pass
    else:
        # 한 글자씩 끊기는 위치 저장
        for a in range(0, ansL-1):
            if ansN[a] == 0:
                if ansN[a + 1] == 1:
                    if(a + 2 <= ansL-1):
                        if ansN[a + 2] == 0:
                            bear.insert(k, a)
                            k = k + 1

            elif ansN[a] == 1:
                if ansN[a + 1] == 1:
                    bear.insert(k, a)
                    k = k + 1

    for d in range(0, ansL):
        for b in bear:
            if d == b:
                DECOMPOSED += ans[d] + 'ᴥ'
                break
            else:
                DECOMPOSED += ans[d]
                break

    DECOMPOSED += 'ᴥ'
    print(DECOMPOSED)
    ans = []
    return test_compose()

def test_compose():
    comAns = hgtk.text.compose(DECOMPOSED)
    print("compose", comAns)
    return comAns

@app.route('/sendResult')
def sendResult():
    global finger
    global answer
    global isNum
    global sum

    fingerL = comb()

    if finger == 1:
        finger = 0
        answer = ""
        return fingerL

    if isNum == 1:
        sentence = sum
        sum = 0
        isNum = 0
        return str(sentence)

    else:
        sentence = answer
        answer = ""
        return sentence

@app.route('/click')
def click():
    global bClick
    global result
    global pResult
    global i
    global DECOMPOSED

    #버튼 온
    if bClick == 0:
        bClick = 1
        result = ""
        pResult = ""
        DECOMPOSED = ""

    #버튼 오프
    elif bClick == 1:
        bClick = 0
        i = 0

    print(bClick)
    return bClick

@app.route('/delM')
def delM():
    global result
    global pResult
    global answer
    global ans
    global i

    result = ""
    pResult = ""
    answer = ""
    ans = []
    i = 0

    return result

@app.route('/post', methods = ['POST'])
def post():
    isClick = request.form['transB']
    print("버튼" + isClick)

@app.route('/trans')
def trans():
    return render_template('trans.html', data = sendData())

@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)

    cam2 = threading.Thread(target=read_cam, args=(camera,))

    cam2.start()

    return Response(myCamera(camera), mimetype='multipart/x-mixed-replace; boundary=mycam')

@app.route("/") #라우팅 설정
def main():
    #return "Welcome"
    return render_template('main.html')

@app.route('/main')
def backMain():
   return render_template('main.html')

@app.route('/manualSelect')
def manual():
   return render_template('manualSelect.html')

@app.route('/hospital')
def hospital():
   return render_template('hospital.html')

@app.route('/login')
def login():
   return render_template('login.html')

@app.route('/sign')
def sign():
   return render_template('sign.html')

if __name__ == "__main__":
    app.run()

if __name__ == '__main__':
    app.run(debug=True)