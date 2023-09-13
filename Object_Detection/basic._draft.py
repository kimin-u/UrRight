#tts threading > 음성이 출력되는 동안 cv2.videocapture를 멈춤. > 다른 tts방법을 찾아보기 ? 
#현재 코드는 음성출력을 하지 않는 코드. 
#         engine.say("string type example.")
#         engine.runAndWait()

import cv2
import torch
import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 200)  # 음성 속도 조절, 수정 가능;

model = torch.hub.load("ultralytics/yolov5", "yolov5s")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # 30frame/s  : 수정 가능; 

#수정가능;
label_map = {
    "person": "사람",
    "chair": "의자",
    "cell phone": "휴대폰",
    "truck": "트럭",
    "traffic light" : "신호등",
    "clock" : "시계"
}


def generate_sentence(lst):
    label_counts = {}

    for label in lst:
        if label in label_map.values():
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

    result = []

    for label, count in label_counts.items():
        if count > 1:
            result.append(f'다수의 {label}, ')
        else:
            result.append(f'{label}, ')

    if result:
        result[-1] = result[-1][:-2] + '이 앞에 있습니다.'

    return ''.join(result)


## 수정가능 ##
compare_frame = 50   # 직전 50개의 프레임과 비교, (새로운 객체 등장여부) 


prev_detections = []  # 이전 n개 프레임에서의 객체 리스트

while True:
    ret, frame = cap.read()
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #results = model(frame_rgb)
    results = model(frame)
    current_detection = list()

    for detection in results.xyxy[0]:
        class_id = int(detection[5])
        class_label = model.names[class_id]
        if class_label in label_map:
            current_detection.append(label_map[class_label])

    
    #for debug.
    results.print()
    print(current_detection)    
    ##
    
    frame = results.render()[0]
    cv2.imshow("Frame", frame)

    current_detection = sorted(current_detection)
    if len(prev_detections) >= compare_frame:
        is_new_detection = all(prev_detections[-1] != detection for detection in prev_detections[-compare_frame:-1])
        if is_new_detection:
            print(generate_sentence(current_detection))


    prev_detections.append(current_detection.copy())  # 복사본을 저장
    
    if len(prev_detections) > compare_frame:
        prev_detections.pop(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()