model used for obj detection : yolov5s.  출처 : https://github.com/ultralytics/yolov5

선행되어야 하는 작업.
pip install ultralytics

git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install

코드 실행 시 감지하고 싶은 객체에 대해서 dictionary 형태로 label_map에 yolov5s가 인식할 수 있는 딕셔너리를 만들어 주어야 함.
