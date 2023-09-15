from Object_Detection.basic_draft import *
from image_to_text.demo import *


class Handler:
    def __init__(self, label_map, caption_model_name, timegap=15,max_length = 16, num_beams = 4):
        #tts setting 
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 200)  # 음성 속도 조절, 수정 가능;
        
        #model set
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        
        #cv2 capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 30frame/s  : 수정 가능; 
        
        #t초마다,, 
        self.timegap = timegap
        
        #인식할 객체 
        self.label_map = label_map
        
        #image to txt
        self.caption_model = ImageCaptioningModel(caption_model_name, max_length, num_beams)
        self.file_path = './image_to_text/img/cat.jpg'
        
    #detection 된 객체 리스트를 바탕으로 텍스트 생성;
    def generate_sentence(self, lst):
        label_counts = {}

        for label in lst:
            if label in self.label_map.values():
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

    def run(self):
        print(self.caption_model.predict([self.file_path]))

        while True:
            ret, frame = self.cap.read()
            results = self.model(frame)
            current_detection = list()

            for detection in results.xyxy[0]:
                class_id = int(detection[5])
                class_label = self.model.names[class_id]
                if class_label in self.label_map:
                    current_detection.append(self.label_map[class_label])

            frame = results.render()[0]
            cv2.imshow("Frame", frame)
            
            
            #for debug
            results.print()
            #
                        
            current_detection = sorted(current_detection)
            
            #출력 condition
            if int(time.time())  % self.timegap == 0:
                print(self.generate_sentence(current_detection))
                ##이후 tts로 바꾸어야 할 부분, 
            
            #from image path
            if cv2.waitKey(1) & 0xFF == ord('a'):
                print(self.caption_model.predict([self.file_path]))
            
            #from cv2capture
            if cv2.waitKey(1) & 0xFF == ord('s'):
                print(self.caption_model.predict_from_frame(frame))                

            #종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # 수정가능; 인식하고자 하는 객체. 
#     label_map = {
#         "person": "사람",
#         "chair": "의자",
#         "cell phone": "휴대폰",
#         "truck": "트럭",
#         "traffic light" : "신호등",
#         "clock" : "시계"
#     }
#     model_name = "nlpconnect/vit-gpt2-image-captioning"

#     handler = Handler(label_map, model_name )
#     handler.run()