from Object_Detection.basic_draft import *
from image_to_text.demo import *
from Distance_Measure.distance import *
from Text_Searching.search import *
import winsound

from speech_recog import * ###
class Handler:
    def __init__(self, label_map,distance_label_map, caption_model_name, timegap=15, beepsoundgap = 10, max_length = 16, num_beams = 4):
        #tts setting 
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 300)  # 음성 속도 조절, 수정 가능;
        
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
        # self.file_path = './image_to_text/img/cat.jpg'

        self.distance_label_map = distance_label_map
        
        self.beepsoundgap = beepsoundgap
        
        #stt.
        self.transcriber = WhisperTranscriber()
        #text_searching 
        self.searchtext = Trie()
        self.img_description_bool = False
        self.obj_detection_bool = False

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

    def FocalLength(self,img_distance, img_width, ref_image_obj_width):
        focal_length = (ref_image_obj_width * img_distance)/ img_width
        return focal_length

    def Distance_finder (self,Focal_length, real_face_width, width_in_frame):
        distance = (real_face_width * Focal_length)/width_in_frame
        return distance

    def obj_width_finder(self,image):
        obj_width = 0
        results = self.model(image)
        
        for detection in results.xyxy[0]:
            x_min, y_min, x_max, y_max = map(int, detection[:4])
            obj_width = x_max-x_min
        return obj_width

    def example_distance(self):
        for obj in self.distance_label_map.keys():
            img_distance = self.distance_label_map[obj][0]
            img_width = self.distance_label_map[obj][1]
            img_path = 'Distance_Measure/my_image/'+obj+'.jpg'
            ref_image = cv2.imread(img_path)
            ref_image = cv2.resize(ref_image, (640, 480))

            ref_image_obj_width = self.obj_width_finder(ref_image)  #frame 너비
            Focal_length = self.FocalLength(img_distance,img_width,ref_image_obj_width)
            self.distance_label_map[obj].extend([ref_image_obj_width,Focal_length])

    def run(self):
        #print(self.caption_model.predict([self.file_path]))
        self.example_distance()
        while True:
            ret, frame = self.cap.read()
            results = self.model(frame)
            current_detection = list()

            class_labels = results.names #distance에서 사용
            
            if int(time.time())  % self.beepsoundgap == 0:
                beepsound = True

            for detection in results.xyxy[0]:
                class_id = int(detection[5])
                class_label = self.model.names[class_id]
                if class_label in self.label_map:
                    current_detection.append(self.label_map[class_label])
                
                
##############
                width_in_frame = 0
                x_min, y_min, x_max, y_max = map(int, detection[:4])
                width_in_frame = x_max-x_min
                class_index = int(detection[5])
                class_name = class_labels[class_index]

                if class_name in self.distance_label_map.keys():
                    distance = self.distance_label_map[class_name][0]
                    width = self.distance_label_map[class_name][1]
                    ref_imge_obj_width = self.distance_label_map[class_name][2]
                    Focal_length = self.distance_label_map[class_name][3]

                    real_width = (width_in_frame/ref_imge_obj_width)*width 

                    if width_in_frame != 0 :
                        Distance = self.Distance_finder(Focal_length,width,width_in_frame)
                        if (Distance <=150) and (beepsound == True) :
                            print("전방에"+class_name+"이있습니다.")
                            winsound.Beep(440,1000)           # hz,  지속시간,  경고음 . .
                            beepsound = False

            #############
            
            
            
            frame = results.render()[0]
            cv2.imshow("Frame", frame)
            
            
            #for debug
            results.print()
                        
            current_detection = sorted(current_detection)
            
            result_text = ""
            if cv2.waitKey(1) & 0xFF == ord('a'):
                audio = self.transcriber.record_audio()
                self.transcriber.save_audio(audio)

                result_text = self.transcriber.transcribe_audio()
                print(result_text)
            
            if (result_text != ""):
                word_list = result_text.split()
                for word in word_list:
                    self.searchtext.insert(word)
                self.img_description_bool =(self.searchtext.search("풍경") or self.searchtext.search("묘사"))
                self.obj_detection_bool = (self.searchtext.search("객체") or self.searchtext.search("인식"))
                print(self.img_description_bool, self.obj_detection_bool)
                result_text = "" 
                self.searchtext.clear()
            
            
            #출력 condition
            if (int(time.time())  % self.timegap == 0) or (self.obj_detection_bool == True):
                #
                print("객체인식ㄱ")
                print(self.generate_sentence(current_detection))
                self.engine.say(self.generate_sentence(current_detection))
                self.engine.runAndWait()
            
            
            
            #from cv2capture
            if (cv2.waitKey(1) & 0xFF == ord('s')) or (self.img_description_bool == True):
                #
                print("풍경묘사====")
                print(self.caption_model.predict_from_frame(frame))                
                self.engine.say(self.caption_model.predict_from_frame(frame))
                self.engine.runAndWait()
            
            self.obj_detection_bool = False
            self.img_description_bool = False

            #종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 수정가능; 인식하고자 하는 객체. 
    label_map = {
        "person": "사람",
        "chair": "의자",
        "cell phone": "휴대폰",
        "truck": "트럭",
        "traffic light" : "신호등",
        "clock" : "시계"
    }
    distance_label_map = {
        "cell phone" : [35,16],
        "bicycle": [100,50],
        "car": [300,180],
        "truck": [250,250],
        "fire hydrant" : [150,60]
    }

    model_name = "nlpconnect/vit-gpt2-image-captioning"

    handler = Handler(label_map, distance_label_map,model_name )
    handler.run()
