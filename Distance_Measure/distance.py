import cv2
import torch

class DistanceDetection:
    def __init__(self,distance_label_map) -> None:

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.distance_label_map = distance_label_map

############################################################################

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
            img_path = "my_image/"+obj+".jpg"
            ref_image = cv2.imread(img_path)
            ref_image_obj_width = self.obj_width_finder(ref_image)  #frame 너비
            Focal_length = self.FocalLength(img_distance,img_width,ref_image_obj_width)
            self.distance_label_map[obj].extend([ref_image_obj_width,Focal_length])

    def run(self):
        self.example_distance()
        while True:
            _, frame = self.cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame_rgb)
            class_labels = results.names

            for detection in results.xyxy[0]:
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

                    if width_in_frame != 0:
                        Distance = self.Distance_finder(Focal_length,width,width_in_frame)

                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2

                    cv2.putText(frame,f"Distance = {Distance:.2f}" , (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                #frame = results.render()[0]
                cv2.imshow("frame",frame)

            if cv2.waitKey(1) == ord("q"):
                break 

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 수정가능; 인식하고자 하는 객체. 
    distance_label_map = {
        "cell phone" : [30,16],
        # "bicycle": [200,60],
        "car": [300,180],
        # "traffic light": [200,60],
        # "truck": [300,250],
        # "fire hydrant" : [200,60],
        # "stop sign" : [200,60]
    }
    obj_detector = DistanceDetection(distance_label_map)
    obj_detector.run()