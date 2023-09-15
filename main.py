from handlerclass import *

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
    model_name = "nlpconnect/vit-gpt2-image-captioning"

    handler = Handler(label_map, model_name)
    handler.run()