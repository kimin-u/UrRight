from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image


#클래스 내 함수 :
#predict :  원래 image_path 기준으로 이미지묘사하는 기존 코드 
# preprocess, predict_from frame  : cv2의 frame단위 캡쳐 화면을 기준으로 이미지 묘사

class ImageCaptioningModel:
    def __init__(self, model_name, max_length = 16, num_beams = 4):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.max_length = max_length
        self.num_beams = num_beams
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

    # predict from image_path.
    def predict(self, image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
    
    def preprocess_frame(self, frame):
        # Convert frame to PIL Image
        frame_pil = Image.fromarray(frame)

        # Ensure the image is in RGB mode
        if frame_pil.mode != "RGB":
            frame_pil = frame_pil.convert(mode="RGB")

        return frame_pil

    def predict_from_frame(self, frame):
        frame_pil = self.preprocess_frame(frame)

        pixel_values = self.feature_extractor(images=[frame_pil], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

# if __name__ == "__main__":
#     model_name = "nlpconnect/vit-gpt2-image-captioning"
#     captioning_model = ImageCaptioningModel(model_name)
#     file_path = './image_to_text/img/test.jpg'
#     predictions = captioning_model.predict([file_path])
#     print(predictions)
