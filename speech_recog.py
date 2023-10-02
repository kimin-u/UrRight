import whisper
import speech_recognition as sr
from Text_Searching.search import *

class WhisperTranscriber:
    def __init__(self, timeout = 5):
        self.model = whisper.load_model("base")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.timeout = timeout

    def record_audio(self):
        with self.microphone as source:
            print("마이크 시작")
            audio = self.recognizer.listen(source, timeout= self.timeout)
            print("끝")
            return audio

    def save_audio(self, audio, filename="audio.wav"):
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())

    def transcribe_audio(self, filename="audio.wav"):
        result = self.model.transcribe(filename)
        return result["text"]

# if __name__ == "__main__":
#     transcriber = WhisperTranscriber()

#     audio = transcriber.record_audio()
#     transcriber.save_audio(audio)

#     result_text = transcriber.transcribe_audio()
#     print(result_text)
    