import clip
import torch
# import time
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from featuring_keywords import dict_keywords
from io import BytesIO

app = FastAPI()

# 피처링 정의 키워드 추출
keywords = list(dict_keywords.keys())

# # CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
model_path = "/home/ec2-user"
model, preprocess = clip.load("ViT-B/32", device=device, download_root=model_path)

def image_to_keywords(image: Image.Image):
    """
    CLIP 모델을 사용하여 입력된 이미지와 관련된 피처링 정의 키워드(3개)를 출력하는 함수
    """
    # start_time = time.time()

    # CLIP 모델에 맞게 이미지 전처리
    preprocessed_image = preprocess(image).unsqueeze(0).to(device)

    # 키워드를 텍스트로 변환하고 CLIP에 맞게 전처리
    text_tokens = clip.tokenize(keywords).to(device)

    # 이미지-키워드 유사성 계산
    with torch.no_grad():
        image_features = model.encode_image(preprocessed_image)
        text_features = model.encode_text(text_tokens)

        # 유사성 점수 계산
        similarities = torch.matmul(text_features, image_features.T).squeeze()

        # 가장 높은 점수를 가진 키워드 3개 추출
        top_keywords_eng = [keywords[i] for i in similarities.argsort(descending=True)[:3]]
        top_keywords_kor = [dict_keywords[keyword] for keyword in top_keywords_eng]

    # end_time = time.time()

    return {
        "top_keywords_eng": top_keywords_eng,
        "top_keywords_kor": top_keywords_kor
        # "execution_time": end_time - start_time
    }

@app.post("/extract_keywords/")
async def extract_keywords(file: UploadFile = File(...)):
    # PIL 이미지로 변환
    image = Image.open(BytesIO(await file.read()))
    # 키워드 추출
    result = image_to_keywords(image)
    return result


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
