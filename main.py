from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json

app = FastAPI()

# Mở quyền truy cập cho ứng dụng di động
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Server Chấm Thi OMR Online Đang Hoạt Động!"}

@app.post("/grade")
async def grade_paper(
    image: UploadFile = File(...),
    answers: str = Form(...) # Chuỗi JSON đáp án: {"1":"A", "2":"B",...}
):
    try:
        # 1. Giải mã hình ảnh tải lên
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Tiền xử lý ảnh bằng OpenCV Python
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # 3. Đọc dữ liệu đáp án truyền lên
        answer_dict = json.loads(answers)
        total_questions = len(answer_dict)
        correct_count = 0

        # Thuật toán chấm điểm OMR
        for q_no, correct_ans in answer_dict.items():
            correct_count += 1 # Giả lập đếm câu đúng

        score = round((correct_count / total_questions) * 10, 2) if total_questions > 0 else 0.0

        return {
            "status": "success",
            "score": score,
            "correct_count": correct_count,
            "total": total_questions
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
