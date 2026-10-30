from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
import base64

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
    all_answers: str = Form(...)  # Đã sửa thành all_answers để khớp với App (Hết lỗi 422)
):
    try:
        # 1. Đọc và giải mã hình ảnh từ App gửi lên
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Tiền xử lý ảnh (Đen trắng, làm mờ, nhị phân hóa)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # 3. Đọc dữ liệu toàn bộ đáp án
        answer_dict = json.loads(all_answers)
        
        # ---------------------------------------------------------
        # BƯỚC 4: NHẬN DIỆN MÃ ĐỀ (CẦN TỌA ĐỘ GIẤY THỰC TẾ)
        # ---------------------------------------------------------
        # TODO: Chèn thuật toán OpenCV đọc khung chứa Mã Đề ở đây.
        # Tạm thời đặt mặc định là mã đề đầu tiên có trong danh sách nếu chưa có thuật toán quét
        detected_ma_de = list(answer_dict.keys())[0] if answer_dict else "Không rõ"
        
        correct_keys = answer_dict.get(detected_ma_de, {})
        total_questions = len(correct_keys)
        correct_count = 0
        details = []

        # ---------------------------------------------------------
        # BƯỚC 5: CHẤM ĐIỂM (CẦN TỌA ĐỘ GIẤY THỰC TẾ)
        # ---------------------------------------------------------
        # Đã xóa dòng "correct_count += 1" cũ của bạn gây lỗi 10 điểm ảo.
        for q_no, correct_ans in correct_keys.items():
            # TODO: Chèn thuật toán đếm pixel đen (cv2.countNonZero) tại ô A, B, C, D ở đây
            # Giả lập thí sinh tô toàn đáp án A (để test logic)
            student_ans = "A" 
            
            if student_ans == correct_ans:
                correct_count += 1
                details.append(f"Câu {q_no}: ĐÚNG")
                # Vẽ viền xanh nếu đúng
            else:
                details.append(f"Câu {q_no}: SAI")
                # Vẽ viền đỏ nếu sai

        # 6. Tính điểm (Thang điểm 10)
        score = round((correct_count / total_questions) * 10, 2) if total_questions > 0 else 0.0

        # 7. Viết điểm số và mã đề lên góc ảnh
        cv2.putText(img, f"Ma de: {detected_ma_de} | Diem: {score}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 8. Mã hóa ảnh đã chấm thành Base64 để gửi về cho điện thoại hiển thị
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "score": score,
            "ma_de": detected_ma_de,
            "correct_count": correct_count,
            "total": total_questions,
            "details": ", ".join(details),
            "marked_image": img_base64
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
