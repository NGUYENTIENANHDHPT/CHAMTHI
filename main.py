from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Kích thước ảnh chuẩn sau khi nắn thẳng (Tỷ lệ A4)
PAPER_W, PAPER_H = 1000, 1414

def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: Trái-Trên, Phải-Trên, Phải-Dưới, Trái-Dưới"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_image(image):
    """Tìm 4 góc đen và nắn phẳng tờ giấy"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    doc_cnt = None
    
    # Tìm viền lớn nhất có 4 góc
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break
                
    if doc_cnt is not None:
        rect = order_points(doc_cnt.reshape(4, 2))
        dst = np.array([[0, 0], [PAPER_W - 1, 0], [PAPER_W - 1, PAPER_H - 1], [0, PAPER_H - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (PAPER_W, PAPER_H))
        return warped, M
    return cv2.resize(image, (PAPER_W, PAPER_H)), None # Trả về ảnh resize nếu không tìm thấy viền

def read_bubble_grid(warped_gray, x, y, w, h, rows, cols):
    """Cắt vùng lưới (ROI), chia thành các ô và đếm mật độ pixel đen (tô chì)"""
    roi = warped_gray[y:y+h, x:x+w]
    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cell_h, cell_w = h // rows, w // cols
    results = []
    
    for r in range(rows):
        row_res = []
        for c in range(cols):
            cell = thresh[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            # Đếm số pixel đen (đã bị đảo ngược thành trắng)
            total_pixels = cv2.countNonZero(cell)
            row_res.append(total_pixels)
        results.append(row_res)
    return results

@app.post("/grade")
async def grade_paper(image: UploadFile = File(...), all_answers: str = Form(...)):
    try:
        # 1. Đọc ảnh
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Nắn phẳng ảnh
        warped, _ = warp_image(img)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        answer_dict = json.loads(all_answers)
        
        # -------------------------------------------------------------------
        # BƯỚC 3: CẤU HÌNH TỌA ĐỘ CÁC VÙNG (CẦN TINH CHỈNH THỰC TẾ)
        # Các con số (x, y, w, h) dưới đây là ước lượng theo tỷ lệ chuẩn. 
        # Bạn có thể tải bức ảnh đã nắn thẳng về máy tính để đo lại tọa độ chính xác.
        # -------------------------------------------------------------------
        
        # Đọc Mã Đề (Giả sử Mã đề nằm ở x=700, y=100, rộng 150, cao 250, gồm 10 hàng 3 cột)
        # detected_ma_de = "Không rõ"
        # made_grid = read_bubble_grid(warped_gray, x=700, y=100, w=150, h=250, rows=10, cols=3)
        # (Logic đối chiếu mảng made_grid để ra chuỗi "2014"...)
        
        # Tạm thời vẫn lấy mã đề đầu tiên để tránh lỗi không tìm thấy đáp án nếu tọa độ sai
        detected_ma_de = list(answer_dict.keys())[0] if answer_dict else "Không rõ"
        correct_keys = answer_dict.get(detected_ma_de, {})
        
        total_questions = len(correct_keys)
        correct_count = 0
        details = []

        # -------------------------------------------------------------------
        # BƯỚC 4: CHẤM ĐIỂM (CẦN BỔ SUNG TỌA ĐỘ PHẦN I, II, III)
        # -------------------------------------------------------------------
        for q_no, correct_ans in correct_keys.items():
            student_ans = "A" # Tạm giả lập để server chạy được vòng lặp không lỗi
            
            if student_ans == correct_ans:
                correct_count += 1
                details.append(f"Câu {q_no}: ĐÚNG")
            else:
                details.append(f"Câu {q_no}: SAI")

        # 5. Tính điểm
        score = round((correct_count / total_questions) * 10, 2) if total_questions > 0 else 0.0

        # In kết quả lên ảnh đã nắn
        cv2.putText(warped, f"Ma de: {detected_ma_de} | Diem: {score}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # 6. Trả ảnh về App
        _, buffer = cv2.imencode('.jpg', warped)
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
