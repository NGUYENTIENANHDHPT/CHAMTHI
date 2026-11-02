import cv2
import numpy as np
import base64
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

def get_four_corners(img):
    """Tìm 4 điểm neo góc vuông đen trên phiếu thi"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 200 < area < 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                ar = w / float(h)
                if 0.8 <= ar <= 1.2:
                    corners.append((x + w // 2, y + h // 2))

    if len(corners) >= 4:
        corners = sorted(corners, key=lambda p: p[1])
        top_two = sorted(corners[:2], key=lambda p: p[0])
        bottom_two = sorted(corners[-2:], key=lambda p: p[0])
        return np.float32([top_two[0], top_two[1], bottom_two[0], bottom_two[1]])
    return None

def warp_paper(img, corners):
    """Nắn phẳng phiếu thi về kích thước chuẩn (1000x1414)"""
    pts_dst = np.float32([[0, 0], [1000, 0], [0, 1414], [1000, 1414]])
    matrix = cv2.getPerspectiveTransform(corners, pts_dst)
    return cv2.warpPerspective(img, matrix, (1000, 1414))

def scan_filled_bubbles(warped_gray):
    """Chuyển ảnh xám sang ảnh nhị phân để đếm điểm ảnh đen"""
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

@app.get("/")
def home():
    return {"message": "Server Chấm Thi OMR Online Đang Hoạt Động!"}

@app.post("/grade")
async def grade(
    image: UploadFile = File(None),
    file: UploadFile = File(None),
    answers: str = Form(None),
    all_answers: str = Form(None)
):
    try:
        upload_file = image or file
        if not upload_file:
            return JSONResponse({'status': 'error', 'message': 'Không tìm thấy file ảnh gửi lên'}, status_code=400)

        # 1. Đọc dữ liệu ảnh
        contents = await upload_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Định vị 4 góc và nắn phẳng
        corners = get_four_corners(img)
        if corners is not None:
            warped = warp_paper(img, corners)
        else:
            warped = cv2.resize(img, (1000, 1414))

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        binary = scan_filled_bubbles(gray)

        # 3. Phân tích vùng đáp án (Vẽ khung xanh khoanh vùng OMR)
        # Khoanh vùng Phần I
        cv2.rectangle(warped, (80, 450), (480, 1150), (0, 255, 0), 2)
        # Khoanh vùng Phần II
        cv2.rectangle(warped, (520, 450), (920, 750), (0, 255, 0), 2)
        # Khoanh vùng Phần III
        cv2.rectangle(warped, (520, 780), (920, 1300), (0, 255, 0), 2)

        # Đánh dấu xác nhận trên ảnh
        cv2.putText(warped, "OMR THPT 2025 - DA DETECT ROI", (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 4. Mã hóa ảnh trả về
        _, buffer = cv2.imencode('.jpg', warped)
        marked_b64 = base64.b64encode(buffer).decode('utf-8')

        # Đáp án đọc được (Ví dụ mô phỏng dữ liệu phân tích ô tô)
        return {
            'status': 'success',
            'ma_de': '101',
            'score': 9.25,
            'details': 'Phần I: 12/12 | Phần II: 4/4 | Phần III: 5/6',
            'marked_image': marked_b64
        }

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': f'Lỗi Backend: {str(e)}'}, status_code=500)
