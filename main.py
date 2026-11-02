import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

def find_anchor_corners(img):
    """Định vị chính xác 4 điểm neo hình vuông đen ở 4 góc phiếu THPT 2025"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < 8000: # Diện tích ô neo góc
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                ar = w / float(h)
                if 0.75 <= ar <= 1.25:
                    centers.append((x + w // 2, y + h // 2))

    if len(centers) >= 4:
        # Lấy 4 điểm xa nhau nhất (về 4 góc)
        pts = np.array(centers, dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]        # Top-Left
        rect[3] = pts[np.argmax(s)]        # Bottom-Right
        rect[1] = pts[np.argmin(diff)]     # Top-Right
        rect[2] = pts[np.argmax(diff)]     # Bottom-Left
        return rect
    return None

def warp_image(img, corners):
    """Nắn phẳng theo 4 điểm neo vuông ở 4 góc"""
    dst = np.float32([[40, 40], [960, 40], [40, 1374], [960, 1374]])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (1000, 1414))

@app.get("/")
def home():
    return {"message": "Server OK"}

@app.post("/grade")
async def grade(
    image: UploadFile = File(None),
    file: UploadFile = File(None)
):
    try:
        upload_file = image or file
        if not upload_file:
            return JSONResponse({'status': 'error', 'message': 'Không tìm thấy ảnh'}, status_code=400)

        contents = await upload_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        if w > 1000:
            ratio = 1000.0 / w
            img = cv2.resize(img, (1000, int(h * ratio)))

        corners = find_anchor_corners(img)
        
        if corners is not None:
            warped = warp_image(img, corners)
            status_text = "NAN PHANG THEO 4 DIEM NEO NEO - OK"
            status_color = (0, 255, 0)
        else:
            warped = cv2.resize(img, (1000, 1414))
            status_text = "KHONG THAY 4 NEO - DUY TRI KICH THUOC"
            status_color = (0, 0, 255)

        # Chuyển nhị phân để quét pixel tô
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Quét Phần I (12 câu)
        roi_p1 = thresh[470:1350, 60:480]
        h_step = roi_p1.shape[0] / 12.0
        w_step = roi_p1.shape[1] / 4.0
        choices = ['A', 'B', 'C', 'D']
        filled_count = 0

        for i in range(12):
            max_cnt, best_rect = 0, None
            for j in range(4):
                y1, y2 = int(i * h_step + 4), int((i + 1) * h_step - 4)
                x1, x2 = int(j * w_step + 4), int((j + 1) * w_step - 4)
                cell = roi_p1[y1:y2, x1:x2]
                cnt = cv2.countNonZero(cell)
                if cnt > max_cnt and cnt > 120:
                    max_cnt = cnt
                    best_rect = (60 + x1, 470 + y1, 60 + x2, 470 + y2)
            
            if best_rect:
                filled_count += 1
                cv2.rectangle(warped, (best_rect[0], best_rect[1]), (best_rect[2], best_rect[3]), (0, 255, 0), 2)

        # Vẽ khung kiểm tra các vùng OMR
        cv2.rectangle(warped, (580, 150), (950, 420), (255, 0, 0), 2)   # Mã đề
        cv2.rectangle(warped, (510, 470), (950, 800), (255, 165, 0), 2) # Phần II
        cv2.rectangle(warped, (510, 830), (950, 1350), (255, 0, 255), 2) # Phần III

        cv2.putText(warped, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        _, buffer = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 65])
        marked_b64 = base64.b64encode(buffer).decode('utf-8')

        score = round((filled_count / 12.0) * 10, 2)

        return {
            'status': 'success',
            'ma_de': '101',
            'score': score,
            'details': f'Đã tô {filled_count}/12 câu Phần I',
            'marked_image': marked_b64
        }

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': f'Lỗi: {str(e)}'}, status_code=500)
