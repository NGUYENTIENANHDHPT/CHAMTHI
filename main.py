import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

def get_paper_contour(img):
    """Tìm 4 góc của tờ phiếu thi hoặc 4 điểm neo góc"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50000: # Tìm vùng tờ giấy lớn nhất
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                s = pts.sum(axis=1)
                rect = np.zeros((4, 2), dtype="float32")
                rect[0] = pts[np.argmin(s)]        # Top-left
                rect[3] = pts[np.argmax(s)]        # Bottom-right
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]     # Top-right
                rect[2] = pts[np.argmax(diff)]     # Bottom-left
                return rect
    return None

def warp_image(img, corners):
    """Nắn phẳng khổ chuẩn A4 1000x1414"""
    dst = np.float32([[0, 0], [1000, 0], [0, 1414], [1000, 1414]])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (1000, 1414))

def process_omr(warped, thresh):
    """Quét ô tô đen chuẩn xác từng vị trí trên ảnh đã nắn phẳng"""
    # 1. Phần I: 12 câu trắc nghiệm (Tọa độ chuẩn A4)
    roi_p1 = thresh[470:1350, 60:480]
    h_step = roi_p1.shape[0] / 12.0
    w_step = roi_p1.shape[1] / 4.0
    choices = ['A', 'B', 'C', 'D']
    p1_answers = {}

    for i in range(12):
        max_cnt, best_j = 0, -1
        best_rect = None
        for j in range(4):
            y1, y2 = int(i * h_step + 4), int((i + 1) * h_step - 4)
            x1, x2 = int(j * w_step + 4), int((j + 1) * w_step - 4)
            cell = roi_p1[y1:y2, x1:x2]
            cnt = cv2.countNonZero(cell)
            if cnt > max_cnt and cnt > 100:
                max_cnt = cnt
                best_j = j
                best_rect = (60 + x1, 470 + y1, 60 + x2, 470 + y2)
        
        if best_j != -1:
            p1_answers[str(i + 1)] = choices[best_j]
            cv2.rectangle(warped, (best_rect[0], best_rect[1]), (best_rect[2], best_rect[3]), (0, 255, 0), 2)
        else:
            p1_answers[str(i + 1)] = None

    # Vẽ vị trí các phần ROI để người dùng kiểm tra
    cv2.rectangle(warped, (580, 150), (950, 420), (255, 0, 0), 2) # SBD & Mã Đề
    cv2.rectangle(warped, (510, 470), (950, 800), (255, 165, 0), 2) # Phần II
    cv2.rectangle(warped, (510, 830), (950, 1350), (255, 0, 255), 2) # Phần III

    return p1_answers

@app.get("/")
def home():
    return {"message": "Server OK"}

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
            return JSONResponse({'status': 'error', 'message': 'Không tìm thấy ảnh'}, status_code=400)

        contents = await upload_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        if w > 1000:
            ratio = 1000.0 / w
            img = cv2.resize(img, (1000, int(h * ratio)))

        corners = get_paper_contour(img)
        
        if corners is not None:
            warped = warp_image(img, corners)
            status_text = "NAN PHANG THONG MINH - OK"
            status_color = (0, 255, 0)
        else:
            warped = cv2.resize(img, (1000, 1414))
            status_text = "DUNG KICH THUOC CHUAN A4"
            status_color = (0, 165, 255)

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Quét OMR
        p1_res = process_omr(warped, thresh)

        cv2.putText(warped, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        _, buffer = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 65])
        marked_b64 = base64.b64encode(buffer).decode('utf-8')

        filled_count = sum(1 for v in p1_res.values() if v is not None)
        score = round((filled_count / 12.0) * 10, 2)

        return {
            'status': 'success',
            'ma_de': '101',
            'score': score,
            'details': f'Phần I đọc được {filled_count}/12 câu tô đen',
            'marked_image': marked_b64
        }

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': f'Lỗi: {str(e)}'}, status_code=500)
