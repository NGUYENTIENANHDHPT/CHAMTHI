import cv2
import numpy as np
import base64
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

def find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if 0.7 <= float(w) / h <= 1.3:
                    boxes.append((x + w // 2, y + h // 2))

    if len(boxes) >= 4:
        pts = np.array(boxes, dtype="float32")
        s = pts.sum(axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]
        rect[3] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[2] = pts[np.argmax(diff)]
        return rect
    return None

def warp_image(img, corners):
    dst = np.float32([[0, 0], [1000, 0], [0, 1414], [1000, 1414]])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (1000, 1414))

def scan_part1(thresh, warped):
    """Đọc 12 câu trắc nghiệm Phần I"""
    choices = ['A', 'B', 'C', 'D']
    answers = {}
    roi = thresh[450:1350, 50:480]
    h_step = roi.shape[0] / 12.0
    w_step = roi.shape[1] / 4.0

    for i in range(12):
        max_pixels, sel = 0, None
        best_box = None
        for j in range(4):
            y1, y2 = int(i * h_step + h_step * 0.1), int((i + 1) * h_step - h_step * 0.1)
            x1, x2 = int(j * w_step + w_step * 0.1), int((j + 1) * w_step - w_step * 0.1)
            cell = roi[y1:y2, x1:x2]
            cnt = cv2.countNonZero(cell)
            if cnt > max_pixels and cnt > 120:
                max_pixels, sel = cnt, choices[j]
                best_box = (50 + x1, 450 + y1, 50 + x2, 450 + y2)
        answers[str(i + 1)] = sel
        if best_box:
            cv2.rectangle(warped, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)
    return answers

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

        corners = find_corners(img)
        warped = warp_image(img, corners) if corners is not None else cv2.resize(img, (1000, 1414))

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Quét thực tế Phần I
        p1_res = scan_part1(thresh, warped)

        # Đánh dấu tiêu đề thành công
        cv2.putText(warped, "CHAM THI OMR 2025 HOAN THANH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Nén ảnh trả về
        _, buffer = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 65])
        marked_b64 = base64.b64encode(buffer).decode('utf-8')

        # Tính điểm mô phỏng dựa trên đáp án thực tế đọc được
        correct_count = sum(1 for k, v in p1_res.items() if v is not None)
        score = round((correct_count / 12.0) * 10, 2)

        return {
            'status': 'success',
            'ma_de': '101',
            'score': score,
            'details': f'Phần I (Đã tô {correct_count}/12 câu)',
            'marked_image': marked_b64
        }

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': f'Lỗi: {str(e)}'}, status_code=500)
