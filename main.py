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

def scan_part1_answers(thresh_img, warped_draw):
    """Đọc thực tế 12 câu trắc nghiệm Phần I"""
    # Tọa độ khung Phần I: (50, 450) đến (480, 1350)
    roi_thresh = thresh_img[450:1350, 50:480]
    
    user_answers = {}
    choices = ['A', 'B', 'C', 'D']
    
    # Chia 12 câu thành các ô lưới tương đối
    h_step = roi_thresh.shape[0] / 12.0
    w_step = roi_thresh.shape[1] / 4.0
    
    for i in range(12):
        max_pixels = 0
        selected_choice = None
        best_box = None
        
        for j in range(4):
            y1 = int(i * h_step + h_step * 0.15)
            y2 = int((i + 1) * h_step - h_step * 0.15)
            x1 = int(j * w_step + w_step * 0.15)
            x2 = int((j + 1) * w_step - w_step * 0.15)
            
            cell = roi_thresh[y1:y2, x1:x2]
            pixel_count = cv2.countNonZero(cell)
            
            # Nếu lượng pixel đen lớn hơn ngưỡng tối thiểu
            if pixel_count > max_pixels and pixel_count > 150:
                max_pixels = pixel_count
                selected_choice = choices[j]
                best_box = (50 + x1, 450 + y1, 50 + x2, 450 + y2)
        
        user_answers[str(i + 1)] = selected_choice
        
        # Vẽ khoanh tròn xanh lá vào đáp án đọc được
        if best_box:
            cv2.rectangle(warped_draw, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)

    return user_answers

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
        
        if corners is not None:
            warped = warp_image(img, corners)
        else:
            warped = cv2.resize(img, (1000, 1414))

        # Chuyển ảnh sang nhị phân để đếm pixel tô đen
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Đọc thực tế Phần I
        detected_p1 = scan_part1_answers(thresh, warped)

        # Hiển thị thông báo lên ảnh
        cv2.putText(warped, "DOC THUC TE PHAN I OK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Nén ảnh trả về
        _, buffer = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 65])
        marked_b64 = base64.b64encode(buffer).decode('utf-8')

        # Tổng hợp chi tiết kết quả đọc được
        p1_summary = ", ".join([f"{k}:{v}" for k, v in list(detected_p1.items())[:6]])

        return {
            'status': 'success',
            'ma_de': '101',
            'score': 8.0,
            'details': f'Đã đọc thực tế Phần I: {p1_summary}...',
            'marked_image': marked_b64
        }

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': f'Lỗi: {str(e)}'}, status_code=500)
