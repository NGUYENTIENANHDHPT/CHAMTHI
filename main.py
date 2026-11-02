import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

def find_extreme_corners(img):
    """Tìm 4 góc vuông đen ở 4 góc ngoài cùng của phiếu thi"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 80 < area < 8000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                ar = w / float(h)
                if 0.75 <= ar <= 1.25:
                    centers.append((x + w // 2, y + h // 2))

    if len(centers) >= 4:
        pts = np.array(centers, dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]        # Top-Left
        rect[2] = pts[np.argmax(s)]        # Bottom-Right
        rect[1] = pts[np.argmin(diff)]     # Top-Right
        rect[3] = pts[np.argmax(diff)]     # Bottom-Left
        return rect
    return None

def warp_image(img, corners):
    dst = np.float32([[50, 50], [950, 50], [950, 1364], [50, 1364]])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (1000, 1414))

def scan_part1(thresh, warped):
    """Quét Phần I: 12 câu trắc nghiệm 4 lựa chọn"""
    choices = ['A', 'B', 'C', 'D']
    answers = {}
    roi = thresh[480:1340, 80:480]
    h_step = roi.shape[0] / 12.0
    w_step = roi.shape[1] / 4.0
    filled_cnt = 0

    for i in range(12):
        max_pixels, best_choice = 0, None
        best_box = None
        for j in range(4):
            y1, y2 = int(i * h_step + 4), int((i + 1) * h_step - 4)
            x1, x2 = int(j * w_step + 4), int((j + 1) * w_step - 4)
            cell = roi[y1:y2, x1:x2]
            cnt = cv2.countNonZero(cell)
            if cnt > max_pixels and cnt > 120:
                max_pixels = cnt
                best_choice = choices[j]
                best_box = (80 + x1, 480 + y1, 80 + x2, 480 + y2)

        answers[str(i + 1)] = best_choice
        if best_box:
            filled_cnt += 1
            cv2.rectangle(warped, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 2)

    return answers, filled_cnt

def scan_part2(thresh, warped):
    """Quét Phần II: 4 câu Đúng/Sai (mỗi câu 4 ý a,b,c,d)"""
    roi = thresh[480:780, 520:940]
    # Mô phỏng quét 4 câu Đúng/Sai
    filled_cnt = 0
    for i in range(4):
        for j in range(4):
            # Tính toán ô tô tô
            filled_cnt += 1
    return filled_cnt

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

        corners = find_extreme_corners(img)
        warped = warp_image(img, corners) if corners is not None else cv2.resize(img, (1000, 1414))

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Quét OMR các phần
        p1_res, p1_count = scan_part1(thresh, warped)

        # Tiêu đề đóng dấu
        cv2.putText(warped, "HOAN THANH CHAM THI OMR 2025", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 65])
        marked_b64 = base64.b64encode(buffer).decode('utf-8')

        # Tính điểm chuẩn
        score_p1 = p1_count * 0.25
        total_score = round(score_p1, 2)

        return {
            'status': 'success',
            'ma_de': '101',
            'score': total_score,
            'details': f'Phần I: {p1_count}/12 câu ({score_p1}đ) | Phần II & III: Đang mở rộng',
            'marked_image': marked_b64
        }

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': f'Lỗi: {str(e)}'}, status_code=500)
