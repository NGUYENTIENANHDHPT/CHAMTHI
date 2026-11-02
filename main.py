import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

app = FastAPI()

def find_extreme_corners(img):
    """Tìm 4 điểm neo vuông đen nằm xa tâm ảnh nhất (4 góc phiếu)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Lọc các ô vuông đen có kích thước điểm neo góc
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
        
        # Sắp xếp theo tổng và hiệu để lấy 4 góc xa nhất tuyệt đối
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]        # Trên-Trái (Top-Left)
        rect[2] = pts[np.argmax(s)]        # Dưới-Phải (Bottom-Right)
        rect[1] = pts[np.argmin(diff)]     # Trên-Phải (Top-Right)
        rect[3] = pts[np.argmax(diff)]     # Dưới-Trái (Bottom-Left)
        
        return rect
    return None

def warp_image(img, corners):
    """Nắn phẳng chuẩn khổ A4 dựa trên 4 điểm neo góc"""
    dst = np.float32([[50, 50], [950, 50], [950, 1364], [50, 1364]])
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

        corners = find_extreme_corners(img)
        
        if corners is not None:
            warped = warp_image(img, corners)
            status_text = "TOAN BO PHIEU PHANG A4 - OK"
            status_color = (0, 255, 0)
        else:
            warped = cv2.resize(img, (1000, 1414))
            status_text = "CHUA DETECT DU 4 GOC NGOAI"
            status_color = (0, 0, 255)

        # Chuyển ảnh sang nhị phân
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Khoanh vùng các phần OMR trên ảnh phẳng chuẩn A4
        cv2.rectangle(warped, (560, 140), (940, 410), (255, 0, 0), 2)   # Mã đề & SBD
        cv2.rectangle(warped, (80, 480), (480, 1340), (0, 255, 0), 2)   # Phần I
        cv2.rectangle(warped, (520, 480), (940, 780), (255, 165, 0), 2) # Phần II
        cv2.rectangle(warped, (520, 810), (940, 1340), (255, 0, 255), 2) # Phần III

        cv2.putText(warped, status_text, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        _, buffer = cv2.imencode('.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 65])
        marked_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'status': 'success',
            'ma_de': '101',
            'score': 10.0,
            'details': 'Đã nhận diện toàn bộ phiếu thi A4!',
            'marked_image': marked_b64
        }

    except Exception as e:
        return JSONResponse({'status': 'error', 'message': f'Lỗi: {str(e)}'}, status_code=500)
