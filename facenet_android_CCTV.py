'''
This script connects to a mobile phone camera stream (e.g., 
via DroidCam http://192.168.1.101:4747/video) and displays the video feed in a window. 
It also allows you to save a snapshot by pressing 's' and exit the 
program by pressing 'q'.
'''

import cv2


def _open_droidcam_stream(base_host: str, port: int = 4747):
    """Try common DroidCam/IP Webcam endpoints with a few capture backends."""
    base = f"http://{base_host}:{port}"
    candidates = [
        f"{base}/video",      # DroidCam HTTP video endpoint
        f"{base}/mjpegfeed",  # Some Android camera apps
        f"{base}/",           # Fallback root endpoint
    ]

    # CAP_ANY works in many setups; CAP_FFMPEG often helps for HTTP streams on Windows.
    backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG]

    for url in candidates:
        for backend in backends:
            cap = cv2.VideoCapture(url, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"已連線: {url} (backend={backend})")
                    return cap
            cap.release()

    return None


# 改成你的手機 IP（手機與電腦需在同一個 Wi-Fi）
PHONE_IP = "192.168.1.110"
PHONE_PORT = 4747
PORTRAIT_MODE = True
ROTATE_CLOCKWISE = True

cap = _open_droidcam_stream(PHONE_IP, PHONE_PORT)

if cap is None:
    print("無法連接到手機攝影機。")
    print("請確認：")
    print("1) 手機 DroidCam 已啟動，且顯示 IP/Port")
    print("2) 電腦與手機在同一個 Wi-Fi")
    print("3) 防火牆允許連到該 IP:Port")
    print("4) 目前程式會嘗試 /video、/mjpegfeed、/")
    raise SystemExit(1)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("無法讀取影像，串流可能中斷。")
        break

    if PORTRAIT_MODE:
        rotate_code = cv2.ROTATE_90_CLOCKWISE if ROTATE_CLOCKWISE else cv2.ROTATE_90_COUNTERCLOCKWISE
        frame = cv2.rotate(frame, rotate_code)

    preview = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Mobile Camera", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        cv2.imwrite("capture_from_phone.jpg", frame)
        print("影像已儲存為 capture_from_phone.jpg")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()