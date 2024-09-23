import cv2

# Video kaynağını başlat
cap = cv2.VideoCapture(0)

# Yüz tanıma için CascadeClassifier kullanımı
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# İlk frame'i oku
ret, frame = cap.read()

if not ret:
    print("Video okunamadı.")
    exit()

# Yüz tespiti yap ve koordinatları al
face_rects = face_cascade.detectMultiScale(frame)

if len(face_rects) > 0:
    (face_x, face_y, w, h) = tuple(face_rects[0])
    track_window = (face_x, face_y, w, h)

    # Yüz alanı seçimi
    roi = frame[face_y:face_y + h, face_x:face_x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # MeanShift algoritması için durdurma kriterleri
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # MeanShift algoritması
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('Takip', img2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print("Yüz bulunamadı.")

cap.release()
cv2.destroyAllWindows()
