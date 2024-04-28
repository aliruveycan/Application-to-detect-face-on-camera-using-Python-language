import cv2

# OpenCV'nin önceden eğitilmiş yüz tanıma modelini yükleme
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamera yakalama nesnesi oluşturma
cap = cv2.VideoCapture(0)  # 0, yerel kamerayı temsil eder. Farklı bir kamera kullanıyorsanız uygun numarayı girin.

if not cap.isOpened():
    print(
        "Kamera açılamadı. Lütfen kameranın bağlı olduğundan ve başka bir uygulama tarafından kullanılmadığından emin olun.")
    exit()

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()

    if not ret:
        print("Kare alınamadı. Çıkılıyor...")
        break

    # Kareyi griye dönüştür (yüz tanıma algoritması gri tonlamalarda daha iyi çalışır)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algılama
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Algılanan yüzlerin etrafına dikdörtgen çizme
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Sonuçları göster
    cv2.imshow('Face Detection', frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı bırak
cap.release()

# Pencereleri kapat
cv2.destroyAllWindows()
