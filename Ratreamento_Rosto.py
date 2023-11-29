import cv2

# Carregar o classificador pré-treinado para detecção de rosto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Abrir o vídeo
video = cv2.VideoCapture('Crowded People Walking Down Oxford Street London 4K UHD Stock Video Footage.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Converter a imagem para escala de cinza (melhora a eficiência da detecção)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Desenhar retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Exibir o resultado
    cv2.imshow('Detecção de Rosto', frame)

    # Sair se a tecla 'Esc' for pressionada
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
