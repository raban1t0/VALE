from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza predicción sobre el frame
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    # Dibuja los resultados sobre el frame
    annotated_frame = results[0].plot()

    # Muestra el frame
    cv2.imshow("Camara Jamachulel", annotated_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()
