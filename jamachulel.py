import cv2
import os
import main as sm

# Configuración
nombre = 'Jamachulel'
direccion = 'C:/Users/auror/Desktop/jamachulel'
carpeta = os.path.join(direccion, nombre)

if not os.path.exists(carpeta):
    print("Carpeta creada:", carpeta)
    os.makedirs(carpeta)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cont = 0
limite_fotos = 58
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detector.encontrarmanos(frame, dibujar=False)
    listas, bboxes, total = detector.encontrarposicion(frame)

    if total > 0 and cont < limite_fotos:
        # Unir todas las cajas en una sola envolvente
        xmins = [bbox[0] for bbox in bboxes]
        ymins = [bbox[1] for bbox in bboxes]
        xmaxs = [bbox[2] for bbox in bboxes]
        ymaxs = [bbox[3] for bbox in bboxes]

        xmin = max(0, min(xmins) - 40)
        ymin = max(0, min(ymins) - 40)
        xmax = min(frame.shape[1], max(xmaxs) + 40)
        ymax = min(frame.shape[0], max(ymaxs) + 40)

        recorte = frame[ymin:ymax, xmin:xmax]

        # Guardar imagen combinada
        filename = os.path.join(carpeta, f"Jama_{cont}.jpg")
        cv2.imwrite(filename, recorte)
        cont += 1

        cv2.imshow("RECORTE", recorte)

    cv2.imshow("LENGUA DE SEÑAS", frame)

    t = cv2.waitKey(1)
    if t == 27 or cont >= limite_fotos:
        break

cap.release()
cv2.destroyAllWindows()
