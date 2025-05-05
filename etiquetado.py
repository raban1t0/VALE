#Importar librerias
import cv2
import os
from ultralytics import YOLO

# Importar la clase
import seguimientomanos as sm

# Lectura de la camara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#Modelo
model = YOLO('valey.pt')

#Declarar el detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    ret, frame = cap.read()

    # Extraer info de la mano
    frame = detector.encontrarmanos(frame, dibujar = False)

    #Posicion
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum = 0, dibujarPuntos = True, dibujarBox = True, color=[0,255,0])

    #Si hay mano
    if mano == 1:
        xmin, ymin, xmax, ymax = bbox

        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        #Recortes
        recorte = frame[ymin:ymax, xmin:xmax]

        # Redimensionar
        #recorte = cv2.resize(recorte, (640, 640), interpolation=cv2.INTER_CUBIC)

        # Extrar resultados
        resultados = model.predict(recorte, conf=0.55)
        if len(resultados) != 0:
            for results in resultados:
                masks = results.masks
                coordenadas = masks

                anotaciones = resultados[0].plot()

        cv2.imshow("RECORTE", anotaciones)

        #cv2.rectangle(frame,(xmin,ymin),(xmax, ymax),[0,255,0],2)

    cv2.imshow("LENGUA DE SENAS", frame)
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()