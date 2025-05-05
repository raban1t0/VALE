# Importar librerias
import cv2
import os

# Importar la clase
import seguimientomanos as sm

# Creacion de carpetas
nombre = 'B'
direccion = 'C:/Users/auror/Desktop/vale.-yolo/vale2'
carpeta = direccion + '/' + nombre

#Si no se ha creado la carpeta
if not os.path.exists(carpeta):
    print("Carpeta creada: ", carpeta)
    os.makedirs(carpeta)

# Lectura de la camara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cont = 0

#Declarar el detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    ret, frame = cap.read()

    # Extraer info de la mano
    frame = detector.encontrarmanos(frame, dibujar = False)

    #Posicion
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum = 0, dibujarPuntos = False, dibujarBox = False, color=[0,255,0])

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
        #recorte = cv2.resize(recorte, (640, 640), Interpolation=cv2.INTER_CUBIC)
        #Almacenar
        cv2.imwrite(carpeta + "VV1{}.jpg".format(cont), recorte)
        cont = cont + 1

        cv2.imshow("RECORTE", recorte)

        #cv2.rectangle(frame,(xmin,ymin),(xmax, ymax),[0,255,0],2)

    cv2.imshow("LENGUA DE SENAS", frame)
    t = cv2.waitKey(1)
    if t == 27 or cont == 5:
        break

cap.release()
cv2.destroyAllWindows()