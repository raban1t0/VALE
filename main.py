import math
import cv2
import mediapipe as mp
import time

class detectormanos():
    def __init__(self, mode=False, maxManos=2, model_complexity=1, Confdeteccion=0.5, Confsegui=0.5):
        self.mode = mode
        self.maxManos = maxManos
        self.compl = model_complexity
        self.Confdeteccion = Confdeteccion
        self.Confsegui = Confsegui

        self.mpmanos = mp.solutions.hands
        self.manos = self.mpmanos.Hands(self.mode, self.maxManos, self.compl, self.Confdeteccion, self.Confsegui)
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]

    def encontrarmanos(self, frame, dibujar=True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(imgcolor)

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame, mano, self.mpmanos.HAND_CONNECTIONS)
        return frame

    def encontrarposicion(self, frame, dibujarPuntos=False, dibujarBox=False, color=(255, 0, 0)):
        self.listas = []
        self.bboxes = []
        player = 0

        if self.resultados.multi_hand_landmarks:
            player = len(self.resultados.multi_hand_landmarks)
            for miMano in self.resultados.multi_hand_landmarks:
                lista = []
                xlista = []
                ylista = []

                for id, lm in enumerate(miMano.landmark):
                    alto, ancho, c = frame.shape
                    cx, cy = int(lm.x * ancho), int(lm.y * alto)
                    xlista.append(cx)
                    ylista.append(cy)
                    lista.append([id, cx, cy])
                    if dibujarPuntos:
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 0), cv2.FILLED)

                xmin, xmax = min(xlista), max(xlista)
                ymin, ymax = min(ylista), max(ylista)
                bbox = (xmin, ymin, xmax, ymax)
                if dibujarBox:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), color, 2)

                self.listas.append(lista)
                self.bboxes.append(bbox)

        return self.listas, self.bboxes, player

    def dedosarriba(self, lista):
        dedos = []
        if lista[self.tip[0]][1] > lista[self.tip[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)
        for id in range(1, 5):
            if lista[self.tip[id]][2] < lista[self.tip[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)
        return dedos

    def distancia(self, lista, p1, p2, frame, dibujar=True, r=15, t=3):
        x1, y1 = lista[p1][1:]
        x2, y2 = lista[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if dibujar:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]

def main():
    ptiempo = 0
    ctiempo = 0

    cap = cv2.VideoCapture(0)
    detector = detectormanos()

    while True:
        ret, frame = cap.read()
        frame = detector.encontrarmanos(frame)
        listas, bboxes, num_manos = detector.encontrarposicion(frame)

        for i, lista in enumerate(listas):
            if len(lista) != 0:
                dedos = detector.dedosarriba(lista)
                print(f"Mano {i+1}, dedos arriba: {dedos}")

        ctiempo = time.time()
        fps = 1 / (ctiempo - ptiempo)
        ptiempo = ctiempo

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Manos", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
