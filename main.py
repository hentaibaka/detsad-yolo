from ultralytics import YOLO
import cv2
import time
import numpy as np
from shapely.geometry import Polygon, LineString
import os
import torch


class CollideDetector:
    @staticmethod
    def checkCollides(polylines1, polylines2):
        out = []
        for p1 in polylines1:
            for p2 in polylines2:
                pgn1 = Polygon(p1)
                pgn2 = Polygon(p2)
                if pgn1.is_valid and pgn2.is_valid and pgn1.intersection(pgn2):
                    int = pgn1.intersection(pgn2)
                    if type(int) == LineString:
                        continue
                    collcoords = int.exterior.coords.xy
                    collcoords = list(zip(collcoords[0], collcoords[1]))
                    out.append([p1, p2, collcoords])
                    print(f"area: {int.area}")
        return out

class ObjectDetector:
    def __init__(self, modelName):
        self._model = YOLO(modelName, task='detect')
        #self._model.export(format='onnx')
        self.names = self._model.names

    def detect(self, frame, classes=None):
        start = time.time()
        results = self._model.predict(frame, save=False, classes=classes, conf=0.55)

        names = results[0].names
        boxes = results[0].boxes
        classes = results[0].boxes.cls
        confidences = results[0].boxes.conf

        boxes = [box.xyxy.to(torch.int32) for box in boxes]
        classes = [names[int(cls)] for cls in classes]

        end = time.time()

        return end - start, boxes, classes, confidences

class Drawer:
    @staticmethod
    def drawPolyline(frame, cls, score, polyline, color):
        label = f"{cls}: {round(float(score), 2)}"
        cv2.polylines(frame, [np.array(polyline, dtype=np.int32)], True, color, 2)
        cv2.putText(frame, label, np.array(polyline[0], dtype=np.int32), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return frame

    @staticmethod
    def drawCollides(collide, color, alpha, frame):
        c = list(map(lambda x: [int(x[0]), int(x[1])], collide[-1][:-1]))
        overlay = frame.copy()
        cv2.rectangle(overlay, c[0], c[2], color, -1)
        new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return new_frame

class DangerZoneHandler:
    DETSAD = None
    _p1 = None
    _p2 = None
    _drawing = False

    @staticmethod
    def twop21l(p1, p2):
        return [p1[0], p1[1], p2[0], p2[1]]

    @staticmethod
    def mouseEventHandler(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            DangerZoneHandler._p1 = (x, y)
            DangerZoneHandler._drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if DangerZoneHandler._drawing:
                DangerZoneHandler._p2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            DangerZoneHandler._p2 = (x, y)

            DangerZoneHandler.DETSAD.addDangerZone(DangerZoneHandler.twop21l(DangerZoneHandler._p1, DangerZoneHandler._p2))

            DangerZoneHandler._p1 = None
            DangerZoneHandler._p2 = None

            DangerZoneHandler._drawing = False
            DangerZoneHandler.DETSAD.currentDangerZone = None

        if DangerZoneHandler._p1 and DangerZoneHandler._p2 and DangerZoneHandler._drawing:
            DangerZoneHandler.DETSAD.currentDangerZone = DangerZoneHandler.twop21l(DangerZoneHandler._p1, DangerZoneHandler._p2)

class DetSad:
    _COLORS = {"blue": (255, 0, 0), "green": (0, 255, 0),
               "red": (0, 0, 255), "white": (255, 255, 255)}

    def __init__(self, capture, modelname, windowname):
        self._capture = capture
        self._objectDetector = ObjectDetector(modelname)
        self._dangerZones = []
        self._windowname = windowname
        DangerZoneHandler.DETSAD = self
        self.currentDangerZone = None
        self.names = self._objectDetector.names
        self.categories = [0]

    def addDangerZone(self, dangerZone):
        self._dangerZones.append(dangerZone)

    def run(self):
        res, frame = self._capture.read()

        time, boxes, classes, confidences = self._objectDetector.detect(frame, classes=self.categories)

        frame, polylines1 = self._drawBoxes(boxes, classes, confidences, frame, self._COLORS['green'])
        frame, polylines2 = self._drawBoxes(self._dangerZones, ['danger'] * len(self._dangerZones),
                                np.zeros(len(self._dangerZones)), frame, self._COLORS['red'])
        if self.currentDangerZone:
            frame, _ = self._drawBoxes([self.currentDangerZone], ['danger'], [0], frame, self._COLORS['blue'])

        collides = CollideDetector.checkCollides(polylines1, polylines2)

        frame = self._drawCollides(collides, self._COLORS['red'], 0.4, frame)

        print(collides)

        cv2.putText(frame, str(round(1 / time, 1)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._COLORS['white'], 1)

        cv2.imshow(self._windowname, frame)

        print(f"FPS: {1 / time}")

    @staticmethod
    def _drawCollides(collides, color, alpha, frame):
        for collide in collides:
            frame = Drawer.drawCollides(collide, color, alpha, frame)
        return frame

    @staticmethod
    def _boxes2Polylines(boxes):
        polylines = []
        for box in boxes:
            if not type(box) == list:
                box = box.tolist()[0]
            polyline = [[box[0], box[1]],
                        [box[0], box[3]],
                        [box[2], box[3]],
                        [box[2], box[1]]]
            polylines.append(polyline)
        return polylines

    def _drawBoxes(self, boxes, classes, scores, frame, color):
        polylines = self._boxes2Polylines(boxes)
        frame = self._drawPolylines(polylines, classes, scores, frame, color)
        return frame, polylines

    def _drawPolylines(self, polylines, classes, scores, frame, color):
        for polyline, cls, score in zip(polylines, classes, scores):
            frame = Drawer.drawPolyline(frame, cls, score, polyline, color)
        return frame

WINDOWNAME = 'detection'

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ds = DetSad(cap, "best.pt", WINDOWNAME)

    cv2.namedWindow(WINDOWNAME)
    cv2.setMouseCallback(WINDOWNAME, DangerZoneHandler.mouseEventHandler)

    while key := cv2.waitKey(1) < 1:
        ds.run()

        if key == ord('q'):
            break

    cv2.destroyAllWindows()