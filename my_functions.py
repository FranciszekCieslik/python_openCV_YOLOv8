import sys
import numpy as np
import cv2 as cv
import my_class as myc
import face_recognition

def img_load(imgpath, scale:float = 1):
    try:
        img = cv.imread(cv.samples.findFile(imgpath))
        img = cv.resize(img, None, fx=scale, fy=scale)
        #(h, w,c) = img.shape[:3]
        #print("width: {} px".format(w))
        #print("hight: {} px".format(h))
        #print("num of channels: {}".format(c))
        return img
    except:
        sys.exit("\nCould not read the image.")

def edges_detection(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blurred, 50, 150)
    return edges

def webcam(camera:int = 0, func=None):
    cap = cv.VideoCapture(camera)
    while True:
        frame = cap.read()[1]
        if func== None:
            cv.imshow("Webcam", frame)
        else:
            cv.imshow("Webcam", func(frame))
        cv.waitKey(1)

def thresholding(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray,5)
    thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    return thresh

def object_detection(img, conf:float=0.5):
    nimg = img.copy()
    yd = myc.YOLOdetection("yolov8n.pt","v8")
    boxes, class_list = yd.detect(nimg, conf)
    for i, box in enumerate(boxes):
        color = np.random.uniform(0, 200, 3)
        clsID, conf, bb = box.cls.numpy()[0], box.conf.numpy()[0], box.xyxy.numpy()[0]
        x, y, x2, y2 = list(map(int, bb))
        label = f"{class_list[int(clsID)]} {conf:.1%}"
        font = cv.FONT_HERSHEY_PLAIN
        font_scale = 1
        thickness = 2
        txtsize, _ = cv.getTextSize(label, font, font_scale, thickness)
        txtsize = list(txtsize)
        cv.rectangle(nimg, (x, y), (x2, y2), color, 2)
        cv.rectangle(nimg,(x - 1, y - txtsize[1]*2 - 5),(x + txtsize[0], y), color, -1)
        cv.putText(nimg, label, (x, y - txtsize[1]), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    return nimg

def clustering(img, K:int=8):
    nimg = cv.GaussianBlur(img.copy(), (5,5), 0)
    Z = nimg.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)[1:]
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((nimg.shape))
    return res2

def img_segmentation(img, conf:float=0.5):
    nimg = img.copy()
    ys = myc.YOLOsegmentation("yolov8n-seg.pt")
    try:
        clas, segmentation, score = ys.detect(nimg, conf)[1:]
    except:
        return clustering(nimg)

    for clas, seg, score in zip(clas, segmentation, score):
        if(score >= conf):
            color = np.random.uniform(0, 180, 3)
            cv.fillPoly(nimg, [seg], color)
            alpha = 0.6
            nimg = cv.addWeighted(nimg, alpha, clustering(img), 1 - alpha, 0)
            cv.polylines(nimg, [seg], True, color, 2)
    return nimg

def face_detection(imgpath):
    img = img_load(imgpath)
    image = face_recognition.load_image_file(imgpath)
    face_locations = face_recognition.face_locations(image)
    for face in face_locations:
        face = list(face)
        y = int(face[0])
        x = int(face[1])
        y2 = int(face[2])
        x2 = int(face[3])
        cv.rectangle(img,(x, y), (x2, y2),(255,255,0),2)
    return img

def img_info(imgpath):
    img = img_load(imgpath)
    image = face_recognition.load_image_file(imgpath)
    face_locations = face_recognition.face_locations(image)
    facenum = len(face_locations)
    yd = myc.YOLOdetection("yolov8n.pt","v8")
    boxes, class_list = yd.detect(img)
    elements = []
    for i, box in enumerate(boxes):
        clsID = box.cls.numpy()[0]
        elements.append(class_list[int(clsID)])
    elements.sort()
    res = {}
    for i in elements:
       res[i] = elements.count(i)
    return facenum, res



        
    



    










