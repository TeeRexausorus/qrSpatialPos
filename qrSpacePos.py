import cv2
import cv2.cv as cv
from scipy import ndimage
import numpy as np
import zbar


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    pts = np.asarray(pts)
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

#Useless, only here to test the perspective...
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    #print rect
    #print dst
    M = cv2.getPerspectiveTransform(rect, dst)
    print M
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

#returns the size of a segment
def get_size(point1, point2):
    a = np.array(point1)
    b = np.array(point2)
    return np.linalg.norm(a-b)

#returns the distance from the QR-code to the camera
def get_distance(point1, point2):
    dPrime = get_size(point1,point2)
    D = 5.7 #taille en cm de l'image etalon
    d = 197 #taille en px de l'image etalon
    Z = 15  #distance en cm entre l'image etalon et la webcam
    f = d*Z/D #calcul de la focale
    DPrime = dPrime * D / d
    Zprime = D * f / dPrime
    return Zprime

#not used
def milieu(point1, point2):
    return (point1[0] + point2[0] / 2, point1[1] + point2[1] / 2)

#returns a rough guess of the orientation of the camera
def guess_orientationH(sizeLine1, sizeLine2):
    if sizeLine1 > sizeLine2:
        return "gauche"
    else:
        return "droite"

#returns a rough guess of the orientation of the camera
def guess_orientationV(sizeLine1, sizeLine2):
    if sizeLine1 > sizeLine2:
        return "bas"
    else:
        return "haut"


class CameraDist():
    font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        cv.NamedWindow("w1", cv.CV_WINDOW_NORMAL)
        self.capture = cv.CreateCameraCapture(-1)
        self.vid_contour_selection()

    def guess_global_orientation(self, frame, sizes):
        cv2.putText(frame, "la camera est sur la {} et en {} du QR Code".format(guess_orientationH(sizes[0], sizes[2]), guess_orientationV(sizes[1], sizes[3])), (0,80), self.font, 0.6, (255,122,0), 2)

    #Al-Kashi FTW \o/
    def getInternalAngle(self, B, A, C):
        a = get_size(B, C)
        b = get_size(A, C)
        c = get_size(A, B)
        cosA = (b*b + c*c - a*a)/(2*b*c)
        return np.degrees(np.arccos(cosA))
    
    def vid_contour_selection(self):
        while True:
            self.frame = cv.QueryFrame(self.capture)
            frame = np.asarray(self.frame[:,:])
            g = cv.fromarray(frame)
            g = np.asarray(g)
            imgray = cv2.cvtColor(g,cv2.COLOR_BGR2GRAY)
            raw = str(imgray.data)
            scanner = zbar.ImageScanner()
            scanner.parse_config('enable')
            imageZbar = zbar.Image( self.frame.width, self.frame.height,'Y800', raw)
            scanner.scan(imageZbar)
            points = []
            sizes = []
            angles = []
            for symbol in imageZbar:
                #print symbol.location
                points = []
                sizes = []
                angles = []
                for point in symbol.location:
                    points.append((point[0],point[1]))
                for i in range(1, len(points)):
                    #cv2.putText(frame, "" + str(self.getAngle(points[i],points[(i+1)%len(points)],points[(i+2)%len(points)])),points[i-1], self.font, 1,(0, 0, 255), 2)
                    cv2.line(frame, points[i-1],points[i], (0, 0, 255))
                    sizes.append(get_size(points[i-1],points[i]))
                    angles.append(self.getInternalAngle(points[i],points[(i+1)%len(points)],points[(i+2)%len(points)]))
                    #cv2.putText(frame, "" + str(self.getAngle(points[i],points[(i+1)%len(points)],points[(i+2)%len(points)])),points[i], self.font, 1, (0, 0, 255), 2)
                angles.append(self.getInternalAngle(points[i], points[(i+1)%len(points)], points[(i+2)%len(points)]))
                cv2.putText(frame, symbol.data, (0,60), self.font, 1, (255,0,0), 2)
                cv2.line(frame, points[i], points[0], (0,0,255))
                sizes.append(get_size(points[i], points[0]))
                dist = np.mean([get_distance(points[0], points[1]), get_distance(points[1], points[2]), get_distance(points[2], points[3]), get_distance(points[3], points[0])])
                cv2.putText(frame, str(round(dist, 1)) + " cm", (0,30), self.font, 1,(255,255,0), 2)
                self.guess_global_orientation(frame, sizes)
                #cv2.putText(frame, "la camera est sur la {} et en {} du QR Code".format(guess_orientationH(sizes[0], sizes[2]), guess_orientationV(sizes[1], sizes[3])), (0,80), self.font, 0.6, (255,122,0), 2)
                #print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data
                four_point_transform(frame, points)
            cv2.imshow("w1", frame)
            c = cv.WaitKey(5)
        if c == 110: #pressing the 'n' key will cause the program to exit
            exit()

p = CameraDist()
