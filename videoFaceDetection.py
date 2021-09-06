import cv2
import numpy as np
import moviepy.editor
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from IPython.core.debugger import set_trace
from sklearn.svm import LinearSVC
from PIL import Image
from skimage.feature import hog
import matplotlib.pyplot as plt
import multiprocessing
import dlib
import statistics
from scipy.spatial import distance as dist


#### Creates init_frames and grayscales frames every 0.05 seconds.
# userVideo = input("What video file would you like to use?\n")
# vid = cv2.VideoCapture(input)
vid = cv2.VideoCapture("jaijai.mp4")
clip = moviepy.editor.VideoFileClip("jaijai.mp4")
duration = clip.duration

def getFrame(sec):
    vid.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vid.read()
    if hasFrames:
        grayIm = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("init_frames/frame" + str(count) + ".jpg", grayIm)
        cv2.imwrite("init_color_frames/frame" + str(count) + ".jpg", image)
        f = open("face_landmarks/frame" + str(count) + ".txt", "w")
    return hasFrames
sec = 0
frameRate = 0.05
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

#### Gets and plots the HOG image of an image.
# fd, hog_image = hog(cv2.imread("init_frames/frame1.jpg"), orientations=9, pixels_per_cell=(8, 8),
#                     cells_per_block=(2, 2), visualize=True, multichannel=True)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
#
# ax1.imshow(cv2.imread("init_frames/frame1.jpg"), cmap=plt.cm.gray)
# ax2.imshow(hog_image, cmap=plt.cm.gray)
# plt.show()


#### Face Cascade
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Euclidian finds shortest distance between the two points.
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B)/(2.0 * C)
    return ear
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[0], mouth[6])
    mar = (1.0)/(2.0 * A)
    return mar

face_cascade = cv2.CascadeClassifier('/Users/nathanwang/Library/Python/3.8/lib/python/site-packages/cv2/data/haarcascade_frontalface_default.xml')
i = 1
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
xtest = 0
ytest = 0
htest = 0
wtest = 0
while i < count:
    # filename = "init_frames/frame" + str(i) + ".jpg"      # Iterating this way circumvents problems with file sorting and any extra files in the directory.
    # gray = cv2.imread(filename)
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5)
    # for x,y,w,h in faces:
    #     image = cv2.rectangle(gray, (x,y), (x+w, y+h), (0, 255, 0),1)
    #     cv2.imshow("Face Detector", image)
    #     k = cv2.waitKey(1)
    # cv2.imwrite(filename, gray)

    fl = open("face_landmarks/frame" + str(i) + ".txt", "a")
    gr = cv2.imread("init_frames/frame" + str(i) + ".jpg")
    size = gr.shape
    co = cv2.imread("init_color_frames/frame" + str(i) + ".jpg")
    rects = detector(gr,1)
    for (j, rect) in enumerate(rects):
        shape = predictor(gr, rect)
        shape = shape_to_np(shape)
        fl.write(str(shape))
        (x, y, w, h) = rect_to_bb(rect)
        if i == 1:
            xtest = x
            ytest = y
            htest = h
            wtest = w
        cv2.rectangle(gr, (x, y), (x + w, y + h), (0, 255, 0), 1)
        for (x,y) in shape:
            cv2.circle(gr, (x,y), 1, (0,0,255),-1)

        image_points = np.array([
            shape[30],  # Tip of nose
            shape[8],   # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corner
            shape[48],  # Left mouth corner
            shape[54]   # Right mouth corner
        ], dtype = "double")

        model_points = np.array([
            (0,0,0),    # Tip of nose_pts
            (0, -330, -65),     # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        focal_l = size[1]
        center = (size[1]/2, size[0]/2)
        camera = np.array(
            [[focal_l, 0, center[0]],
            [0, focal_l, center[1]],
            [0, 0, 1]], dtype = "double")


        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera, dist_coeffs)

        for p in image_points:
            cv2.circle(gr, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        p3 = (int(shape[27][0]), int(shape[27][1]))
        dy = (int(image_points[0][1]) - int(shape[27][1]))
        dx = (int(image_points[0][0]) - int(shape[27][0]))
        p4 = (int(shape[27][0]) - 5*dx, int(shape[27][1]) - 5*dy)
        cv2.line(gr, p1, p2, (255,0,0), 2)
        cv2.line(gr, p1, p4, (255,255,0), 2)

        hullIndex = cv2.convexHull(shape, returnPoints = False)
        rec = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rec)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)



	#### Mouth open, eyes blink detection. Emoji mirror
        # leftEye = shape[42:48]
        # rightEye = shape[36:42]
        # leftEAR = eye_aspect_ratio(leftEye)
        # rightEAR = eye_aspect_ratio(rightEye)
        # ear = (leftEAR + rightEAR) / 2.0
        #
        # ellipseLen = 8
        # if ear < 0.22:
        #     ellipseLen=2
        #
        # cv2.circle(co, (shape[33][0], shape[33][1]-15), shape[33][0]-shape[4][0]+30, (0, 255, 255), -1)
        # cv2.ellipse(co, (shape[46][0]-7, shape[46][1]), (7, ellipseLen), 0, 0, 360, (0, 0, 0), -1)
        # cv2.ellipse(co, (shape[40][0]-3, shape[40][1]), (7, ellipseLen), 0, 0, 360, (0, 0, 0), -1)
        #
        # mouthShape = shape[48:68]
        # mouth = cv2.convexHull(mouthShape)
        # cv2.drawContours(co, [mouth], -1, (0, 0, 0), 3)
        # mar = mouth_aspect_ratio(shape[48:60])

    cv2.imwrite("edited_frames/frame" + str(i) + ".jpg", co)
    cv2.imwrite("init_frames/frame" + str(i) + ".jpg",gr)
    cv2.imshow("Head Pose", gr)
    cv2.waitKey(10)
    i += 1

#### Face detection and landmarking of cropped face.
xx = 0
yy = 0
hh = 0
ww = 0
crop_left_eye_pts = 0
crop_right_eye_pts = 0
crop_mouth_pts = 0
crop_left_eyebrow_pts = 0
crop_right_eyebrow_pts = 0
crop_nose_pts = 0
crop_jaw_pts = 0

crop = cv2.imread("to_crop.jpg")
graycrop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
size = graycrop.shape
rects = detector(graycrop, 1)
cl = open("cropped_landmarks.txt", "w")
hullIndex = 0
for (i, rect) in enumerate(rects):
    shape = predictor(graycrop, rect)
    shape = shape_to_np(shape)
    crop_left_eye_pts = shape[42:48]
    crop_right_eye_pts = shape[36:42]
    crop_mouth_pts = shape[48:68]
    crop_left_eyebrow_pts = shape[17:22]
    crop_right_eyebrow_pts = shape[22:27]
    crop_nose_pts = shape[27:35]
    crop_jaw_pts = shape[0:17]
    hullIndex = cv2.convexHull(shape, returnPoints = False)
    # rect = cv2.boundingRect(convexhull)
    # subdiv = cv2.Subdiv2D(rect)
    # subdiv.insert(landmarks_points)
    # triangles = subdiv.getTriangleList()
    # triangles = np.array(triangles, dtype=np.int32)

    image_points = np.array([
        shape[30],  # Tip of nose
        shape[8],   # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]   # Right mouth corner
    ], dtype = "double")

    model_points = np.array([
        (0,0,0),    # Tip of nose_pts
        (0, -330, -65),     # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corne
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype = "double")


    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(crop, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    dy = (int(image_points[0][1]) - int(shape[27][1]))
    dx = (int(image_points[0][0]) - int(shape[27][0]))
    p3 = (int(shape[27][0]) - 5*dx, int(shape[27][1]) - 5*dy)
    cv2.line(crop, p1, p2, (255,0,0), 2)
    cv2.line(crop, p1, p3, (255,255,0), 2)

    (x, y, w, h) = rect_to_bb(rect)
    xx = x
    yy = y
    hh = h
    ww = w
    cv2.rectangle(crop, (x,y), (x+w, y+h), (0,255,0),2)
    for (x,y) in shape:
        cv2.circle(crop, (x,y),1,(0,0,255),-1)
        cl.write(str(x) + " " + str(y) + "\n")
cv2.imwrite("crop_face.jpg", crop)
cropped = crop[yy:yy+hh,xx:xx+ww].copy()
cv2.imwrite("cropped.jpg", cropped)
cv2.imshow("Crop", crop)
cv2.waitKey(1000)

#### Transforms cropped face.
def getEyeCenter(eye_landmarks):
    xsum = 0
    ysum = 0
    num_pts = 0
    for point in eye_landmarks:
        xsum += point[0]
        ysum += point[1]
        num_pts += 1
    if num_pts > 0:
        return [xsum/num_pts, ysum/num_pts]
    else:
        return 0

crop_leftEyeCenter = crop_left_eye_pts.mean(axis=0).astype("int")
print(crop_leftEyeCenter)
crop_rightEyeCenter = crop_right_eye_pts.mean(axis=0).astype("int")
print(crop_rightEyeCenter)
crop_dY = crop_rightEyeCenter[1] - crop_leftEyeCenter[1]
crop_dX = crop_rightEyeCenter[0] - crop_leftEyeCenter[0]
crop_eye_dist = np.sqrt((crop_dX ** 2) + (crop_dY ** 2))


def getLandmarks(image, gray, rect):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    landmarks = shape
    left_eye_pts = shape[42:48]
    right_eye_pts = shape[36:42]
    mouth_pts = shape[48:68]
    left_eyebrow_pts = shape[17:22]
    right_eyebrow_pts = shape[22:27]
    nose_pts = shape[27:35]
    jaw_pts = shape[0:17]
    return (landmarks, left_eye_pts, right_eye_pts, mouth_pts, left_eyebrow_pts, right_eyebrow_pts, nose_pts, jaw_pts)

#### TODO: Fix Face Rotation.
# image = cv2.imread("init_color_frames/frame1.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# rects = detector(gray, 1)
# (landmarks, left_eye_pts, right_eye_pts, mouth_pts, left_eyebrow_pts, right_eyebrow_pts, nose_pts, jaw_pts) = getLandmarks(image, gray, rects[0])
# leftEyeCenter = left_eye_pts.mean(axis=0).astype("int")
# print(leftEyeCenter)
# rightEyeCenter = right_eye_pts.mean(axis=0).astype("int")
# print(rightEyeCenter)
# dY = rightEyeCenter[1] - leftEyeCenter[1]
# dX = rightEyeCenter[0] - leftEyeCenter[0]
# angle = np.degrees(np.arctan2(dY, dX))
# print(angle)
# eye_dist = np.sqrt((dX ** 2) + (dY ** 2))


# rightpt = landmarks[0]
# leftpt = landmarks[16]
# dYY = rightpt[1] - leftpt[1]
# dXX = rightpt[0] - leftpt[0]
# face_width = 400
#
# toppt = landmarks[21]
# bottompt = landmarks[8]
# dYYY = toppt[1] - bottompt[1]
# dXXX = toppt[0] - bottompt[0]
# face_height = 400

# desired_dist = eye_dist * face_width

# scale = desired_dist / crop_eye_dist
# crop_angle = np.degrees(np.arctan2(crop_dY, crop_dX)) - 150
#
# cropImage = cv2.imread("to_crop.jpg")
#
# crop_eyes_center = ((crop_leftEyeCenter[0] + crop_rightEyeCenter[0]) // 2, (crop_leftEyeCenter[1] + crop_rightEyeCenter[1]) // 2)
# cv2.imshow("Eye center", cropImage)
# cv2.waitKey(1000)

# A = cv2.getRotationMatrix2D(crop_eyes_center, crop_angle, scale)
# tX = 100
# tY = 50
# A[0, 2] += (tX - crop_eyes_center[0])
# A[1, 2] += (tY - crop_eyes_center[1])
# print(A)

# M = cv2.getRotationMatrix2D(crop_eyes_center,30,1)
# print(M)
# rows,cols,ch = cropImage.shape
#
# output = cv2.warpAffine(cropImage, M, (cols, rows))
# cv2.imwrite("rotate.jpg", output)
# cv2.imshow("Aligned", output)
# cv2.waitKey(1000)
#
# cc = cv2.imread("rotate.jpg")
# gc = cv2.cvtColor(cc, cv2.COLOR_BGR2GRAY)
# rects = detector(gc, 1)
# for (i, rect) in enumerate(rects):
#     shape = predictor(gc, rect)
#     shape = shape_to_np(shape)
#     (x, y, w, h) = rect_to_bb(rect)
#     cv2.rectangle(cc, (x,y), (x+w, y+h), (0,255,0),2)
#     for (x,y) in shape:
#         cv2.circle(cc, (x,y),1,(0,0,255),3)
# cv2.imwrite("rotate.jpg", cc)
# cv2.imshow("rotated", cc)
# cv2.waitKey(1000)

# testblock = test[ytest:ytest+htest, xtest:xtest+wtest].copy()
# test_h, test_w, _ = testblock.shape
# cropped_h, cropped_w, _ = cropped.shape
# fy = test_h / cropped_h
# fx = test_w / cropped_w
# scaledCrop = cv2.resize(cropped, (0,0), fx=fx,fy=fy)
# test[ytest:ytest+htest, xtest:xtest+wtest].copy() = scaledCrop
# cv2.imwrite("cropped.jpg", test)

# file = cv2.imread("crop_face.jpg")
# graycrop = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(graycrop, scaleFactor=1.10, minNeighbors=4)
# rect = cv2.imread("cropped.jpg")
# for x,y,w,h in faces:
#     image = cv2.rectangle(file, (x,y), (x+w, y+h), (0, 255, 0),1)
#     cv2.imshow("Face Detector", image)
#     k = cv2.waitKey(1)
# cv2.imwrite("crop_face.jpg", file)






#### Images to AVI file
frame_array = []
i = 1
while i < count:
    filename = "edited_frames/frame" + str(i) + ".jpg"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
    i += 1

output = cv2.VideoWriter("project.mp4", cv2.VideoWriter_fourcc(*'mp4v'), count/duration, size)

for i in range(len(frame_array)):
    output.write(frame_array[i])
output.release()
