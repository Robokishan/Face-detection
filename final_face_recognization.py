import glob
import cv2
import Queue
import numpy as np
import os
import math
import sys
from matplotlib import pyplot as plt
import time
import pyttsx
import threading

thread = threading.Thread()
engine = pyttsx.init()
onecount=0
engine.setProperty('rate',150)
threads=[]

# half face detection left side
# xml_name = "haarcascade_profileface.xml"

xml_name = "haarcascade_frontalface_default.xml"
xml = "/usr/local/share/OpenCV/haarcascades/"


# for embedded systems
# xml_name = "lbpcascade_frontalface.xml"
# xml = "/usr/local/share/OpenCV/lbpcascades/"

xml = xml+xml_name

def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT |                     cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else                     cv2.CASCADE_SCALE_IMAGE
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=flags)
        return faces_coord
def check_choice():
    """ Check if choice is good
    """
    is_valid = 0
    while not is_valid:
        try:
            choice = int(raw_input('Enter your choice [1-3] : '))
            if choice in [1, 2, 3]:
                is_valid = 1
            else:
                print "'%d' is not an option.\n" % choice
        except ValueError, error:
            print "%s is not an option.\n" % str(error).split(": ")[1]
    return choice
class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print self.video.isOpened()           ####i commented

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame   

def cut_faces(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    return faces

def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2)
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h),
                              (255, 255, 255), 2)

def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image, 0))
            labels.append(i)
    return (images, np.array(labels), labels_dic)
try:
    images, labels, labels_dic = collect_dataset()
    #rec_eig = cv2.face.createEigenFaceRecognizer()
    #rec_eig.train(images, labels)

    # needs at least two people
    #rec_fisher = cv2.face.cv2.face.BIF_create()
    #rec_fisher.train(images, labels)
    rec_lbph = cv2.face.createLBPHFaceRecognizer()
    rec_lbph.train(images, labels)
    print "Models Trained Succesfully"
    
except Exception as e:
    print e
else:
    pass
finally:
    pass
detector = FaceDetector(xml_path=xml)
# def recognization():

#     frame = webcam.get_frame()
#     faces_coord = detector.detect(frame, False) # detect more than one face
#     if len(faces_coord):
#         faces = normalize_faces(frame, faces_coord) # norm pipeline
#         for i, face in enumerate(faces): # for each detected face
#             collector = cv2.face.MinDistancePredictCollector()
#             rec_lbph.predict(face, collector)
#             conf = collector.getDist()
#             pred = collector.getLabel()
#             threshold = 140
#             print "Prediction: " + labels_dic[pred].capitalize() + "\nConfidence: " + str(round(conf))
#             #aclear_output(wait = True)
#             if conf < threshold: # apply threshold
#                 cv2.putText(frame, labels_dic[pred].capitalize(),
#                             (faces_coord[i][0], faces_coord[i][1] - 10),
#                             cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
#             else:
#                 cv2.putText(frame, "Unknown",
#                             (faces_coord[i][0], faces_coord[i][1]),
#                             cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
#         draw_rectangle(frame, faces_coord) # rectangle around face
#     cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
#                     cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
#     cv2.imshow("White Shadow", frame) # live feed in external
#     if cv2.waitKey(40) & 0xFF == 27:
#         cv2.destroyAllWindows()
def rec():

    count_kishan = 0
    thread = threading.Thread()
    if thread.isAlive() != True:
	    try:
	    	# thread.start_new_thread(speech,("jarvis is up and running",))##not appropriate method of threading
	    	thread = threading.Thread(target=speech,args=("jarvis is up and running",))
	    	thread.start()
	    except Exception as e:
	    	raise

    webcam = VideoCamera(0)
    count = 0
    cv2.namedWindow("White Shadow", cv2.WINDOW_AUTOSIZE)
    while True:
        frame = webcam.get_frame()
        faces_coord = detector.detect(frame, False) # detect more than one face
        if len(faces_coord):
            faces = normalize_faces(frame, faces_coord) # norm pipeline
            for i, face in enumerate(faces): # for each detected face
                collector = cv2.face.StandardCollector_create(0.6)

                pred, conf = rec_lbph.predict(face)

                # conf = collector.getMinDist()
                # pred = collector.getMinLabel()




                # conf = 140
                # pred = 1


                print ("conf => ",conf)
                print("pred =>",pred)


                threshold = 145
                # print "Prediction: " + labels_dic[pred].capitalize() + "\nConfidence: " + str(round(conf))
                nameofperson=labels_dic[pred]
                # print nameofperson

                # opencv 3 is not compatible to this code yet
                if conf < threshold: # apply threshold
                    cv2.putText(frame, labels_dic[pred].capitalize(),
                                (faces_coord[i][0], faces_coord[i][1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

                    # folder = "people/" + labels_dic[pred].lower() # input name
                    # files = folder+"/*"
                    # list_of_files = glob.glob(files) # * means all if need specific format then *.csv
                    # latest_file = max(list_of_files, key=os.path.getctime)
                    # # base = os.path.basename(latest_file)
                    # base = "1.jpg"
                    # count = os.path.splitext(base)[0]
                    # counter = int(count)
                    # print "counts: ",count,"  ",counter
                    # counter+=1
                    # cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0])
                else:
                    cv2.putText(frame, "Unknown",
                                (faces_coord[i][0], faces_coord[i][1]),
                                cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)



                # print engine.isBusy()


                # if engine.isBusy() == False:
                #     if nameofperson != "person":
                #         try:
                #        	    thread.start_new_thread(speech,(nameofperson,))
                #         except:
                #             pass
                #         finally:
                #             pass
                #     if nameofperson == "kishan":
                #         try:
                #             thread.start_new_thread(speech,("hello sir how are you",))
                #         except:
                #             pass
                #         finally:
                #             pass

                #         try:
                #             thread.start_new_thread(speech,("i am your new personal assistant",))
                #         except:
                #             pass
                #         finally:
                #             pass
                # print count_kishan
                # print thread.isAlive()
            draw_rectangle(frame, faces_coord) # rectangle around face
        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
        cv2.imshow("White Shadow", frame) # live feed in external
        if cv2.waitKey(40) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    del webcam
def building_data_set():
    webcam = cv2.VideoCapture(0)
    _, frame = webcam.read()
    webcam.release()
    #plt_show(frame)

    detector = cv2.CascadeClassifier(xml)

    scale_factor = 1.2
    min_neighbors = 5
    min_size = (30, 30)
    biggest_only = True
    flags = cv2.CASCADE_FIND_BIGGEST_OBJECT |             cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else             cv2.CASCADE_SCALE_IMAGE

    faces_coord = detector.detectMultiScale(frame,
                                            scaleFactor=scale_factor,
                                            minNeighbors=min_neighbors,
                                            minSize=min_size,
                                            flags=flags)
    # print "Type: " + str(type(faces_coord))
    # print faces_coord
    # print "Length: " + str(len(faces_coord))





    webcam = VideoCamera()
    detector = FaceDetector(xml)




    folder = "people/" + raw_input('Person: ').lower() # input name
    cv2.namedWindow("White Shadow")
    if not os.path.exists(folder):
        os.mkdir(folder)
        counter = 1
        timer = 0
        while counter < 31 : # take 20 pictures
            frame = webcam.get_frame()
            faces_coord = detector.detect(frame) # detect
            if len(faces_coord) and timer % 700 == 50: # every Second or so
                faces = normalize_faces(frame, faces_coord) # norm pipeline
                cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0])
                # plt_show(faces[0], "Images Saved:" + str(counter))
                #clear_output(wait = True) # saved face in notebook
                counter += 1
                print counter
            draw_rectangle(frame, faces_coord) # rectangle around face
            cv2.imshow("White Shadow", frame) # live feed in external
            cv2.waitKey(50)
            timer += 50
        cv2.destroyAllWindows()
    else:
        print "This name already exists."

    del webcam

####################speech
def speech(say):
    engine.say(say)
    # engine.iterate(say)
    try:
        engine.runAndWait()
       	# engine.stop()
    except Exception as e:
        pass
###############################speeecch complete

def main():

	print """
------------------------------
   POSSIBLE ACTIONS
------------------------------
1. Add person to the recognizer system
2. Start recognizer(lbph recognization model)
3. Exit
------------------------------


"""
	CHOICE = check_choice()
	if CHOICE == 1:
		building_data_set()
	elif CHOICE == 2:
		rec()
	elif CHOICE == 3:
		sys.exit()
# building_data_set()
main()
