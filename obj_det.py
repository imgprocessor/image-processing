import numpy as np
#import os
import zipfile
#import tarfile
import tensorflow as tf
from skimage.measure import compare_ssim
import imutils

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
#from email.utils import COMMASPACE, formatdate
from email import encoders
#import os
import cv2
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation 
# ## Variables
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 80


# ## Download Model
#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())



# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    cap = cv2.VideoCapture(0)
    t0=time.time()
    count = 0

    
    while(1):
        ret, frame = cap.read()
        if frame is not None: 
            cv2.imwrite("initial.jpg",frame)
            image_np = frame# load_image_into_numpy_array(frame)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.


            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
            min_score_thresh=0.5
            
            print([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index]> min_score_thresh])
            if [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index]> min_score_thresh] ==[{'id': 1, 'name': 'person'}]:
                
                out.write(frame)
                if int(time.time()-t0)%2 == 0:
                    cv2.imwrite("frame%d.jpg" % count, frame)
                    ##cv2.imwrite("face%d.jpg" % count, roi)
                    #print "clicked"
                    time.sleep(1)
                    count += 1
                    print(count)
                #out.write(frame)
                print(num_detections)
            #plt.figure(figsize=IMAGE_SIZE)
            #plt.imshow(image_np)

            cv2.imshow('frame',image_np)
            cv2.imwrite("final.jpg",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    out.release()     
    cap.release()
    cv2.destroyAllWindows()
imageA = cv2.imread('initial1.jpg')
imageB = cv2.imread('final.jpg') 
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
# show the output images
cv2.imwrite("Original.jpg", imageA)
cv2.imwrite("Modified.jpg", imageB)
    
    
fromaddr = "ankmi.ip@gmail.com"
toaddr = "rashmidabir@gmail.com"
 
msg = MIMEMultipart()
 
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "ALERT"
 
body = "Check this"
 
msg.attach(MIMEText(body, 'plain'))

vid_zip = zipfile.ZipFile('output.zip', 'w')
vid_zip.write('output.avi', compress_type=zipfile.ZIP_DEFLATED)
 
vid_zip.close()
 
filename = "output.avi"
attachment = open("C:/Users/User/PycharmProjects/face_detect/output.avi", "rb")
 
part = MIMEBase('application', 'octet-stream')
part.set_payload((attachment).read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
 
msg.attach(part)
 
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, "ankmi.ip2018")
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()


