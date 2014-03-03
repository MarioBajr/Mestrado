__author__ = 'Mario'


#: The path to the opencv install's Haar cascade folder
HCDIR = '/opt/local/share/OpenCV/haarcascades/'

#: The name of the eyepair Haar cascade file to use
HC_EYEPAIR_NAME = 'haarcascade_mcs_eyepair_big.xml'

#: The name of the left eye Haar cascade file to use
HC_LEFTEYE_NAME = 'haarcascade_lefteye_2splits.xml'

#: The name of the right eye Haar cascade file to use
HC_RIGHTEYE_NAME = 'haarcascade_righteye_2splits.xml'

#: The name of the face Haar cascade file to use
HC_FACE_NAME = 'haarcascade_frontalface_alt2.xml'



#: For Debug purpose we load only a few samples from the database
QDT_DEBUG_LOAD = 10

#: Quantity of SIFT features
QDT_SIFT_FEATURES = 20


# Face characteristics, may need to be tweaked per face

#: An eyepair is probably valid with this width/height ratio
EYEPAIR_RATIO = 2

#: Conversion factor between the width of an eyepair and the width between eyes,
# used when valid eyes are not detected and the eyepair is used to calculate scaling factors
EYEPAIR_WIDTH_TO_EYE_WIDTH = .6

#: The minimum distance threshold for left/right eye. Usually just necessary to
# ensure that detected left/right eyes w/o eyepair are not the same eye
EYE_MIN_DISTANCE = .05

#: Conversion factor from the height of a detected face to the eyes midpoint.
# Used when falling back on face detection from eye detection
FACE_HEIGHT_TO_EYE_MID = .4

#: Conversion factor from the width of a face to the eye width.
# Used when falling back on face detection from eye detection
FACE_WIDTH_TO_EYE_WIDTH = .41

#: The minimum size detection threshold for eyepair as a fraction of the image size
EYEPAIR_MIN_SIZE = (.15, .03)

#: The maximum size detection threshold for eyepair as a fraction of the image size
EYEPAIR_MAX_SIZE = (.55, 1)