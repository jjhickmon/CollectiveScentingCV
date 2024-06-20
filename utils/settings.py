# Default bee label colors in BGR color space
# COLORS = [(73, 0, 230), (255, 180, 11), (145, 233, 80), (0, 216, 230),
#           (245, 25, 155), (0, 163, 255), (180, 10, 220), (255, 212, 179), (160, 191, 0)]
# COLORS = [(0, 255, 0), (70, 150, 250), (250, 80, 160)]
# COLORS = [(181, 144, 87), (188,194,132), (164,213,159), (142,229,189), (109,245,253), (72,213,247)]
COLORS = [(181, 144, 87), (188,194,132), (164,213,159), (142,229,189), (72,213,247)]

# define the locations of unwanted areas in the frame like the queen cage, etc.
ARTIFACT_LOCATIONS = [(0, 0, 10, 2050), (0, 1050, 90, 40), (70, 30, 90, 390), (850, 75, 60, 50)]
ARTIFACT_LOCATIONS = [(loc[0] + 420, loc[1], loc[2], loc[3]) for loc in ARTIFACT_LOCATIONS]


VIDEO_NAME = ""
BACKGROUND_NAME = ""
ALLOW_MANUAL_LABELLING = False
ALLOW_MANUAL_SEGMENTING = True
MAX_AUTOMATIC_ITERATIONS = 15
WINDOW_NAME = "frame"
MANUAL_WINDOW_NAME = "manual point select - press ENTER to continue"
FRAMES_PATH = "denoised_frames"

# Settings for defining bee contours
MIN_BEE_AREA = 0 # NOTE: make sure to modify these values depending on the test
MAX_BEE_AREA = 20000
MIN_GROUP_AREA = 100
MAX_GROUP_AREA = 50000
MAX_MOVE_DISTANCE = 100
MAX_THRESH_COLOR_DIFF = 60

LOAD_PREPROCESS_SETTINGS = True # NOTE: make sure to modify this if you want to load the previously settings or create new ones
DEBUG = True # typically leave this on to print helpful debugging messages