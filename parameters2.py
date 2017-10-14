COLOR_SPACE='YUV'
HOG_CHANNEL='ALL'
ORIENT=11
PIX_PER_CELL=16
CELL_PER_BLOCK=2
SPATIAL_SIZE=(32, 32)
HIST_BINS=32
BINS_RANGE=(0,256)
YSTART = 400
YSTOP = 650
THRESHOLD=5
DEFAULT_BOX_COLOR=(0,0,255)
DEFAULT_BOX_THICKNESS=6
SPATIAL_FEAT=False
HIST_FEAT=False
HOG_FEAT=True
WINDOWS={}
WINDOWS['x_limit']= [[None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None]]
WINDOWS['y_limit'] = [[464,660], [400,596], [432,560],[400,528],[400,496],[416,480],[400,464]]
WINDOWS['size'] = []#[[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
WINDOWS['overlap'] = [(0.5,0.5), (0.4,0.4), (0.3,0.3), (0.3,0.3), (0.3,0.3), (0.3,0.3), (0.3,0.3)]
WINDOWS['scale'] = [3.5, 3.5, 2.0, 2.0, 1.5, 1.5, 1.0]
NUM_WINDOWS=7
NUM_FRAMES=15
det=None
svc=None

def apply_scales(image):
    imshape = image.shape
    for i in range(NUM_WINDOWS):
        WINDOWS['size'].append((int(imshape[0]/WINDOWS['scale'][i]),int(imshape[1]/WINDOWS['scale'][i])))
    print(WINDOWS)
