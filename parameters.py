COLOR_SPACE='GRAY'
HOG_CHANNEL='ALL'
ORIENT=8
PIX_PER_CELL=16
CELL_PER_BLOCK=1
SPATIAL_SIZE=(16, 16)
HIST_BINS=16
BINS_RANGE=(0,1)
YSTART = 400
YSTOP = 650
THRESHOLD=5
DEFAULT_BOX_COLOR=(0,0,255)
DEFAULT_BOX_THICKNESS=6
SPATIAL_FEAT=False
HIST_FEAT=False
HOG_FEAT=True
WINDOWS={}
WINDOWS['x_limit']= [[None, None], [32, None], [412, 1280]]
WINDOWS['y_limit'] = [[400,640], [400,600], [390,540]]
WINDOWS['size'] = [(128,128), (96,96), (80,80)]
WINDOWS['overlap'] = [(0.5,0.5), (0.5,0.5), (0.5,0.5)]
NUM_FRAMES=10
