from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import glob

# Run inference on the source
model = YOLO(f'../../logs/yolo_v8n_mars2/weights/last.pt')
results = model('/home/tj/data/yoco_journal/mars/mars2/images/val/', stream=True)  # generator of Results objects

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    plt.imshow(im)
    plt.show()

paths = glob.glob('/home/tj/data/yoco_journal/mars/mars2/images/val/*.png')
for path in paths:
    
