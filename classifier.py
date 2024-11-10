#
# Depends on:
# - tensorflow: pip install tensorflow
# - keras: pip install keras
# - opencv: pip install opencv-python
# - scikit-image: pip install scikit-image
# Windows users may need to use "py -m pip install" or "python -m pip install" instead of "pip install".

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
from skimage.metrics import structural_similarity

MODEL = ResNet50(weights='imagenet')
PATH = "val/n01440764/ILSVRC2012_val_00009111.JPEG"

def classify(img):
  try:
    x = cv2.resize(img, (224,224))
    x = x[:,:,::-1].astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = MODEL.predict(x)
    classes = decode_predictions(preds)[0]
    for c in classes:
      print("\t%s (%s): %.2f%%"%(c[1], c[0], c[2]*100))

  except Exception as e:
    print("Classification failed.")

def open_img(path):
  return cv2.imread(path)

def ssim(img1, img2):
  return structural_similarity(img1, img2, channel_axis=2)*100

def jpeg(img, quality):
  _, x = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
  return cv2.imdecode(x, cv2.IMREAD_COLOR)

def resize(img, w, h):
  orig_h, orig_w = img.shape[:2]
  x = cv2.resize(img, (w,h))
  return cv2.resize(x, (orig_w,orig_h))

def canny(img):
  x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  x = cv2.Canny(x, 100, 200)
  return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

orig_img = open_img(PATH)

print("Original image:")
classify(orig_img)

print("After JPEG q=70%:")
after_jpeg = jpeg(orig_img, 70)
print("SSIM = %.2f"%(ssim(orig_img, after_jpeg)))
classify(after_jpeg)

print("After resizing to 64x64:")
after_resize = resize(orig_img, 64, 64)
print("SSIM = %.2f"%(ssim(orig_img, after_resize)))
classify(after_resize)

print("After Canny edge detection:")
after_canny = canny(orig_img)
print("SSIM = %.2f"%(ssim(orig_img, after_canny)))
classify(after_canny)