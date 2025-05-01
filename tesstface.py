from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition, 'damo/cv_resnet34_face-attribute-recognition_fairface')
src_img_path = 'data/test/images/face_recognition_1.png'
raw_result = fair_face_attribute_func(src_img_path)
print('face attribute output: {}.'.format(raw_result))

# if you want to show the result, you can run
from modelscope.utils.cv.image_utils import draw_face_attribute_result
from modelscope.preprocessors.image import LoadImage
import cv2
import numpy as np

# load image from url as rgb order
src_img = LoadImage.convert_to_ndarray(src_img_path)
# save src image as bgr order to local
src_img  = cv2.cvtColor(np.asarray(src_img), cv2.COLOR_RGB2BGR)
cv2.imwrite('src_img.jpg', src_img) 
# draw dst image from local src image as bgr order
dst_img = draw_face_attribute_result('src_img.jpg', raw_result)
# save dst image as bgr order to local
cv2.imwrite('dst_img.jpg', dst_img)
# show dst image by rgb order
import matplotlib.pyplot as plt
dst_img  = cv2.cvtColor(np.asarray(dst_img), cv2.COLOR_BGR2RGB)
plt.imshow(dst_img)