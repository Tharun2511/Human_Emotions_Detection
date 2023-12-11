import onnxruntime as rt
import cv2
import time
import numpy as np
import service.main as s

def emotions_detector(img_array):

    if(len(img_array)==2):
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    time_init = time.time()

    test_image = cv2.resize(img_array, (256, 256))
    im = np.float32(test_image)
    img_array = np.expand_dims(im, axis = 0)
    time_taken_preprocess = time.time()-time_init

    onnx_pred = s.m_q.run(['dense_11'], {'input': img_array})

    time_taken = time.time() - time_init

    emotion = ""
    if np.argmax(onnx_pred[0][0])==0:
        emotion = "angry"
    elif np.argmax(onnx_pred[0][0])==1:
        emotion = "happy"
    else:
        emotion = "sad"

    return {"emotion" : emotion,
            "time_taken" : str(time_taken),
            "time_taken_preprocess": str(time_taken_preprocess)
            }