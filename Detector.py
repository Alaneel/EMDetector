import cv2
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from keras.applications.vgg16 import preprocess_input as vgg_preprocess

# live video
mask_model_type = 'Mobilenet'
mask_dim_dict = {'Normal': (150, 150), 'Mobilenet': (224, 224)}

emotion_model_type = 'Normal'
emotion_dim_dict = {'Normal': (48, 48), 'Mobilenet': (224, 224), 'VGG16': (224, 224)}

classifier = cv2.CascadeClassifier(f'ModelWeights/haarcascade_frontalface_default.xml')
mask_model = load_model(f'ModelWeights/{mask_model_type}_Masks.h5')
emotion_model = load_model(f'ModelWeights/{emotion_model_type}_Emotions.h5')

emotion_dim = emotion_dim_dict[emotion_model_type]
mask_dim = mask_dim_dict[mask_model_type]
emotion_bw = False

mask_dict = {0: 'No Mask', 1: 'Mask'}
mask_dict_color = {0: (0, 0, 255), 1: (0, 255, 0)}
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'disgust', 5: 'fear', 6: 'surprise'}

vid_frames = []
cap = cv2.VideoCapture(0)
cap_video = False

if cap_video == True:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Tests/FaceDetector.mp4', fourcc, 10, (int(width), int(height)))
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1, 1)
    clone = frame.copy()
    bboxes = classifier.detectMultiScale(clone)
    for i in bboxes:
        x, y, width, height = i[0], i[1], i[2], i[3]
        x2, y2 = x + width, y + height
        mask_roi = clone[y:y2, x:x2]
        emotion_roi = mask_roi.copy()
        emotion_roi = cv2.resize(emotion_roi, emotion_dim, interpolation = cv2.INTER_CUBIC)
        mask_roi = cv2.resize(mask_roi, mask_dim, interpolation = cv2.INTER_CUBIC)
        if emotion_bw == True:
            mask_roi = cv2.cvtColor(mask_roi, cv2.COLOR_BGR2GRAY)
            if mask_model_type.upper() != 'Normal':
                mask_roi = np.stack((mask_roi, ) * 3, axis = -1)
            else:
                mask_roi = mask_roi.reshape(mask_roi.shape[0], mask_roi.shape[1], 1)
        # preprocess mask input
        if mask_model_type == 'Mobilenet':
            mask_roi = mobile_preprocess(mask_roi)
        elif mask_model_type == 'Normal':
            mask_roi = mask_roi / 255
        # preprocess emotion input
        if emotion_model_type == 'VGG16':
            emotion_roi = vgg_preprocess(emotion_roi)
        elif emotion_model_type == 'Mobilenet':
            emotion_roi = mobile_preprocess(emotion_roi)
        elif emotion_model_type == 'Normal':
            emotion_roi = emotion_roi / 255
        # resize emotion and mask to feed into nn
        mask_roi = mask_roi.reshape(1, mask_roi.shape[0], mask_roi.shape[1], mask_roi.shape[2])
        emotion_roi = emotion_roi.reshape(1, emotion_roi.shape[0], emotion_roi.shape[1], emotion_roi.shape[2])
        # mask predictions
        mask_predict = mask_model.predict(mask_roi)[0]
        mask_idx = np.argmax(mask_predict)
        mask_conf = f'{round(np.max(mask_predict) * 100)}%'
        mask_cat = mask_dict[mask_idx]
        mask_color = mask_dict_color[mask_idx]
        if mask_idx == 0:
            # emotion predictions
            emotion_predict = emotion_model.predict(emotion_roi)[0]
            emotion_idx = np.argmax(emotion_predict)
            emotion_cat = emotion_dict[emotion_idx]
            emotion_conf = f'{round(np.max(emotion_predict) * 100)}%'
            cv2.putText(clone, f'{mask_cat}: {mask_conf} || {emotion_cat}: {emotion_conf}', (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, .5, mask_color, 2)
            cv2.rectangle(clone, (x, y), (x2, y2), mask_color, 1)
            continue
        cv2.putText(clone, f'{mask_cat}: {mask_conf}', (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, .5, mask_color, 2)
        cv2.rectangle(clone, (x, y), (x2, y2), mask_color, 1)
    
    cv2.imshow('LIVE', clone)
    vid_frames.append(clone)
    if cap_video == True:
        out.write(clone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()

if cap_video == True:
    out.release()

for idx, i in enumerate(vid_frames):
    cv2.imwrite(f'Tests/VidFrames/{idx}.png', i)

img_loc = [22, 52, 74, 103]

images = []
for i in img_loc:
    img = cv2.imread(f'Tests/VidFrames/{i}.png')
    images.append(img)

stack_img = np.hstack(images)
cv2.imwrite('Images/DemoStack.png', stack_img)