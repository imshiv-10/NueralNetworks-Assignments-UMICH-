from keras.models import load_model
import numpy as np
import cv2

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
# image = Image.open('test/rock1.png')
size = (224, 224)
classes = ['rock', 'paper', 'scissors']
thresold = 90
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    # get the center of the camera image in a square shape
    h, w, _ = img.shape
    cx = int(w / 2)
    hh = int(h / 2)
    img = img[:, cx - hh:cx + hh]
    img_inference = cv2.resize(img, size)
    img_inference = cv2.cvtColor(img_inference, cv2.COLOR_BGR2RGB)
    # convert

    # Normalize the image
    normalized_image_array = (img_inference.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    class_id = np.argmax(prediction)
    if class_id == 3:
        class_id = 2

    print('The image is {}.'.format(classes[class_id]))
    # index = np.argmax(prediction)
    # class_name = class_names[index]
    # confidence_score = prediction[0][index]
    print("prediction: ", prediction[0])
    # print("Confidence Score:", confidence_score)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)
    # font scale
    fontScale = 1
    # Blue color
    color = (255, 0, 0)
    # Line thickness in BGR
    thickness = 2
    # Using  cv2.putText() method
    img = cv2.putText(img, classes[class_id], org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('show your hand', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
# resize the image to a 224x224 with the same strategy as in TM2:
# resizing the image to be at least 224x224 and then cropping from the center
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
# image_array = np.asarray(image)

# if image_array.shape[2] == 4:
#    image_array = image_array[:, :, 0:-1]
