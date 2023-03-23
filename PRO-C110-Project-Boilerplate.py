

# To Capture Frame
import cv2
import numpy as np

# To Load the Pre-trained Model
import tensorflow as tf

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Loading the pre-trained model : keras_model.h5
model = tf.keras.models.load_model('keras_model.h5')

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)

		# Resize the frame
		resize_frame = cv2.resize(frame , (224,224))

		
		resize_frame = np.expand_dims(resize_frame , axis = 0)
		resize_frame = resize_frame / 255
		predictions = model.predict(resize_frame)

	
		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)

		# printing percentage confidence
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")

		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
