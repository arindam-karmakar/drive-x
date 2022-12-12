import cv2
 
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "src/models/ESPCN_x4.pb"
sr.readModel(path) 
sr.setModel("espcn", 4) # set the model by passing the value and the upsampling ratio
result = sr.upsample(img) # upscale the input image