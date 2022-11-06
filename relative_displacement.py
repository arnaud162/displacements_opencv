import cv2
import numpy as np
thres = 0.45 # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

#frame = cv2.imread ("928c90e398af4f1d8da5867055de4a46-removebg-preview.png")
#frame = cv2.resize(frame, (1280,720))

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


initPosX = -1
initPosY=-1
initArea=-1

calibrationFactorX = 0.001
calibrationFactorY = 1
calibrationFactorZ = 1


while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)

   
    #print(classIds,bbox.shape)

    
    #if more than 1 object are detected, only the biggest is kept
    try:
        if bbox.shape[0]>1:
            max_Area = bbox[0][2]*bbox[0][3]
            max_Area_Indice = 0
            for i in range(1,bbox.shape[0]):
                if(bbox[i][2]*bbox[i][3]> max_Area ):
                    max_Area=bbox[i][2]*bbox[i][3]
                    max_Area_Indice=i
            bbox = np.array([bbox[max_Area_Indice][:]])
            print(bbox.shape)
    except Exception as e:
        print("empty bbox")
                

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):

            area = box[2]*box[3]

            #here we get the initial positions, at the first iteration of the loop
            if (initPosX==-1 and initPosY==-1 and initArea==-1):
                initPosX=box[1]
                initPosY=box[0]
                initArea=area

            

            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            
            #cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            dx = int((initArea-area)*calibrationFactorX)
            dy = int((initPosY-box[0])*calibrationFactorY)
            dz = int((box[1]-initPosX)*calibrationFactorZ)

            
            #print("displacements x "+str(dx)+" y : "+ str(dy)+" z : "+str(dz))
            #displaying the displacements on the video flow
            cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,"dx "+str(dx),(50,100), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,"dy : "+ str(dy),(50,150), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,"dz : "+str(dz),(50,200), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            
            #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        

    #img = cv2.addWeighted(img,1,frame,0.1,0)
    cv2.imshow("Output",img)
    k = cv2.waitKey(25)
    if k == 27:
        break