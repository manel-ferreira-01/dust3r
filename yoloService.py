import io
from scipy.io import savemat, loadmat
import cv2  #install opencv-python and opencv-contrib-python
import generic_box_pb2
import numpy as np
import os
import logging

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

YOLO_CONFIG_DIR = ""

#----YOLO_Predict---------------------------------
def predict(datafile,model):
    '''
    * Function:     predict
    * Arguments:    datafile (binary .mat file)                 -.mat file containing the image in 'im'(int) and frame number in 'frame'(int). It must have these elements.
    *               model (yolo model)                          -Model loaded to run the prediction funtion
    *               
    * Returns:      generic_box_pb2.Data (grpc binary message)  -grpc message containing the .mat file with yolo's prediction
    *
    * Description:  Runs predict function from the yolo model (from model variable) with the input image from datafile (binary .mat file). Used as the method function.
    '''
    # Read data from mat file
    dados = loadmat(io.BytesIO(datafile)) 
    if not ('im' in dados):
        logging.exception(f'''[ERRO IN PREDICT: No image found in dictionary key 'im']''')
    img = dados['im']

    if not ('frame' in dados):
        frameNum = 0
    else:
        frameNum = dados['frame']

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    #process yolo
    yoloResults = YOLOPredict(img,model)

    DataMat = saveResultsToMat(yoloResults,frameNum)

    return generic_box_pb2.Data(file=DataMat)

def YOLOPredict(img,model):
    '''
    * Function:     YOLOPredict
    * Arguments:    img                                         -Image array containing the image to analyse
    *               model (yolo model)                          -Model loaded to run the prediction funtion
    *               
    * Returns:      results (yolo results)                      -yolo Tuple with all the predict's results. 
    *                                                           Check https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results for details on result's structure.
    *
    * Description:  Runs predict function from the yolo model (from model variable) with the input image from img
    '''
    image = cv2.resize(img,(640,369))
    # Run YOLOv8 prediction on the frame
    results = model.predict(image)

    return results

#----YOLO_Track---------------------------------
def track(datafile,model):
    '''
    * Function:     track
    * Arguments:    datafile (binary .mat file)                 -.mat file containing the image in 'im'(int) and frame number in 'frame'(int). It must have these elements.
    *               model (yolo model)                          -Model loaded to run the traking funtion
    *               
    * Returns:      generic_box_pb2.Data (grpc binary message)  -grpc message containing the .mat file with yolo's traking
    *
    * Description:  Runs track function from the yolo model (from model variable) with the input image from datafile (binary .mat file). Used as the method function.
    '''
    # Read data from mat file
    dados = loadmat(io.BytesIO(datafile)) 
    if not ('im' in dados):
        logging.exception(f'''[ERRO IN TRACK: No image found in dictionary key 'im']''')
    img = dados['im']
    
    if not ('frame' in dados):
        frameNum = 0
    else:
        frameNum = dados['frame']

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    #process yolo
    yoloResults = YOLOTrack(img,model)
    
    DataMat = saveResultsToMat(yoloResults,frameNum)

    return generic_box_pb2.Data(file=DataMat)

def YOLOTrack(img,model):
    '''
    * Function:     YOLOTrack
    * Arguments:    img                                         -Image array containing the image to analyse
    *               model (yolo model)                          -Model loaded to run the traking funtion
    *               
    * Returns:      results (yolo results)                      -yolo Tuple with all the track's results. 
    *                                                           Check https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results for details on result's structure.
    *
    * Description:  Runs track function from the yolo model (from model variable) with the input image from img
    '''

    image = cv2.resize(img,(640,369))
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(image, persist=True)

    return results

#----YOLO_Plot---------------------------------
def plot(im,data,model):
    '''
    * Function:     plot
    * Arguments:    im (binary .mat file)                       -.mat file containing the image in 'im'(int) and the uses's session hash in 'session_hash'. It must have these elements.
    *               data (binary .mat file)                     -.mat file containing the information form the yolo models.
    *                                                             Must have 'cls' representing object's classes.
    *                                                             If 'cls' is difrente from -1 (meaning yolo detected some object), this function expects to also have in the .mat file:
    *                                                             'xyxy' -> The top left corner and bottom right corner coordenates of the rectangle that surrounds the object;
    *                                                             'conf' -> The confidence that the yolo model has of the object's classification;
    *                                                             'id'   -> The identifier of the object in case of tracking.
    *                                                       
    *               model (yolo model)                          -Model used for traking/predict, used to get the class names (might change depending on the model)
    *               
    * Returns:      generic_box_pb2.Data (grpc binary message)  -grpc message containing the .mat file with the ploted image
    *
    * Description:  Plots the resulting image from the data from yolo results and the original image. 
    '''

    # Read data from mat file
    imdata = loadmat(io.BytesIO(im)) 

    if not ('im' in imdata):
        logging.exception(f'''[ERRO IN PLOT: No image found in dictionary key 'im']''')
    img = imdata['im']
    
    if not ('session_hash' in imdata):
        session_hash = 0
    else:
        session_hash = imdata['session_hash']

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(640,369))
    Data = loadmat(io.BytesIO(data)) 

    dados = Data[list(Data)[-1]]

    if (dados['cls'][0][0][0][0] == -1):
        # Save the numpy array to a .mat file
        imgMatFile = saveBinaryMat({'im': img,'session_hash':session_hash})

        return generic_box_pb2.Data(file = imgMatFile)

    # Extract relevant data from the dictionary
    try:
        xyxy = dados['xyxy'][0][0]  # Bounding boxes in (x1, y1, x2, y2) format
        conf = dados['conf'][0][0][0]  # Confidence scores
        cls = dados['cls'][0][0][0]     # Class indices 
        ids = dados['id'][0][0][0]
    except:
        logging.exception(f'''[WARNING IN PLOT: data from mat file did not have enough information for ploting. Returning empty image.]''')
        # Save the numpy array to a .mat file
        imgMatFile = saveBinaryMat({'im': img,'session_hash':session_hash})
        return generic_box_pb2.Data(file = imgMatFile)

    # Define class names
    class_names = model.names


    # Display the image
    fig, ax = plt.subplots(1)
    ax.imshow(img)


    for i, box in enumerate(xyxy):

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Label with class and confidence
        label = f"{class_names[int(cls[i])]}: {conf[i]:.2f}"
        if not ids=='None':
           label+=f", ID: {int(ids[i])}"
        plt.text(x1, y1 - 10, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Convert the plot to a numpy array
    buf_image = Image.open(buf)
    buf_array = np.array(buf_image)

    # Save the numpy array to a .mat file
    imgMatFile = saveBinaryMat({'im': buf_array,'session_hash':session_hash})

    # Close the plot
    plt.close(fig)


    return generic_box_pb2.Data(file = imgMatFile)


#----Save Results To Mat---------------------------------

def saveResultsToMat(results,franeNum):
    '''
    * Function:     saveResultsToMat
    * Arguments:    results (yolo results)                 -yolo results outputed by the tracking and predict functions.
    *               franeNum (int)                          -The number of the frame that this image comes from.
    *               
    * Returns:      saveBinaryMat(MatDic) (binary .mat file)  -binary .mat file containing all the necessary results from yolo results
    *
    * Description:  Creates a binary .mat file with all important information from yolo results.
    '''


    dataDic={}
    for result in results:
        
        #Make sure there was something detected
        if len(result.boxes.cls.cpu().numpy())<1:
            dataDic['cls'] = np.array([[-1]], dtype=np.float32)
            break

        dataDic['cls'] = np.array(result.boxes.cls.cpu().numpy())
        dataDic['conf'] = np.array(result.boxes.conf.cpu().numpy())
        dataDic['data'] = np.array(result.boxes.data.cpu().numpy())

        #In the case of Predict, the id is None
        if result.boxes.id != None:
            dataDic['id'] = result.boxes.id.numpy()
        else:
            dataDic['id'] = ['None']

        dataDic['is_track'] = result.boxes.is_track
        dataDic['orig_shape'] = result.boxes.orig_shape
        dataDic['xywh'] = np.array(result.boxes.xywh.cpu().numpy())
        dataDic['xywhn'] = np.array(result.boxes.xywhn.cpu().numpy())
        dataDic['xyxy'] = np.array(result.boxes.xyxy.cpu().numpy())
        dataDic['xyxyn'] = np.array(result.boxes.xyxyn.cpu().numpy())

    print(dataDic)

    MatDic={'data_'+ f'{int(franeNum):05d}':dataDic}
    return saveBinaryMat(MatDic)


def saveBinaryMat(dic):
    '''
    * Function:     saveBinaryMat
    * Arguments:    dic (dictionary)                 -dictionary to save in to the binary .mat file
    *               
    * Returns:      bytesData (binary .mat file)  -binary .mat file containing the dictionary
    *
    * Description:  Gets the binary data from  a .mat file containing the dinctionary dic
    '''

    #save mat file and open it as binary
    savemat(str(list(dic)[-1])+"data.mat",dic,long_field_names=True)
    with open(str(list(dic)[-1])+"data.mat", 'rb') as fp:
        bytesData = fp.read()
    os.remove(str(list(dic)[-1])+"data.mat")

    return bytesData
