import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import argparse
import cv2

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def check_yolo(orig_image, yolo_net):
    Width = orig_image.shape[1]
    Height = orig_image.shape[0]
    scale = 0.00392   
    blob = cv2.dnn.blobFromImage(orig_image, scale, (416,416), (0,0,0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(get_output_layers(yolo_net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    indices = [class_ids[i[0]] for i in indices]
    return indices

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred, human_pred, output_dir, result_dir, orig_image):

    pred = pred.squeeze()
    pred_np = pred.cpu().data.numpy()
    pred_np = pred_np*255
    mask = pred_np.astype(np.uint8)

    human_pred = human_pred.squeeze()
    human_pred_np = human_pred.cpu().data.numpy()
    human_pred_np = human_pred_np*255
    human_mask = human_pred_np.astype(np.uint8)

    mask = cv2.bitwise_or(mask,human_mask)
    
    # kernel2 = np.ones((3,3), np.uint8)
    # ret,mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)
    # mask = cv2.erode(mask, kernel2, iterations= 2)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_TREE
    max_contour_area = max([cv2.contourArea(cnt) for cnt in contours])
    # remove contours having less than 1/3 area
    rem_contours = list(filter(lambda x: cv2.contourArea(x)/max_contour_area < 0.33, contours))
    cont = np.ones(mask.shape[:2], dtype="uint8") * 255
    cv2.drawContours(cont,rem_contours,-1,0,-1,)
    mask = cv2.bitwise_and(mask, mask, mask=cont)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)

    # orig_image = cv2.imread(image_name)
    mask = cv2.resize(
        mask, (orig_image.shape[1], orig_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # masked_image = cv2.bitwise_and(orig_image, mask)
    masked_white_bg = cv2.bitwise_or(orig_image, 255-mask)

    # img_tile = [[orig_image, mask],
    #             [masked_image, masked_white_bg]]
    # img_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in img_tile])

    masked_white_bg = cv2.hconcat([orig_image, masked_white_bg]) 

    img_name = image_name.split("/")[-1].rsplit(".", 1)[0]
    cv2.imwrite(output_dir+img_name+'.png', mask)
    cv2.imwrite(result_dir+img_name+'.jpg', masked_white_bg)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="input directory",
                        default='./test_data/test_images/')
    parser.add_argument("-o", "--output_dir", help="output directory",
                        default='./test_data/test_results/')
    parser.add_argument("-r", "--result_dir", help="result directory",
                        default='./test_data/test_results/')
    args = parser.parse_args()

    image_dir = args.input_dir
    prediction_dir = args.output_dir
    result_dir = args.result_dir
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp

    model_dir = './saved_models/u2net/u2net.pth'
    human_model_dir = './saved_models/u2net_human_seg/u2net_human_seg.pth'

    img_name_list = glob.glob(image_dir + '*.jpg')

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    net = U2NET(3,1)
    human_net = U2NET(3,1)

    yolo_net = cv2.dnn.readNet('./yolo/yolov3.weights', './yolo/yolov3.cfg')

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
        human_net.load_state_dict(torch.load(human_model_dir))
        human_net.cuda()  
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        human_net.load_state_dict(torch.load(human_model_dir, map_location='cpu'))
    
    net.eval()
    human_net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        del d1,d2,d3,d4,d5,d6,d7

        ## use yolo prediction to determine if person in image
        orig_image = cv2.imread(img_name_list[i_test])
        indices = check_yolo(orig_image, yolo_net)

        if 0 in indices:
            d1,d2,d3,d4,d5,d6,d7= human_net(inputs_test)
            # normalization
            human_pred = d1[:,0,:,:]
            human_pred = normPRED(human_pred)
            del d1,d2,d3,d4,d5,d6,d7
        else:
            human_pred = pred

        # save results to test_results folder
        save_output(img_name_list[i_test],pred, human_pred, prediction_dir, result_dir, orig_image)

if __name__ == "__main__":
    main()
