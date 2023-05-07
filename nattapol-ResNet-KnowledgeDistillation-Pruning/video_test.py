import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import time
from utils.models import resnet
from blazeface.blazeface import BlazeFace
import argparse
from enum import Enum

parser = argparse.ArgumentParser(description='Video Config')
parser.add_argument('--device', type=str, default='cpu', help='cpu | gpu | cuda:0 | cuda:1 | mps')
parser.add_argument('--proj_path', type=str, 
                    default='/Users/nattapolchanpaisit/Documents/GitHub/Eyegaze-Estimation/nattapol-ResNet-KnowledgeDistillation-Pruning',
                    help='path to project')
parser.add_argument('--model', type=str, default='resnet10', 
                    choices=['resnet10', 'resnet18'], 
                    help='model type')
parser.add_argument('--kd', action=argparse.BooleanOptionalAction)
parser.add_argument('--prune', action=argparse.BooleanOptionalAction, 
                    help='pruned model?')
parser.add_argument('--video_device_id', type=int, default=0)

args = parser.parse_args()

first_read = True
cap = cv2.VideoCapture(args.video_device_id)
ret, img = cap.read()
video_shape = (img.shape[0], img.shape[1])
video_padding = (img.shape[1] - img.shape[0]) // 2

font = cv2.FONT_HERSHEY_SIMPLEX

blazeface = BlazeFace()
blazeface.load_weights(f'{args.proj_path}/blazeface/blazeface.pth')
blazeface.load_anchors(f'{args.proj_path}/blazeface/anchors.npy')

if args.model=='resnet10':
    model = resnet.resnet10()
    if args.prune:
        model.load_state_dict(torch.load(f'{args.proj_path}/pretrained/ResNet10+P.pt', map_location=torch.device(args.device)))
    elif args.kd:
        model.load_state_dict(torch.load(f'{args.proj_path}/pretrained/ResNet10+.pt', map_location=torch.device(args.device)))
    else:
        model.load_state_dict(torch.load(f'{args.proj_path}/pretrained/ResNet10.pt', map_location=torch.device(args.device)))
else:
    model = resnet.resnet18()
    if args.prune:
        model.load_state_dict(torch.load(f'{args.proj_path}/pretrained/ResNet18+P.pt', map_location=torch.device(args.device)))
    elif args.kd:
        model.load_state_dict(torch.load(f'{args.proj_path}/pretrained/ResNet18+.pt', map_location=torch.device(args.device)))
    else:
        model.load_state_dict(torch.load(f'{args.proj_path}/pretrained/ResNet18.pt', map_location=torch.device(args.device)))
        
model.to(args.device)
model.eval()

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Resize(size=(224,224))
])
preprocess_face = transforms.Compose([
    transforms.CenterCrop(size=(min(video_shape),min(video_shape))),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize(size=(128,128))
])

cur_time = 0
prev_time = 0
fps = 0

histFps = 0

while(ret):
    ret, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray,5,1,1)
    faceTensor = np.transpose(img, (2, 0, 1))
    faceTensor = torch.from_numpy(faceTensor) / 255
    faceTensor = preprocess_face(faceTensor)
    faceTensor = faceTensor.view(1, 3, 128, 128)
    blazefaceOutput = blazeface.predict_on_batch(faceTensor * 127.5 + 127.5)
    
    faces = []
    for each_face in blazefaceOutput:
        if each_face.size(0) < 1:
            continue
        ymin, xmin, ymax, xmax = each_face[0, :4] * video_shape[0]
        ymin, xmin, ymax, xmax = int(ymin), int(xmin + video_padding), int(ymax), int(xmax + video_padding)
        faces.append([xmin, ymin, ymax - ymin, xmax - xmin])
        
    cur_time = time.time()
    if cur_time-prev_time > 1:
        prev_time = time.time()
        histFps = fps
        fps = 0
    fps += 1
    timeEach = {}
    now = time.time()
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            prev = time.time()
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            face_gray = img_gray[y:y+h,x:x+w]
            face = img[y:y+h,x:x+w]
            face_crop = face
            face_crop = np.transpose(face_crop, (2, 0, 1))
            face_torch = torch.from_numpy(face_crop) / 255
            face_torch = face_torch.to(torch.float32)
            if min(face_torch.size()[1:]) < 200:
                continue
                
            face_torch = preprocess(face_torch)
            face_torch = face_torch.view(1, 1, 224, 224)
            with torch.no_grad():
                timeEach['preprocess'] = time.time() - prev
                prev = time.time()
                face_torch = face_torch.to(args.device)
                output, _, _, _, _ = model(face_torch)
                timeEach['inference'] = time.time() - prev
                prev = time.time()
                face_torch = face_torch.view(1, 1, 224, 224)
                output = output.to('cpu')
                output = np.array(output)
                size = 200
                point_to = [-size * math.sin(math.pi/180 * (output[0][0])) * math.cos(math.pi/180 * output[0][1]), size * math.sin(math.pi/180 * output[0][1])]
                yaw = str(float(output[0][0]))[:5]
                pitch = str(float(output[0][1]))[:5]
                timeEach['postprocess'] = time.time() - prev
                prev = time.time()
                cv2.arrowedLine(img, (x + w//2, y + h//2), 
                                (x + w//2 + int(point_to[1]), y + h//2 + int(point_to[0])), 
                                color = (255, 0 ,0), thickness = 4, 
                                tipLength=0.2)
                cv2.putText(img, f'V : {yaw}   H : {pitch}', org = (5, 50), 
                            fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 2, 
                            color = (255, 0, 0), thickness=2)
                if face.shape[0] > 200:
                    img[70:270, 0:200] = face[h//2 - 100 : h//2 + 100, w//2 -100 : w//2 + 100]
    timeEach['total'] = time.time() - now
    count = 0
    for key in timeEach:
        cv2.putText(img, 
                    f'{key}: {str(1000 * timeEach[key])[:5]} ms', 
                    org = (5, 290 + count * 30), 
                    fontFace = cv2.FONT_HERSHEY_PLAIN, 
                    fontScale = 1.5, 
                    color = (255, 0, 0), 
                    thickness=2)
        count += 1
    if histFps > 0:
        cv2.putText(img, 
                    f'FPS : {histFps} s', 
                    org = (5, 290 + count * 30), 
                    fontFace = cv2.FONT_HERSHEY_PLAIN, 
                    fontScale = 2, 
                    color = (0, 255), 
                    thickness=2)
    cv2.putText(img, 
                f'{args.model} kd={args.kd} pruned={args.prune}', 
                org = (5, img.shape[0]-20), 
                fontFace = cv2.FONT_HERSHEY_PLAIN, 
                fontScale = 2, 
                color = (0, 255), 
                thickness=2)
    cv2.imshow("image", img)
    a = cv2.waitKey(1)
    if a & 0xFF == ord('q'):
        break
    elif a==ord('s') and first_read:
        first_read = False
cap.release()
cv2.destroyAllWindows()