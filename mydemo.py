import torch
from torch.autograd import Variable
from ssd import build_ssd
from data import BaseTransform
import torch.backends.cudnn as cudnn
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import VOCroot, VOC_CLASSES as labelmap
import numpy as np
import cv2
import os.path


show_threshold = 0.05
net = build_ssd('test', 300, 2)
net.load_state_dict(torch.load('weights/ssd300_0712_115000.pth'))
testset = VOCDetection(VOCroot, [('2007', 'test')], None, AnnotationTransform())
transform = BaseTransform(net.size, (104, 117, 123))
net.eval()
net.cuda()
cudnn.benchmark = True

print('Finished loading model!')

num_images = len(testset)
for i in range(num_images):
    img = testset.pull_image(i)
    img_id, annotation = testset.pull_anno(i)
    gtloc = tuple(np.array(annotation[0][:-1]).astype('int32'))
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    x = x.cuda()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                         img.shape[1], img.shape[0]])
    im2show = np.copy(img)
    for i in range(1, detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= show_threshold:
            score = detections[0, i, j, 0]
            label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy().astype('int32')
            coords = (pt[0], pt[1], pt[2], pt[3])
            cv2.rectangle(im2show, coords[:2], coords[2:], (255, 205, 51), 2)
            
            cv2.putText(im2show, '%.3f' % ( score), (pt[0], pt[1] - 11), cv2.FONT_HERSHEY_PLAIN,
                1.0, (255, 205, 51), thickness=1)
            j += 1
    cv2.putText(im2show, '%s' % ( img_id), (1, 10), cv2.FONT_HERSHEY_PLAIN,
                1.0, (255, 255, 255), thickness=1)
    cv2.rectangle(im2show, gtloc[:2], gtloc[2:], (0, 0, 255), 2)
    cv2.imshow('demo', im2show)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break