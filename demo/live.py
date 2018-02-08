from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')

parser.add_argument('--weights', default='../weights/ssd300_0712_0.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                int(pt[3])), COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), FONT,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        frame = stream.read()
        # update FPS counter
        fps.update()
        frame = predict(frame)
        # convert cv2 -> matplotlib, with r,g,b diff
        (r,g,b) = cv2.split(frame)
        frame = cv2.merge([b,g,r])
        plt.imshow(frame)
        plt.show()


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 21)    # initialize SSD
    
    #net.load_state_dict(torch.load(args.weights))
    # Model from GPU->CPU(get state dict)
    net.load_state_dict((torch.load(args.weights, map_location=lambda storage,loc:storage)))

    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    fps = FPS().start()
    # stop the timer and display FPS information
    cv2_demo(net.eval(), transform)
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    stream.stop()
