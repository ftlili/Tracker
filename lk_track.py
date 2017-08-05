#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
sys.path.append("./lib/")
import numpy as np
import cv2
import video
from time import clock
from matplotlib import pyplot as plt
import random

from cv2.xfeatures2d import SIFT_create, SURF_create


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 5000,
                       qualityLevel = 0.0001,
                       minDistance = 7,
                       blockSize = 7 )

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)




def detect_box(img):
    # l = []
    # while (len(l) < 2):
    #     bbox = cv2.selectROI(img, False)
    #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0,255,0))
    #     cv2.waitKey(0)
    #     l.append(((int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))))
    # return l


    # return [((1362, 421), (1428, 479))]
    return [((1362, 421), (1428, 479)), ((545, 488), (598, 511))]



class App:
    def __init__(self, video_src):
        # len of how much to keep track of
        # seems like its the src, pred1, pred2, pred3 ...
        self.track_len = 5
        self.detect_box_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.count_frames = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.detections = []
        # number of random points to add not enough pts are detected
        self.k = 5
        # the number of times we try to get enough random to compute the hommography
        self.tries = 5
        self.FRCNN = False
        self.dif_thresh = 1



    def compute_tracks(self, img0, img1, vis, tracks):
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < self.dif_thresh
        new_tracks = []
        # tr tracked points (src pts), p1 predicted points (dest pts) good (array of whether the predicted points are good or not)
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                print ("not good")
                continue
            # if the prediction is good, append prediction to the src point
            tr.append((x, y))
            if len(tr) > self.track_len:
                del tr[0]
            new_tracks.append(tr)
            # draw the new good predited pts
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        return new_tracks


    def extract_features(self, frame_gray, vis, ft='gftt'):
        mask = np.zeros_like(frame_gray)
        if len(self.detections) != 0:
            for det in self.detections:
                ((x1,y1),(x2,y2)) = det
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0))
                mask[y1:y2, x1:x2] = 255

        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        if ft=='gftt':
            return cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        elif ft=='sift':
            sift = SIFT_create()
            kp = sift.detect(frame_gray, mask=mask)
            points = [k.pt for k in kp]
            return points




    def track(self, frame_idx, frame, frame_gray, vis):
        if len(self.tracks) > 0:
                # if we're currently tracking len(self.tracks) number of corners
                img0, img1 = self.prev_gray, frame_gray
                self.tracks = self.compute_tracks(img0, img1, vis, self.tracks)
                pts = [p for p in self.tracks]
                # for every tr is tracks draw a line between all its points
                # That would be from src, to pred1, to pred2, to pred3 ... to predtrack_len -1
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                # only keep the last 2 tracks for each corner
                pts = list(map(lambda p: [p[-2], p[-1]], pts))
                # seperate the points by their bounding box
                pts_per_box = [[[],[]] for b in self.detections]

                for [(psx, psy), (pdx, pdy)] in pts:
                    for idx, ((bx1, by1), (bx2, by2)) in enumerate(self.detections):
                        # if the point falls within the box
                        if (psx >= bx1) and (psx <= bx2) and (psy >= by1) and (psy <= by2):
                            pts_per_box[idx][0].append((psx, psy))
                            pts_per_box[idx][1].append((pdx, pdy))
                            # move to the next pt
                            break
                

                # to be able to compute the hommography, we need at least 4 points
                new_detections = []
                for i, ((bx1, by1), (bx2, by2)) in enumerate(self.detections):
                    n = 0
                    while len(pts_per_box[i][0]) <= 3:
                        if n == self.tries:
                            self.FRCNN = True
                            break
                        n += 1
                        print (bx1, bx2)
                        print (min(self.k, abs(bx1 - bx2)))
                        extra_pts = []
                        lx = np.float32(random.sample(range(bx1, bx2), min(self.k, abs(bx1 - bx2))))
                        ly = np.float32(random.sample(range(by1, by2), min(self.k, abs(by1 - by2))))
                        for x in lx:
                            for y in ly:
                                extra_pts.append([(x,y)])
                        # extra_pts = list(zip(lx, ly))
                        # extra_pts = list(map(lambda p: [p], extra_pts))
                        print ("extra_pts", extra_pts)
                        print (len(extra_pts))
                        extra_tracks = self.compute_tracks(img0, img1, vis, extra_pts)
                        print ("extra_tracks", extra_tracks)
                        print (len(extra_tracks))
                        for [(psx, psy), (pdx, pdy)] in extra_tracks:
                            pts_per_box[i][0].append((psx, psy))
                            pts_per_box[i][1].append((pdx, pdy))

                    # compute the hommorgraphy to get the new bounding box

                # for i in range(len(self.detections)):
                    print ("number of points for box", i, "is ", len(pts_per_box[i][0]))

                    # BAD STYLE: TODO: FIX ME PLZ
                    failed = False
                    try:
                        H, mask = cv2.findHomography(np.int32(pts_per_box[i][0]), 
                            np.int32(pts_per_box[i][1]), cv2.RANSAC, 5.0)
                    except:
                        failed = True
                    if H is None: failed = True
                    if failed:
                        print ('FAILED')
                        self.FRCNN = True
                        self.detections = []
                        return

                    ((xb1, yb1), (xb2, yb2)) = self.detections[i]
                    a = np.array([[xb1, yb1], [xb2, yb2]], dtype='float32')
                    a = np.array([a])
                    new_box = cv2.perspectiveTransform(a, H)
                    x1 = int(new_box[0][0][0])
                    y1 = int(new_box[0][0][1])
                    x2 = int(new_box[0][1][0])
                    y2 = int(new_box[0][1][1])
                    if x1 <= x2:
                        if y1 <= y2:
                            new_detections.append(((x1, y1), (x2, y2)))
                        else:
                            new_detections.append(((x1, y2), (x2, y1)))
                    else:
                        if y1 <= y2:
                            new_detections.append(((x2, y1), (x1, y2)))
                        else:
                            new_detections.append(((x2, y2), (x1, y1)))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255,0,0))
                self.detections = new_detections






    def detect(self, frame_idx, frame, frame_gray, vis):
        # We're at a detection frame:
        if (frame_idx == 2) or (self.FRCNN == True):
        # if (frame_idx % self.detect_box_interval) == 0:
            self.detections = detect_box(frame)

        p = self.extract_features(frame_gray, vis, ft='sift')
        # p = self.extract_features(frame_gray, vis, ft='gftt')

        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.tracks.append([(x, y)])
        print ("in detect", len(self.tracks))





    def run(self, temp_box=None):
        for frame_idx in range(self.count_frames):
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            # Tracking phase:
            self.track(frame_idx, frame, frame_gray, vis)
            print ("detections: ", self.detections)
            # Detection phase:
            self.detect(frame_idx, frame, frame_gray, vis)
            self.prev_gray = frame_gray
            cv2.namedWindow('lk_track',cv2.WINDOW_NORMAL)
            imS = cv2.resize(vis, (1400, 1024))
            cv2.imshow('lk_track', imS)
            ch = cv2.waitKey(0)
            # if the key is the esc key then exit
            if ch == 27:
                break





def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
