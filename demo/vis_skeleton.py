# import glob
# from pyskl.smp import *
from visualize_skl import *
from mmcv import load, dump
import cv2
import time
import numpy as np
import glob
import os

settings = {"ntu60": {
    "annotations": os.path.join(os.path.expandvars("$LSDF"), 'data/activity/NTU_RGBD/poses/pyskl/ntu60_hrnet.pkl'),
    "vid_root": os.path.join(os.path.expandvars("$LSDF"), 'data/activity/NTU_RGBD/rgb'),
    "pattern": "{vid_root}/{frame_dir}_rgb.avi"
},
    "kinetics": {
        "annotations": os.path.join(os.path.expandvars("$LSDF"),
                                    'data/activity/Kinetics/kinetics400/poses/pyskl/k400_hrnet.pkl'),
        "vid_root": os.path.join(os.path.expandvars("$LSDF"), 'data/activity/Kinetics/kinetics400/videos'),
        "pattern": "{vid_root}/*/{frame_dir}_*_*.mp4"
    },
    "hmdb51": {
        "annotations": os.path.join(os.path.expandvars("$LSDF"), 'data/activity/HMDB51/poses/pyskl/hmdb51_hrnet.pkl'),
        "vid_root": os.path.join(os.path.expandvars("$LSDF"), 'data/activity/HMDB51/videos'),
        "pattern": "{vid_root}/*/{frame_dir}.avi"
    }
}

dataset = "kinetics"

annotations_root = os.path.split(settings[dataset]["annotations"])[0]
print(f"Loading annotation file from {settings[dataset]['annotations']}...")
annotations = load(settings[dataset]["annotations"])
print("Finished.")
vid_root = settings[dataset]["vid_root"]

for index in range(4, 100):
    anno = annotations["annotations"][index]

    vid = next(iter(glob.glob(settings[dataset]["pattern"].format(vid_root=vid_root, frame_dir=anno["frame_dir"]))),
               None)
    frames, fps = Vis2DPose(anno, thre=0.2, out_shape=(540, 960), layout='coco', fps=12,
                            video=vid, annotations_root=annotations_root)

    # Read until video is completed
    lastt = time.time()
    for i in range(len(frames)):
        # Capture frame-by-frame
        frame = frames[i]

        while lastt + 1. / (fps * 2) > time.time():
            time.sleep(0.001)

        lastt = time.time()

        # Display the resulting frame
        cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Closes all the frames
    cv2.destroyAllWindows()

    print("Finished")
