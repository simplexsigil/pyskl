import io
import os.path

import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
from mmcv import load
from tqdm import tqdm
from scipy.stats import mode as get_mode

class DecompressPose:
    """Load Compressed Pose

    In compressed pose annotations, each item contains the following keys:
    Original keys: 'label', 'frame_dir', 'img_shape', 'original_shape', 'total_frames'
    New keys: 'frame_inds', 'keypoint', 'anno_inds'.
    This operation: 'frame_inds', 'keypoint', 'total_frames', 'anno_inds'
         -> 'keypoint', 'keypoint_score', 'total_frames'

    Args:
        squeeze (bool): Whether to remove frames with no human pose. Default: True.
        max_person (int): The max number of persons in a frame, we keep skeletons with scores from high to low.
            Default: 10.
    """

    def __init__(self,
                 squeeze=True,
                 max_person=10):

        self.squeeze = squeeze
        self.max_person = max_person

    def __call__(self, results):

        required_keys = ['total_frames', 'frame_inds', 'keypoint']
        for k in required_keys:
            assert k in results

        total_frames = results['total_frames']
        frame_inds = results.pop('frame_inds')
        keypoint = results['keypoint']

        if 'anno_inds' in results:
            frame_inds = frame_inds[results['anno_inds']]
            keypoint = keypoint[results['anno_inds']]

        assert np.all(np.diff(frame_inds) >= 0), 'frame_inds should be monotonical increasing'

        def mapinds(inds):
            uni = np.unique(inds)
            map_ = {x: i for i, x in enumerate(uni)}
            inds = [map_[x] for x in inds]
            return np.array(inds, dtype=np.int16)

        if self.squeeze:
            frame_inds = mapinds(frame_inds)
            total_frames = np.max(frame_inds) + 1

        results['total_frames'] = total_frames

        num_joints = keypoint.shape[1]
        num_person = get_mode(frame_inds)[-1][0]

        new_kp = np.zeros([num_person, total_frames, num_joints, 2], dtype=np.float16)
        new_kpscore = np.zeros([num_person, total_frames, num_joints], dtype=np.float16)
        # 32768 is enough
        nperson_per_frame = np.zeros([total_frames], dtype=np.int16)

        for frame_ind, kp in zip(frame_inds, keypoint):
            person_ind = nperson_per_frame[frame_ind]
            new_kp[person_ind, frame_ind] = kp[:, :2]
            new_kpscore[person_ind, frame_ind] = kp[:, 2]
            nperson_per_frame[frame_ind] += 1

        if num_person > self.max_person:
            for i in range(total_frames):
                nperson = nperson_per_frame[i]
                val = new_kpscore[:nperson, i]
                score_sum = val.sum(-1)

                inds = sorted(range(nperson), key=lambda x: -score_sum[x])
                new_kpscore[:nperson, i] = new_kpscore[inds, i]
                new_kp[:nperson, i] = new_kp[inds, i]
            num_person = self.max_person
            results['num_person'] = num_person

        results['keypoint'] = new_kp[:num_person]
        results['keypoint_score'] = new_kpscore[:num_person]
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(squeeze={self.squeeze}, max_person={self.max_person})')


class Vis3DPose:

    def __init__(self, item, layout='nturgb+d', fps=12, angle=(30, 45), fig_size=(8, 8), dpi=80):
        kp = item['keypoint']
        self.kp = kp
        assert self.kp.shape[-1] == 3
        self.layout = layout
        self.fps = fps
        self.angle = angle  # For 3D data only
        self.colors = ('#3498db', '#000000', '#e74c3c')  # l, m, r
        self.fig_size = fig_size
        self.dpi = dpi

        assert layout == 'nturgb+d'
        if self.layout == 'nturgb+d':
            self.num_joint = 25
            self.links = np.array([
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
                (19, 18), (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)], dtype=np.int) - 1
            self.left = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23], dtype=np.int) - 1
            self.right = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25], dtype=np.int) - 1
            self.num_link = len(self.links)
        self.limb_tag = [1] * self.num_link

        for i, link in enumerate(self.links):
            if link[0] in self.left or link[1] in self.left:
                self.limb_tag[i] = 0
            elif link[0] in self.right or link[1] in self.right:
                self.limb_tag[i] = 2

        assert len(kp.shape) == 4 and kp.shape[3] == 3 and kp.shape[2] == self.num_joint
        x, y, z = kp[..., 0], kp[..., 1], kp[..., 2]

        min_x, max_x = min(x[x != 0]), max(x[x != 0])
        min_y, max_y = min(y[y != 0]), max(y[y != 0])
        min_z, max_z = min(z[z != 0]), max(z[z != 0])
        max_axis = max(max_x - min_x, max_y - min_y, max_z - min_z)
        mid_x, mid_y, mid_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2
        self.min_x, self.max_x = mid_x - max_axis / 2, mid_x + max_axis / 2
        self.min_y, self.max_y = mid_y - max_axis / 2, mid_y + max_axis / 2
        self.min_z, self.max_z = mid_z - max_axis / 2, mid_z + max_axis / 2

        self.images = []

    def get_img(self, dpi=80):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        return cv2.imdecode(img, -1)

    def vis(self):
        self.images = []
        plt.figure(figsize=self.fig_size)
        for t in range(self.kp.shape[1]):
            ax = plt.gca(projection='3d')
            ax.set_xlim3d([self.min_x, self.max_x])
            ax.set_ylim3d([self.min_y, self.max_y])
            ax.set_zlim3d([self.min_z, self.max_z])
            ax.view_init(*self.angle)
            ax.set_aspect('auto')
            for i in range(self.num_link):
                for m in range(self.kp.shape[0]):
                    link = self.links[i]
                    color = self.colors[self.limb_tag[i]]
                    j1, j2 = self.kp[m, t, link[0]], self.kp[m, t, link[1]]
                    if not ((np.allclose(j1, 0) or np.allclose(j2, 0)) and link[0] != 1 and link[1] != 1):
                        ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], lw=1, c=color)
            self.images.append(self.get_img(dpi=self.dpi))
            ax.cla()
        return mpy.ImageSequenceClip(self.images, fps=self.fps)


def Vis2DPose(item, thre=0.2, out_shape=(540, 960), layout='coco', fps=30, video=None, annotations_root=None,
              pose_decompressor: DecompressPose = DecompressPose(squeeze=False)):
    if isinstance(item, str):
        item = load(item)

    assert layout == 'coco'

    if "keypoint" not in item:  # Kinetics:
        print("Loading keypoint subset...")
        sample_keypoints_subset = load(os.path.join(annotations_root, "kpfiles", os.path.split(item["raw_file"])[1]))
        print("Finished.")
        kp = sample_keypoints_subset[item["frame_dir"]]["keypoint"]
        item['keypoint'] = kp
        item['frame_inds'] -= 1

        item = pose_decompressor(item)

    kp = item['keypoint']
    kp_frames = kp.shape[1]

    # Cat-ing score
    if 'keypoint_score' in item:
        kpscore = item['keypoint_score']
        kp = np.concatenate([kp, kpscore[..., None]], -1)

    assert kp.shape[-1] == 3

    # Keypoint output scaling
    img_shape = item.get('img_shape', out_shape)
    kp[..., 0] *= out_shape[1] / img_shape[1]
    kp[..., 1] *= out_shape[0] / img_shape[0]

    total_frames = item.get('total_frames', kp_frames)
    assert total_frames == kp_frames

    if video is None:
        frames = [np.ones([out_shape[0], out_shape[1], 3], dtype=np.uint8) * 255 for i in range(total_frames)]
    else:
        frames = []

        cap = cv2.VideoCapture(video)
        fpsc = cap.get(cv2.CAP_PROP_FPS)
        if not fpsc:
            print(f"Could not read fps from file, assuming {fps}")
        else:
            fps = fpsc

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # frames = [x.asnumpy() for x in vid]
        frames = [cv2.resize(x, (out_shape[1], out_shape[0])) for x in frames]
        if len(frames) != total_frames:
            frames = [frames[int(i / total_frames * len(frames))] for i in range(total_frames)]

    if layout == 'coco':
        edges = [
            (0, 1, 'f'), (0, 2, 'f'), (1, 3, 'f'), (2, 4, 'f'), (0, 5, 't'), (0, 6, 't'),
            (5, 7, 'ru'), (6, 8, 'lu'), (7, 9, 'ru'), (8, 10, 'lu'), (5, 11, 't'), (6, 12, 't'),
            (11, 13, 'ld'), (12, 14, 'rd'), (13, 15, 'ld'), (14, 16, 'rd')
        ]
    color_map = {
        'ru': ((0, 0x96, 0xc7), (0x3, 0x4, 0x5e)),
        'rd': ((0xca, 0xf0, 0xf8), (0x48, 0xca, 0xe4)),
        'lu': ((0x9d, 0x2, 0x8), (0x3, 0x7, 0x1e)),
        'ld': ((0xff, 0xba, 0x8), (0xe8, 0x5d, 0x4)),
        't': ((0xee, 0x8b, 0x98), (0xd9, 0x4, 0x29)),
        'f': ((0x8d, 0x99, 0xae), (0x2b, 0x2d, 0x42))}

    for i in tqdm(range(total_frames)):
        for m in range(kp.shape[0]):
            ske = kp[m, i]
            for e in edges:
                st, ed, co = e
                co_tup = color_map[co]
                j1, j2 = ske[st], ske[ed]
                j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
                conf = min(j1[2], j2[2])
                if conf > thre:
                    color = [x + (y - x) * (conf - thre) / 0.8 for x, y in zip(co_tup[0], co_tup[1])]
                    color = tuple([int(x) for x in color])
                    frames[i] = cv2.line(frames[i], (j1x, j1y), (j2x, j2y), color, thickness=2)
    return frames, fps
