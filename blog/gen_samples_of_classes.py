from PIL import Image
import numpy as np
from data import training_dataset

SAMPLE_GRID_SIZE = 5   # each sample will have SGSxSGS images
NUM_CLASSES = 10


def pil_img_from_array(array):
    return Image.fromarray((array * 255.0).astype(np.uint8))


class Collage(object):
    def __init__(self):
        self.collage = Image.new('RGB', (64*SAMPLE_GRID_SIZE,
                                         64*SAMPLE_GRID_SIZE))
        self.insert_idx = 0
        self.full = False

    def add(self, np_array):
        # return true on last pasted
        if self.full:
            return False
        x = int((self.insert_idx // SAMPLE_GRID_SIZE) * 64)
        y = int((self.insert_idx % SAMPLE_GRID_SIZE) * 64)
        self.collage.paste(pil_img_from_array(np_array), (x, y))
        self.insert_idx += 1
        if self.insert_idx == SAMPLE_GRID_SIZE * SAMPLE_GRID_SIZE:
            self.full = True
        return self.full


collages = [Collage() for _ in range(NUM_CLASSES)]

num_collages_full = 0
for imgs, labels in training_dataset(batch_size=128):
    for img, label in zip(imgs, labels):
        if collages[label].add(img):
            num_collages_full += 1
            if num_collages_full == 10:
                break

class_names = ['Annual_Crop', 'Forest', 'Herbaceous_Vegetation', 'Highway',
               'Industrial_Buildings', 'Pasture', 'Permanent_Crop',
               'Residential_Buildings', 'River', 'Sea_Lake']
for i, c in enumerate(collages):
    c.collage.save("collage.%02d.%s.png" % (i, class_names[i]))
