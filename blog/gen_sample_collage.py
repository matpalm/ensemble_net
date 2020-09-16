from PIL import Image, ImageDraw
from data import validation_dataset
import numpy as np


def pil_img_from_array(array):
    return Image.fromarray((array * 255.0).astype(np.uint8))


classes = ['Annual Crop', 'Forest', 'Herbaceous\nVegetation', 'Highway',
           'Industrial\nBuildings', 'Pasture', 'Permanent Crop',
           'Residential\nBuildings', 'River', 'Sea & Lake']
collage = Image.new('RGB', (128*5, 128*2))
canvas = ImageDraw.Draw(collage, 'RGB')
insert_class = 0
for imgs, labels in validation_dataset(batch_size=128):
    for img, label in zip(imgs, labels):
        if label == insert_class:
            r, c = insert_class % 5, insert_class//5
            print("insert", insert_class, "at", r, c)
            pil_img = pil_img_from_array(img).resize((128, 128))
            canvas = ImageDraw.Draw(pil_img, 'RGB')
            canvas.text((10, 10), classes[insert_class], 'black')
            canvas.text((9, 9), classes[insert_class], 'white')
            collage.paste(pil_img, (r*128, c*128))
            insert_class += 1
            if insert_class > 9:
                break

collage.save("/tmp/foo.png")
