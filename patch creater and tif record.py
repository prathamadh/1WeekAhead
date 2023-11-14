!ls /kaggle/input/pyvips-python-and-deb-package-gpu
# intall the deb packages
!yes | dpkg -i --force-depends /kaggle/input/pyvips-python-and-deb-package/linux_packages/archives/*.deb
# install the python wrapper
!pip install pyvips -f /kaggle/input/pyvips-python-and-deb-package/python_packages/ --no-index
!pip list | grep pyvips

from IPython import display
display.clear_output()


import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET_FOLDER = "/kaggle/input/UBC-OCEAN/"
IMAGES_FOLDER = "./test_tiles"

os.environ['VIPS_CONCURRENCY'] = '4'
os.environ['VIPS_DISC_THRESHOLD'] = '15gb'


import os
import pyvips
import numpy as np
import random
from PIL import Image
from IPython import display
from tqdm import tqdm
def extract_image_tiles(
    p_img,label, folder, size: int = 2048, scale: float = 0.5,
    drop_thr: float = 0.6, white_thr: int = 240, max_samples: int = 50
) -> list:
    name, _ = os.path.splitext(os.path.basename(p_img))
    im = pyvips.Image.new_from_file(p_img)
    w = h = size
    # https://stackoverflow.com/a/47581978/4521646
    idxs = [(y, y + h, x, x + w) for y in range(0, im.height, h) for x in range(0, im.width, w)]
    # random subsample
    max_samples = max_samples if isinstance(max_samples, int) else int(len(idxs) * max_samples)
    random.shuffle(idxs)
    files = []
    for y, y_, x, x_ in tqdm(idxs, total=len(idxs)):        # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
        tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3]
        if tile.shape[:2] != (h, w):
            tile_ = tile
            tile_size = (h, w) if tile.ndim == 2 else (h, w, tile.shape[2])
            tile = np.zeros(tile_size, dtype=tile.dtype)
            tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
        black_bg = np.sum(tile, axis=2) == 0
        tile[black_bg, :] = 255
        img = np.dot(tile[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        white_pixels = np.sum(img>220)
#         mask_bg = np.mean(tile, axis=2) > white_thr
        if np.sum(white_pixels) >= (np.prod(img.shape) * drop_thr):
#             display.clear_output()
#             plt.imshow(tile)
#             plt.show()
            continue
        p_img = os.path.join(folder, f"label-{label}-{int(x_ / w)}-{int(y_ / h)}.png")
        # print(tile.shape, tile.dtype, tile.min(), tile.max())
#         new_size = int(size * scale), int(size * scale)
         #Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img)
        Image.fromarray(tile).save(p_img)

        files.append(p_img)
        # need to set counter check as some empty tiles could be skipped earlier
        if len(files) >= max_samples:
            break
    return files


from torchvision import transforms as T

img_color_mean = [0.8721593659261734, 0.7799686061900686, 0.8644588534918227]
img_color_std = [0.08258995918115268, 0.10991684444009092, 0.06839816226731532]

VALID_TRANSFORM = T.Compose([
    T.CenterCrop(512),
    T.ToTensor(),
    #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.Normalize(img_color_mean, img_color_std),  # custom
])



import torch
from PIL import Image
from torch.utils.data import Dataset

class TilesFolderDataset(Dataset):

    def __init__(
        self,
        folder: str,
        image_ext: str =  '.png',
        transforms = None
    ):
        assert os.path.isdir(folder)
        self.transforms = transforms
        self.imgs = glob.glob(os.path.join(folder, "*" + image_ext))

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.imgs[idx]
        assert os.path.isfile(img_path), f"missing: {img_path}"
        img = np.array(Image.open(img_path))[..., :3]
        # filter background
        mask = np.sum(img, axis=2) == 0
        img[mask, :] = 255
        if np.max(img) < 1.5:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        # augmentation
        if self.transforms:
            img = self.transforms(Image.fromarray(img))
        #print(f"img dim: {img.shape}")
        return img

    def __len__(self) -> int:
        return len(self.imgs)



def extract_prune_tiles(
    path_img: str,label:str, folder: str, size: int = 2048, scale: float = 0.25,
    drop_thr: float = 0.6,white_thr: int=0.5, max_samples: int = 12000
) -> str:
    print(f"processing: {path_img}")
    name, _ = os.path.splitext(os.path.basename(path_img))
    folder = os.path.join(folder, name)
    os.makedirs(folder, exist_ok=True)
    tiles = extract_image_tiles(
        path_img,label, folder, size=size, scale=scale,
        drop_thr=drop_thr,white_thr=225 ,max_samples=max_samples)
    return folder


import pandas as pd
df=pd.read_csv("/kaggle/input/cancerdatasetwithpath/cancerdf.csv")
df.head()


%%time
for i in range (5):
    path=df.at[i,'Image_path']
    label=df.at[i,'label']
    folder_tiles = extract_prune_tiles(path,label, IMAGES_FOLDER, size=256, scale=1,drop_thr=0.4,white_thr=222)
record()
# print(f"found tiles: {len(dataset)}")

# # quick view
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
# for i in range(9):
#     img = dataset[i]
#     axes[i // 3, i % 3].imshow(img)
# fig.tight_layout()




dataset = TilesFolderDataset("/kaggle/working/test_tiles/1952")
print(f"found tiles: {len(dataset)}")

# quick view
fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
axes=axes.flatten()
for i in range(100):
    ax=axes[i]
    img = dataset[i+2000]
    ax.imshow(img)
    ax.axis('off')
fig.tight_layout()


import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io


import shutil

def record():
    folderpath='/kaggle/working/test_tiles'
    folder = glob.glob(os.path.join(folderpath, "*"))
    image_list=[]
    for path in folder:
        image_list= image_list+(glob.glob(os.path.join(path,"*.png")))
        print(len(image_list))

    random.shuffle(image_list)
    with tf.io.TFRecordWriter("caltech_dataset.tfrecords") as writer:
        for imagepath in image_list:

            pattern = r'label-(\d+)-\d+-\d+\.png'

            match = re.search(pattern, imagepath)
            label = match.group(1)
            label=int(label)
            image = Image.open(imagepath)

            bytes_buffer = io.BytesIO()
            image.convert("RGB").save(bytes_buffer, "JPEG")
            image_bytes = bytes_buffer.getvalue()

            bytes_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            class_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

            example = tf.train.Example(
              features=tf.train.Features(feature={
                  "image": bytes_feature,
                  "class": class_feature
              })
            )

            writer.write(example.SerializeToString())

            image.close()
    for directory in folder:
        try:
            shutil.rmtree(directory)
            print(f"Directory '{directory}' removed successfully.")
        except OSError as e:
            print(f"Error removing directory '{directory}': {e}")




def _bytestring_to_pixels(parsed_example):
    byte_string = parsed_example['image']
    image = tf.io.decode_image(byte_string)
    image = tf.reshape(image, [256, 256, 3])
    return image, parsed_example["class"]
AUTOTUNE = tf.data.AUTOTUNE
def load_and_extract_images(filepath):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_data, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(_bytestring_to_pixels, num_parallel_calls=AUTOTUNE) # .cache()
    return dataset

caltech_dataset = load_and_extract_images("caltech_dataset.tfrecords")
t.map(_bytestring_to_pixels, num_parallel_calls=AUTOTUNE) # .cache()
    return dataset
train_dataset = caltech_dataset.take(90)

cropsize=256
def _train_data_preprocess_and_augment(image, label):
#     image = tf.cast(image, tf.float32)
#     image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    
    return image, label
train_preprocessed_augmented = train_dataset.map(_train_data_preprocess_and_augment)

for images, label_batch in train_preprocessed_augmented.batch(32):
    for image in 
image_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string), 
    "class": tf.io.FixedLenFeature([], tf.int64), 
    }
def _parse_data(unparsed_example):
    return tf.io.parse_single_example(unparsed_example, image_feature_description)

def _bytestring_to_pixels(parsed_example):
    byte_string = parsed_example['image']
    image = tf.io.decode_image(byte_string)
    image = tf.reshape(image, [256, 256, 3])
    return image, parsed_example["class"]
AUTOTUNE = tf.data.AUTOTUNE
def load_and_extract_images(filepath):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_data, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(_bytestring_to_pixels, num_parallel_calls=AUTOTUNE) # .cache()
    return dataset

caltech_dataset = load_and_extract_images("caltech_dataset.tfrecords")

train_dataset = caltech_dataset.take(90)

cropsize=256
def _train_data_preprocess_and_augment(image, label):
#     image = tf.cast(image, tf.float32)
#     image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    
    return image, label
train_preprocessed_augmented = train_dataset.map(_train_data_preprocess_and_augment)
fig,axes=plt.subplots(8,4,figsize=(10,10))
axes=axes.flatten()
for(images, label_batch) in train_preprocessed_augmented.batch(32):
    for i in range(32):
        ax=axes[i] 
        ax.imshow(images[i])
        ax.set_title(f"label-{label_batch[i]}")

    plt.axis("off")
    plt.show()
    break
        
