import numpy as np
from imageio import imwrite
from skimage.transform import resize
import seaborn as sns
import matplotlib.pyplot as plt

# this function is borrowed from https://github.com/hwalsuklee/tensorflow-mnist-AAE/blob/master/plot_utils.py
class plot_samples():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28):
        self.DIR = DIR
        assert n_img_x > 0 and n_img_y > 0
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_total_imgs = n_img_x * n_img_y
        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        imwrite(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        img = np.zeros((h * size[0], w * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])
            image_ = resize(image, output_shape=(w,h))
            img[j*h:j*h+h, i*w:i*w+w] = image_
        img = img * 255
        img = img.astype(np.uint8)
        return img

def tsne_plot(tsne_results, labels, path, title=''):
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=tsne_results[:, 0],
                    y=tsne_results[:, 1],
                    hue=labels,
                    legend='full',
                    palette=sns.color_palette('hls', len(set(labels))))
    plt.title(title)
    plt.savefig(dpi=150,
                fname=path,
                bbox_inches='tight')
    plt.close()

