from PIL import Image
import random


def resize(img, size, random_interpolation=False):
    '''
    random_interpolation: randomly choose a filter
    '''

    method = random.choice([
        Image.BOX,
        Image.NEAREST,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
        Image.BILINEAR]) if random_interpolation else Image.BILINEAR
    img = img.resize(size, method)
    return img
