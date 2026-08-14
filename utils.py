import os

import tqdm
from PIL import Image


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def compress_image(input_path, output_path, quality=85):
    """Convert an image to RGB JPEG at the requested quality."""
    try:
        with Image.open(input_path) as image:
            image.convert('RGB').save(output_path, 'JPEG', quality=quality)
    except Exception as error:
        print(f'Failed to compress {input_path}: {error}')


def jpeg_folder(ori_imgs_path, save_path, quality=75):
    """Convert all supported images in a folder to JPEG."""
    image_names = sorted(
        name for name in os.listdir(ori_imgs_path)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    output_folder = os.path.join(save_path, f'jpeg_{quality}')
    os.makedirs(output_folder, exist_ok=True)

    for image_name in tqdm.tqdm(image_names, desc='Converting to JPEG'):
        input_path = os.path.join(ori_imgs_path, image_name)
        image_stem = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_folder, f'{image_stem}.jpg')
        compress_image(input_path, output_path, quality=quality)

    return output_folder


def resize2_other_folder(image_path, save_path, target_size=512):
    """Resize all supported images in a folder to a square target size."""
    os.makedirs(save_path, exist_ok=True)
    image_names = sorted(
        name for name in os.listdir(image_path)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_names:
        print(f'No images found in {image_path}')
        return

    print(f'Resizing {len(image_names)} images...')
    for image_name in tqdm.tqdm(image_names, desc='Resizing images'):
        input_path = os.path.join(image_path, image_name)
        output_path = os.path.join(save_path, image_name)
        try:
            with Image.open(input_path) as image:
                resized = image.resize(
                    (target_size, target_size), Image.Resampling.LANCZOS
                )
                save_options = {'quality': 95}
                if 'exif' in image.info:
                    save_options['exif'] = image.info['exif']
                resized.save(output_path, **save_options)
        except Exception as error:
            print(f'Failed to resize {image_name}: {error}')
