import argparse
import os
import os.path as osp
import subprocess
import sys
from multiprocessing import Process, Queue

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image

import utils
from BiSeNet.model import BiSeNet


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def list_images(folder):
    return sorted(
        name for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def transform_batch(batch_tensor):
    """Apply ImageNet normalization to a [B, C, H, W] tensor."""
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=batch_tensor.device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=batch_tensor.device
    ).view(1, 3, 1, 1)
    return (batch_tensor - mean) / std


def merge_imgs(imgs_path1, imgs_path2, output_path, alpha):
    imgs1 = list_images(imgs_path1)
    imgs2 = list_images(imgs_path2)
    if len(imgs1) != len(imgs2):
        raise ValueError(
            f'Cannot merge folders with different image counts: '
            f'{len(imgs1)} != {len(imgs2)}'
        )

    os.makedirs(output_path, exist_ok=True)
    image_pairs = zip(imgs1, imgs2)
    for img1_name, img2_name in tqdm.tqdm(
        image_pairs, desc='Merging images', total=len(imgs1)
    ):
        input_img1 = os.path.join(imgs_path1, img1_name)
        input_img2 = os.path.join(imgs_path2, img2_name)
        output_img = os.path.join(output_path, img1_name)

        img1 = Image.open(input_img1).convert('RGB')
        img2 = Image.open(input_img2).convert('RGB').resize(
            img1.size, Image.Resampling.LANCZOS
        )
        arr1 = np.asarray(img1, dtype=np.float32)
        arr2 = np.asarray(img2, dtype=np.float32)
        merged = np.clip(alpha * arr1 + (1 - alpha) * arr2, 0, 255)
        Image.fromarray(merged.astype(np.uint8)).save(output_img)


def face_sr(imgs_path, output_path, alpha, result_queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    output_sr = os.path.join(output_path, 'face_sr')
    merge_path = os.path.join(output_path, 'merge')
    os.makedirs(output_sr, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            osp.join(PROJECT_ROOT, 'GFPGAN', 'inference_gfpgan.py'),
            '-i',
            imgs_path,
            '-o',
            output_sr,
            '-v',
            '1.3',
            '-s',
            '2',
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    restored_path = os.path.join(output_sr, 'restored_imgs')
    merge_imgs(restored_path, imgs_path, merge_path, alpha)
    print(f'Face SR results saved to {merge_path}')
    result_queue.put(merge_path)


def common_sr(imgs_path, output_path, result_queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    output_common = os.path.join(output_path, 'common_sr')
    os.makedirs(output_common, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            osp.join(PROJECT_ROOT, 'ESRGAN', 'inference_realesrgan.py'),
            '-i',
            imgs_path,
            '-o',
            output_common,
            '-n',
            'RealESRGAN_x4plus',
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print(f'Common SR results saved to {output_common}')
    result_queue.put(output_common)


def cut_sr(merge_path, output_common, output_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = BiSeNet(n_classes=19).to(device)
    model_path = osp.join(PROJECT_ROOT, 'BiSeNet', '79999_iter.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    imgs1 = list_images(merge_path)
    imgs2 = list_images(output_common)
    if len(imgs1) != len(imgs2):
        raise ValueError(
            f'Cannot combine folders with different image counts: '
            f'{len(imgs1)} != {len(imgs2)}'
        )

    os.makedirs(output_path, exist_ok=True)
    mask_path = os.path.join(output_path, 'parsing_maps')
    os.makedirs(mask_path, exist_ok=True)

    image_pairs = zip(imgs1, imgs2)
    for img1_name, img2_name in tqdm.tqdm(
        image_pairs, desc='Processing images', total=len(imgs1)
    ):
        input_img1 = os.path.join(merge_path, img1_name)
        input_img2 = os.path.join(output_common, img2_name)
        output_img = os.path.join(output_path, img1_name)

        img1 = Image.open(input_img1).convert('RGB')
        img2 = Image.open(input_img2).convert('RGB').resize(
            img1.size, Image.Resampling.LANCZOS
        )
        arr1 = np.asarray(img1, dtype=np.float32)
        arr2 = np.asarray(img2, dtype=np.float32)

        tensor_img1 = (
            torch.from_numpy(arr1)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        tensor_img1 = transform_batch(tensor_img1).to(device)

        with torch.no_grad():
            output = net(tensor_img1)
        parsing = output.squeeze(0).cpu().numpy().argmax(0)

        image_stem = osp.splitext(img1_name)[0]
        mask_save_path = osp.join(mask_path, f'{image_stem}_parsing.png')
        image_size = (arr1.shape[1], arr1.shape[0])
        vis_parsing_maps(
            parsing,
            stride=1,
            save_im=True,
            save_path=mask_save_path,
            img_size=image_size,
        )

        mask = np.zeros_like(parsing, dtype=np.uint8)
        for class_id in range(1, 15):
            mask[parsing == class_id] = 1
        mask = np.expand_dims(mask, axis=2)

        merged = np.clip(arr1 * mask + arr2 * (1 - mask), 0, 255)
        Image.fromarray(merged.astype(np.uint8)).save(output_img)


def vis_parsing_maps(
    parsing_anno,
    stride,
    save_im=True,
    save_path='vis_results/parsing_map_on_im.jpg',
    img_size=(512, 512),
):
    """Save a color visualization of a face parsing map."""
    color_map = np.array(
        [[255, 255, 255]] + [[255, 0, 0]] * 14 + [[0, 255, 0]] * 4,
        dtype=np.uint8,
    )

    parsing_map = cv2.resize(
        parsing_anno.astype(np.uint8),
        None,
        fx=stride,
        fy=stride,
        interpolation=cv2.INTER_NEAREST,
    )
    parsing_map_color = color_map[parsing_map]

    visualization = cv2.resize(
        parsing_map_color, img_size, interpolation=cv2.INTER_NEAREST
    )
    if save_im:
        cv2.imwrite(save_path, visualization)


def sr_forward(imgs_path, jpeg_resize_path, save_folder, alpha):
    face_queue = Queue()
    common_queue = Queue()

    face_process = Process(
        target=face_sr,
        args=(jpeg_resize_path, save_folder, alpha, face_queue),
    )
    common_process = Process(
        target=common_sr,
        args=(imgs_path, save_folder, common_queue),
    )
    face_process.start()
    common_process.start()
    face_process.join()
    common_process.join()

    if face_process.exitcode or common_process.exitcode:
        raise RuntimeError(
            f'SR process failed: face={face_process.exitcode}, '
            f'common={common_process.exitcode}'
        )

    print('SR processes completed. Combining results...')
    merge_path = face_queue.get()
    output_common = common_queue.get()
    output_path = os.path.join(save_folder, 'cut_imgs')
    cut_sr(merge_path, output_common, output_path)
    print(f'Processing completed. Results saved to {output_path}')


def main(ori_imgs, save_folder, quality, re_size, alpha=0.8):
    jpeg_path = utils.jpeg_folder(
        ori_imgs_path=ori_imgs,
        save_path=save_folder,
        quality=quality,
    )
    jpeg_resize_path = os.path.join(save_folder, 'jpeg_resize')
    utils.resize2_other_folder(
        image_path=jpeg_path,
        save_path=jpeg_resize_path,
        target_size=re_size,
    )
    sr_forward(ori_imgs, jpeg_resize_path, save_folder, alpha)


def args_parse():
    parser = argparse.ArgumentParser(
        description='Restore images with GFPGAN, Real-ESRGAN, and BiSeNet.'
    )
    parser.add_argument('--ori_imgs', required=True, help='Input image folder')
    parser.add_argument('--save_folder', required=True, help='Output folder')
    parser.add_argument(
        '--jpeg_quality', type=int, default=75, help='JPEG quality (default: 75)'
    )
    parser.add_argument(
        '--re_size', type=int, default=512, help='Intermediate image size (default: 512)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='Face merge weight (default: 0.8)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parse()
    main(
        args.ori_imgs,
        args.save_folder,
        args.jpeg_quality,
        args.re_size,
        args.alpha,
    )
