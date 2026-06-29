"""
Restormer: Efficient Transformer for High-Resolution Image Restoration
Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
https://arxiv.org/abs/2111.09881

Python port of the original MATLAB evaluation script.
Computes PSNR and SSIM (on the Y/luma channel) between restored images
and their ground-truth targets across several deraining datasets.

SSIM implementation follows:
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
"Image quality assessment: From error measurement to structural similarity"
IEEE Transactions on Image Processing, vol. 13, no. 1, Jan. 2004.
"""

import os
import glob
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.ndimage import convolve
from skimage import io
from skimage.color import rgb2ycbcr

# ----------------------------------------------------------------------
# Datasets to evaluate (mirrors the MATLAB `datasets` cell array)
# ----------------------------------------------------------------------
# datasets = ['Rain100L']
DATASETS = ['Test100', 'Rain100H', 'Rain100L', 'Test2800', 'Test1200']

NUM_WORKERS = 20  # mirrors parpool('local', 20)


def fspecial_gaussian(shape=(11, 11), sigma=1.5):
    """
    Equivalent of MATLAB's fspecial('gaussian', shape, sigma).
    Returns a normalized 2D Gaussian window.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2_valid(window, img):
    """
    Equivalent of MATLAB's filter2(window, img, 'valid').
    filter2 performs correlation (not convolution), and MATLAB's
    'valid' mode returns only the region computed without zero-padding.
    """
    # filter2(B, A) == convolve2d(A, rot90(B, 2)) in 'same'/'valid' modes.
    # Using scipy.ndimage.convolve with the kernel flipped reproduces
    # MATLAB's filter2 (which is itself a correlation), restricted to
    # the 'valid' region.
    kernel = np.flipud(np.fliplr(window))
    full = convolve(img, kernel, mode='constant', cval=0.0)

    kh, kw = window.shape
    ih, iw = img.shape
    # 'valid' output size: (ih-kh+1, iw-kw+1)
    out_h = ih - kh + 1
    out_w = iw - kw + 1
    if out_h <= 0 or out_w <= 0:
        return np.zeros((0, 0))

    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    # scipy's 'constant' convolve output is aligned the same way as a
    # 'same'-mode filter; crop to the 'valid' region from the center.
    start_h = kh // 2
    start_w = kw // 2
    return full[start_h:start_h + out_h, start_w:start_w + out_w]


def ssim_index(img1, img2, K=(0.01, 0.03), window=None, L=255):
    """
    Direct port of Zhou Wang's SSIM_index.m (2003).

    Input:
        img1, img2 : 2D grayscale images (numpy arrays), same size
        K          : tuple of two constants (default (0.01, 0.03))
        window     : 2D local window for statistics (default: 11x11 Gaussian, sigma=1.5)
        L          : dynamic range of the images (default 255)

    Output:
        mssim    : mean SSIM index value between the two images
        ssim_map : the SSIM index map
    """
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    if img1.shape != img2.shape:
        return -np.inf, -np.inf

    M, N = img1.shape

    if window is None:
        if M < 11 or N < 11:
            return -np.inf, -np.inf
        window = fspecial_gaussian((11, 11), 1.5)

    H, W = window.shape
    if (H * W) < 4 or H > M or W > N:
        return -np.inf, -np.inf

    if K[0] < 0 or K[1] < 0:
        return -np.inf, -np.inf

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    window = window / window.sum()

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = filter2_valid(window, img1)
    mu2 = filter2_valid(window, img2)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter2_valid(window, img1 * img1) - mu1_sq
    sigma2_sq = filter2_valid(window, img2 * img2) - mu2_sq
    sigma12 = filter2_valid(window, img1 * img2) - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones_like(mu1)
        index = denominator1 * denominator2 > 0
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (
            denominator1[index] * denominator2[index]
        )
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = ssim_map.mean()
    return mssim, ssim_map


def to_y_channel(img):
    """
    Converts an RGB image to its Y (luma) channel via YCbCr,
    matching MATLAB's rgb2ycbcr(img); img(:,:,1).
    Grayscale images are returned unchanged.
    """
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 3:
        ycbcr = rgb2ycbcr(img)
        y = ycbcr[:, :, 0]
    else:
        y = img
    return y


def compute_ssim(img1, img2):
    y1 = to_y_channel(img1)
    y2 = to_y_channel(img2)
    mssim, _ = ssim_index(y1, y2)
    return mssim


def compute_psnr(img1, img2):
    y1 = to_y_channel(img1).astype(np.float64)
    y2 = to_y_channel(img2).astype(np.float64)

    imdff = y1 - y2
    rmse = np.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / rmse)
    return psnr


def _process_pair(args):
    """Worker function for multiprocessing: loads a pair and computes metrics."""
    input_path, gt_path = args
    input_img = io.imread(input_path)
    gt_img = io.imread(gt_path)
    ssim_val = compute_ssim(input_img, gt_img)
    psnr_val = compute_psnr(input_img, gt_img)
    return psnr_val, ssim_val


def list_images(folder):
    """Mirrors: [dir(*.jpg); dir(*.png)], sorted for deterministic pairing."""
    files = sorted(glob.glob(os.path.join(folder, '*.jpg')) +
                   glob.glob(os.path.join(folder, '*.png')))
    return files


def main():
    start_time = time.time()

    psnr_alldatasets = 0.0
    ssim_alldatasets = 0.0
    num_set = len(DATASETS)

    n_workers = min(NUM_WORKERS, cpu_count())

    for dataset in DATASETS:
        file_path = os.path.join('.', 'results', dataset, '')
        gt_path = os.path.join('.', 'Datasets', 'test', dataset, 'target', '')

        path_list = list_images(file_path)
        gt_list = list_images(gt_path)
        img_num = len(path_list)

        total_psnr = 0.0
        total_ssim = 0.0

        if img_num > 0:
            pairs = list(zip(path_list, gt_list[:img_num]))
            with Pool(processes=n_workers) as pool:
                results = pool.map(_process_pair, pairs)

            for psnr_val, ssim_val in results:
                total_psnr += psnr_val
                total_ssim += ssim_val

        qm_psnr = total_psnr / img_num if img_num > 0 else 0.0
        qm_ssim = total_ssim / img_num if img_num > 0 else 0.0

        print(f'For {dataset} dataset PSNR: {qm_psnr:.6f} SSIM: {qm_ssim:.6f}')

        psnr_alldatasets += qm_psnr
        ssim_alldatasets += qm_ssim

    print(f'For all datasets PSNR: {psnr_alldatasets / num_set:.6f} '
          f'SSIM: {ssim_alldatasets / num_set:.6f}')

    elapsed = time.time() - start_time
    print(f'Elapsed time: {elapsed:.2f} seconds')


if __name__ == '__main__':
    main()