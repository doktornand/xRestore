#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     ARCHIVAL RESTORATION SUITE — Version GUI v4.0                           ║
║     Laboratoire de Restauration Numérique — Interface Tkinter                ║
║                                                                              ║
║     Machine à bidouiller les photos avec prévisualisation en temps réel!     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage: python archival_suite_gui.py
"""
import argparse
import os
import sys
import time
import json
import warnings
import threading
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener, convolve2d

try:
    from skimage import restoration, filters, morphology, measure, segmentation, feature, exposure
    from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
    from skimage.filters import threshold_multiotsu, threshold_sauvola
    from skimage.morphology import remove_small_objects, remove_small_holes
    from skimage.measure import label, regionprops
    from skimage.segmentation import felzenszwalb, slic, watershed
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.exposure import match_histograms, equalize_adapthist
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA, NMF
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import traceback

# =============================================================================
# UTILITAIRES
# =============================================================================
def to_float(img): return img.astype(np.float32) / 255.0
def to_uint8(img_f): return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)

# =============================================================================
# CONSTANTES & RAPPORT
# =============================================================================
class DegradationType(Enum):
    FADING = auto(); YELLOWING = auto(); SILVERING = auto(); FOXING = auto()
    MOLD = auto(); WATER_DAMAGE = auto(); CRACKS = auto(); TEARS = auto()
    SCRATCHES = auto(); DUST = auto(); VIGNETTING = auto(); BLUR = auto()
    NOISE = auto(); COLOR_SHIFT = auto(); BANDING = auto(); HALATION = auto()
    OXIDATION = auto(); SILVER_MIGRATION = auto()

@dataclass
class RestorationReport:
    input_path: str = ""; output_path: str = ""
    original_dimensions: Tuple[int, int] = (0, 0); final_dimensions: Tuple[int, int] = (0, 0)
    detected_degradations: List[str] = field(default_factory=list)
    applied_steps: List[Dict] = field(default_factory=list)
    spectral_analysis: Dict = field(default_factory=dict)
    quality_metrics: Dict = field(default_factory=dict)
    processing_time: float = 0.0
    def to_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=str)

# =============================================================================
# MODULES DE TRAITEMENT (identiques à la version console)
# =============================================================================
class DegradationAnalyzer:
    def __init__(self, img: np.ndarray):
        self.img = to_float(img)
        self.gray = cv2.cvtColor((self.img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        self.h, self.w = img.shape[:2]
        self.degradations = []
    def analyze_all(self) -> List[DegradationType]:
        self._detect_fading(); self._detect_yellowing(); self._detect_noise()
        self._detect_blur(); self._detect_scratches(); self._detect_foxing()
        self._detect_vignetting(); self._detect_banding(); self._detect_color_shift()
        return self.degradations
    def _detect_fading(self):
        if self.gray.mean() > 0.65 and self.gray.std() < 0.15:
            self.degradations.append(DegradationType.FADING)
    def _detect_yellowing(self):
        r, g, b = self.img[:,:,0], self.img[:,:,1], self.img[:,:,2]
        if ((r + g)/2 - b).mean() > 0.08:
            self.degradations.append(DegradationType.YELLOWING)
    def _detect_noise(self):
        if SKIMAGE_AVAILABLE:
            sigma = estimate_sigma(self.img, channel_axis=-1, average_sigmas=True)
            if sigma > 0.025: self.degradations.append(DegradationType.NOISE)
    def _detect_blur(self):
        sx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        if np.sqrt(sx**2 + sy**2).mean() < 8.0:
            self.degradations.append(DegradationType.BLUR)
    def _detect_scratches(self):
        edges = cv2.Canny((self.gray * 255).astype(np.uint8), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=5)
        if lines is not None and len(lines) > 20:
            self.degradations.append(DegradationType.SCRATCHES)
    def _detect_foxing(self):
        hsv = cv2.cvtColor((self.img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([10, 40, 50]), np.array([30, 180, 200]))
        if mask.sum() / (mask.shape[0] * mask.shape[1] * 255) > 0.02:
            self.degradations.append(DegradationType.FOXING)
    def _detect_vignetting(self):
        cm = self.gray[self.h//3:2*self.h//3, self.w//3:2*self.w//3].mean()
        corners = [self.gray[:self.h//4, :self.w//4].mean(),
                   self.gray[:self.h//4, -self.w//4:].mean(),
                   self.gray[-self.h//4:, :self.w//4].mean(),
                   self.gray[-self.h//4:, -self.w//4:].mean()]
        if cm - np.mean(corners) > 0.15:
            self.degradations.append(DegradationType.VIGNETTING)
    def _detect_banding(self):
        row_fft = np.abs(np.fft.fft(self.gray.mean(axis=1)))
        peaks = np.where(row_fft[1:len(row_fft)//2] > row_fft[1:len(row_fft)//2].mean() * 3)[0]
        if len(peaks) > 3: self.degradations.append(DegradationType.BANDING)
    def _detect_color_shift(self):
        means = [self.img[:,:,i].mean() for i in range(3)]
        if max(means) - min(means) > 0.12:
            self.degradations.append(DegradationType.COLOR_SHIFT)

class SpectralRestorer:
    @staticmethod
    def homomorphic_filter(img: np.ndarray, cutoff: float = 30, order: int = 2,
                          low_gain: float = 0.5, high_gain: float = 2.0) -> np.ndarray:
        log_img = np.log1p(img.astype(np.float32))
        for c in range(3):
            f = np.fft.fft2(log_img[:,:,c])
            fshift = np.fft.fftshift(f)
            rows, cols = img.shape[:2]
            crow, ccol = rows//2, cols//2
            Y, X = np.ogrid[:rows, :cols]
            D = np.sqrt((X-ccol)**2 + (Y-crow)**2)
            D[D == 0] = 1e-5
            H = (high_gain - low_gain) * (1 - 1/(1 + (D/cutoff)**(2*order))) + low_gain
            fshift *= H
            log_img[:,:,c] = np.real(np.fft.ifft2(np.fft.ifftshift(fshift)))
        return np.clip(np.expm1(log_img), 0, 255).astype(np.uint8)
    @staticmethod
    def phase_congruency_sharpen(img: np.ndarray) -> np.ndarray:
        if not SKIMAGE_AVAILABLE: return img
        from skimage.filters import sobel
        result = img.copy().astype(np.float32)
        for c in range(3):
            ch = img[:,:,c].astype(np.float32) / 255.0
            sh, sv = sobel(ch, axis=0), sobel(ch, axis=1)
            amp = np.sqrt(sh**2 + sv**2) + 1e-5
            pc = np.abs(sh + sv) / amp
            result[:,:,c] = np.clip(ch * (1 + 0.5 * pc) * 255, 0, 255)
        return result.astype(np.uint8)
    @staticmethod
    def anisotropic_diffusion(img: np.ndarray, n_iter: int = 15,
                             kappa: float = 30, gamma: float = 0.1) -> np.ndarray:
        result = img.astype(np.float32).copy()
        for _ in range(n_iter):
            for c in range(3):
                I = result[:,:,c]
                dN, dS = np.roll(I, -1, axis=0) - I, np.roll(I, 1, axis=0) - I
                dE, dW = np.roll(I, -1, axis=1) - I, np.roll(I, 1, axis=1) - I
                cN = np.exp(-(dN/kappa)**2); cS = np.exp(-(dS/kappa)**2)
                cE = np.exp(-(dE/kappa)**2); cW = np.exp(-(dW/kappa)**2)
                I += gamma * (cN*dN + cS*dS + cE*dE + cW*dW)
                result[:,:,c] = I
        return np.clip(result, 0, 255).astype(np.uint8)

class InpaintingEngine:
    @staticmethod
    def telea_inpaint(img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        m = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        return cv2.cvtColor(cv2.inpaint(bgr, m, radius, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
    @staticmethod
    def ns_inpaint(img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        m = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        return cv2.cvtColor(cv2.inpaint(bgr, m, radius, cv2.INPAINT_NS), cv2.COLOR_BGR2RGB)
    @staticmethod
    def exemplar_based_inpaint(img: np.ndarray, mask: np.ndarray, patch_size: int = 11) -> np.ndarray:
        result = img.copy().astype(np.float32)
        mask_bool = mask > 0.5 if mask.max() <= 1 else mask > 127
        from scipy.ndimage import binary_dilation
        front = binary_dilation(mask_bool) & ~mask_bool
        for _ in range(500):
            if not front.any(): break
            yx = np.argwhere(front)
            if len(yx) == 0: break
            py, px = yx[np.random.randint(len(yx))]
            half = patch_size // 2
            y1, y2 = max(0, py-half), min(img.shape[0], py+half+1)
            x1, x2 = max(0, px-half), min(img.shape[1], px+half+1)
            patch = result[y1:y2, x1:x2]
            pm = mask_bool[y1:y2, x1:x2]
            best_ssd, best_match = float('inf'), None
            sy1, sy2 = max(0, py-50), min(img.shape[0]-(y2-y1), py+50)
            sx1, sx2 = max(0, px-50), min(img.shape[1]-(x2-x1), px+50)
            for sy in range(sy1, sy2, 3):
                for sx in range(sx1, sx2, 3):
                    if mask_bool[sy:sy+(y2-y1), sx:sx+(x2-x1)].any(): continue
                    cand = result[sy:sy+(y2-y1), sx:sx+(x2-x1)]
                    if cand.shape != patch.shape: continue
                    diff = (cand - patch)**2
                    ssd = np.sum(diff * ~pm[:diff.shape[0], :diff.shape[1]])
                    if ssd < best_ssd: best_ssd, best_match = ssd, cand
            if best_match is not None:
                fm = pm[:best_match.shape[0], :best_match.shape[1]]
                patch[fm] = best_match[fm]
                result[y1:y2, x1:x2] = patch
        return np.clip(result, 0, 255).astype(np.uint8)

class ColorScience:
    @staticmethod
    def gray_world(img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32)
        avgs = [img_f[:,:,i].mean() for i in range(3)]
        avg_gray = sum(avgs) / 3.0
        scale = np.array([avg_gray/a for a in avgs])
        return np.clip(img_f * scale.reshape(1,1,3), 0, 255).astype(np.uint8)
    @staticmethod
    def white_patch(img: np.ndarray, percentile: float = 99.9) -> np.ndarray:
        img_f = img.astype(np.float32)
        maxs = [np.percentile(img_f[:,:,i], percentile) for i in range(3)]
        max_val = max(maxs)
        scale = np.array([max_val/m for m in maxs])
        return np.clip(img_f * scale.reshape(1,1,3), 0, 255).astype(np.uint8)
    @staticmethod
    def shades_of_gray(img: np.ndarray, p: float = 6.0) -> np.ndarray:
        img_f = to_float(img)
        avgs = [np.mean(img_f[:,:,i]**p)**(1/p) for i in range(3)]
        avg_gray = sum(avgs) / 3.0
        scale = np.array([avg_gray/a for a in avgs])
        return to_uint8(np.clip(img_f * scale.reshape(1,1,3), 0, 1))
    @staticmethod
    def color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        if SKIMAGE_AVAILABLE:
            return match_histograms(source, target, channel_axis=-1).astype(np.uint8)
        return source
    @staticmethod
    def recover_faded(img: np.ndarray) -> np.ndarray:
        img_f = to_float(img)
        means = [img_f[:,:,i].mean() for i in range(3)]
        target = 0.5
        for i in range(3):
            if means[i] < target:
                factor = (target / means[i]) ** 0.7
                img_f[:,:,i] = np.power(img_f[:,:,i], 1.0 / factor)
        return to_uint8(np.clip(img_f, 0, 1))
    @staticmethod
    def remove_yellowing(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        img_f = to_float(img)
        yellow = (img_f[:,:,0] + img_f[:,:,1])/2 - img_f[:,:,2]
        yellow = np.clip(yellow, 0, 1)
        img_f[:,:,0] -= yellow * strength * 0.3
        img_f[:,:,1] -= yellow * strength * 0.3
        img_f[:,:,2] += yellow * strength * 0.1
        return to_uint8(np.clip(img_f, 0, 1))

class DeconvolutionRestorer:
    @staticmethod
    def estimate_psf_blind(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        f_gray = np.fft.fft2(gray)
        autocorr = np.fft.fftshift(np.abs(np.fft.ifft2(f_gray * np.conj(f_gray))))
        cy, cx = autocorr.shape[0]//2, autocorr.shape[1]//2
        c = kernel_size // 2
        kernel = autocorr[cy-c:cy+c+1, cx-c:cx+c+1]
        return kernel / kernel.sum()
    @staticmethod
    def richardson_lucy(img: np.ndarray, psf: np.ndarray, iterations: int = 10) -> np.ndarray:
        img_f = to_float(img) + 1e-8
        psf = psf / psf.sum()
        pad_h = (img.shape[0] - psf.shape[0]) // 2
        pad_w = (img.shape[1] - psf.shape[1]) // 2
        psf_p = np.pad(psf, ((pad_h, pad_h), (pad_w, pad_w)))
        result = img_f.copy()
        for _ in range(iterations):
            for c in range(3):
                conv = signal.fftconvolve(result[:,:,c], psf_p, mode='same')
                conv = np.clip(conv, 1e-8, None)
                rel = img_f[:,:,c] / conv
                result[:,:,c] *= signal.fftconvolve(rel, psf_p[::-1, ::-1], mode='same')
        return to_uint8(np.clip(result, 0, 1))

class SuperResolution:
    @staticmethod
    def lanczos_upscale(img: np.ndarray, scale: int = 2) -> np.ndarray:
        pil = Image.fromarray(img)
        new_size = (img.shape[1] * scale, img.shape[0] * scale)
        return np.array(pil.resize(new_size, Image.LANCZOS))
    @staticmethod
    def iterative_back_projection(img: np.ndarray, scale: int = 2, iterations: int = 5) -> np.ndarray:
        hr = SuperResolution.lanczos_upscale(img, scale)
        hr_f = to_float(hr)
        kernel = cv2.getGaussianKernel(5, 1.0) @ cv2.getGaussianKernel(5, 1.0).T
        for _ in range(iterations):
            lr_sim = cv2.resize(hr_f, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            lr_sim = cv2.filter2D(lr_sim, -1, kernel)
            error = to_float(img) - lr_sim
            error_up = cv2.resize(error, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            hr_f = np.clip(hr_f + 0.5 * error_up, 0, 1)
        return to_uint8(hr_f)
    @staticmethod
    def edge_directed(img: np.ndarray, scale: int = 2) -> np.ndarray:
        h, w = img.shape[:2]
        result = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sx, sy = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = cv2.resize(np.sqrt(sx**2 + sy**2), (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
        mask = edge_map > np.percentile(edge_map, 75)
        for c in range(3):
            ch = result[:,:,c]
            filtered = cv2.bilateralFilter(ch.astype(np.uint8), 5, 50, 50).astype(np.float32)
            result[:,:,c] = np.where(mask, filtered, ch)
        return np.clip(result, 0, 255).astype(np.uint8)

class NoiseReduction:
    @staticmethod
    def bm3d(img: np.ndarray, sigma: float = 25) -> np.ndarray:
        if not SKIMAGE_AVAILABLE: return NoiseReduction.nlm(img, sigma)
        try:
            from skimage.restoration import denoise_bm3d
            return to_uint8(np.clip(denoise_bm3d(to_float(img), sigma=sigma/255.0), 0, 1))
        except: return NoiseReduction.nlm(img, sigma)
    @staticmethod
    def nlm(img: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        if SKIMAGE_AVAILABLE:
            img_f = to_float(img)
            if sigma is None: sigma = estimate_sigma(img_f, channel_axis=-1, average_sigmas=True)
            return to_uint8(np.clip(denoise_nl_means(img_f, patch_size=7, patch_distance=11,
                                                      channel_axis=-1, sigma=sigma, h=0.8*sigma), 0, 1))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21), cv2.COLOR_BGR2RGB)
    @staticmethod
    def wavelet(img: np.ndarray, sigma: Optional[float] = None, wavelet: str = 'db8') -> np.ndarray:
        if not SKIMAGE_AVAILABLE: return NoiseReduction.nlm(img, sigma)
        return to_uint8(np.clip(denoise_wavelet(to_float(img), channel_axis=-1, convert2ycbcr=True,
                                                wavelet=wavelet, rescale_sigma=True), 0, 1))
    @staticmethod
    def bilateral_stack(img: np.ndarray, sigma_spatial: float = 5,
                       sigma_color: float = 0.1, iterations: int = 3) -> np.ndarray:
        result = img.copy().astype(np.float32)
        for _ in range(iterations):
            for c in range(3):
                result[:,:,c] = cv2.bilateralFilter(result[:,:,c].astype(np.uint8),
                                                   int(sigma_spatial), int(sigma_color*255), int(sigma_spatial))
        return result.astype(np.uint8)

class ContrastEnhancer:
    @staticmethod
    def clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)
    @staticmethod
    def retinex_msrcr(img: np.ndarray, sigmas: List[float] = [15, 80, 250]) -> np.ndarray:
        img_f = img.astype(np.float32) + 1.0
        def ssr(ch, sigma):
            return np.log10(ch) - np.log10(cv2.GaussianBlur(ch, (0,0), sigma) + 1.0)
        msrcr = np.zeros_like(img_f)
        for i in range(3):
            ch = img_f[:,:,i]
            rx = sum(ssr(ch, s) for s in sigmas) / len(sigmas)
            msrcr[:,:,i] = rx
        sum_ch = img_f.sum(axis=2)
        cr = 46 * (np.log10(125 * img_f) - np.log10(sum_ch[:,:,None]))
        msrcr = 5 * (msrcr * cr + 25)
        msrcr = np.power(10, msrcr)
        for i in range(3):
            msrcr[:,:,i] = (msrcr[:,:,i] - msrcr[:,:,i].min()) / (msrcr[:,:,i].max() - msrcr[:,:,i].min() + 1e-8) * 255
        return np.clip(msrcr, 0, 255).astype(np.uint8)
    @staticmethod
    def reinhard_tmo(img: np.ndarray, gamma: float = 2.2, light_adapt: float = 0.8) -> np.ndarray:
        img_f = to_float(img)
        lum = 0.2126*img_f[:,:,0] + 0.7152*img_f[:,:,1] + 0.0722*img_f[:,:,2]
        lum = np.clip(lum, 1e-6, 1.0)
        lum_tmo = lum / (1.0 + lum)
        if light_adapt > 0:
            lum_blur = cv2.GaussianBlur(lum, (0,0), 32)
            lum_tmo = lum_tmo * (1 - light_adapt) + (lum / (1.0 + lum_blur)) * light_adapt
        ratio = lum_tmo / lum
        result = img_f * ratio[:,:,None]
        return to_uint8(np.clip(np.power(result, 1.0/gamma), 0, 1))

class GeometryCorrector:
    @staticmethod
    def auto_straighten(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        if lines is None: return img
        angles = [np.degrees(np.arctan2(y2-y1, x2-x1)) for line in lines for x1,y1,x2,y2 in [line[0]] if x2!=x1 and abs(np.degrees(np.arctan2(y2-y1, x2-x1))) < 20]
        if not angles: return img
        angle = np.median(angles)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        cos, sin = np.abs(M[0,0]), np.abs(M[0,1])
        new_w, new_h = int(h*sin + w*cos), int(h*cos + w*sin)
        M[0,2] += (new_w/2) - w//2; M[1,2] += (new_h/2) - h//2
        return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC)
    @staticmethod
    def lens_distortion(img: np.ndarray, k1: float = -0.2) -> np.ndarray:
        h, w = img.shape[:2]
        K = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
        D = np.array([k1, 0, 0, 0, 0], dtype=np.float32)
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
        return cv2.remap(img, m1, m2, cv2.INTER_CUBIC)

class DamageRestorer:
    @staticmethod
    def remove_foxing(img: np.ndarray, aggressiveness: float = 0.7) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([10, 30, 50]), np.array([35, 200, 220]))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        return InpaintingEngine.telea_inpaint(img, mask, 7) if mask.sum() > 0 else img
    @staticmethod
    def remove_mold(img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 30]), np.array([85, 255, 200]))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return InpaintingEngine.ns_inpaint(img, mask, 9) if mask.sum() > 0 else img
    @staticmethod
    def repair_cracks(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        mask = np.zeros_like(dark)
        for i in range(1, n):
            w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
            if area < 500 and max(w,h) / (min(w,h)+1e-5) > 3:
                mask[labels == i] = 255
        return InpaintingEngine.telea_inpaint(img, mask, 5) if mask.sum() > 0 else img
    @staticmethod
    def repair_tears(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        v = cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
        h = cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)))
        mask = cv2.bitwise_or(v, h)
        return InpaintingEngine.exemplar_based_inpaint(img, mask) if mask.sum() > 0 else img
    @staticmethod
    def remove_water_damage(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        local_var = cv2.blur(gray**2, (15,15)) - cv2.blur(gray, (15,15))**2
        _, mask = cv2.threshold(local_var.astype(np.uint8), int(np.percentile(local_var, 85)), 255, cv2.THRESH_BINARY)
        result = img.astype(np.float32)
        mask_f = mask.astype(np.float32) / 255.0
        smoothed = cv2.bilateralFilter(img, 15, 100, 100).astype(np.float32)
        return np.clip(result * (1 - mask_f[:,:,None]) + smoothed * mask_f[:,:,None], 0, 255).astype(np.uint8)
    @staticmethod
    def remove_silvering(img: np.ndarray) -> np.ndarray:
        img_f = to_float(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        mask = (hsv[:,:,2] > 0.7) & (hsv[:,:,1] < 0.3)
        result = img_f.copy()
        result[mask] *= 0.7
        return to_uint8(np.clip(result, 0, 1))
    @staticmethod
    def daguerreotype_restore(img: np.ndarray, oxidation: bool = True, tarnish: bool = True) -> np.ndarray:
        result = img.astype(np.float32)
        if oxidation:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (51, 51), 0)
            diff = gray.astype(np.float32) - blur.astype(np.float32)
            ox_mask = diff < -20
            corrected = cv2.createCLAHE(2.0, (8,8)).apply(gray)
            mask_f = ox_mask.astype(np.float32)
            result_gray = gray * (1 - mask_f) + corrected * mask_f
            ratio = result_gray / (gray.astype(np.float32) + 1e-5)
            for c in range(3): result[:,:,c] *= ratio
        if tarnish:
            result = 255 - result
            result = cv2.GaussianBlur(result, (3, 3), 0)
            result = 255 - result
        return np.clip(result, 0, 255).astype(np.uint8)

class VintageEffects:
    @staticmethod
    def vignette(img: np.ndarray, strength: float = 0.5, shape: float = 2.0) -> np.ndarray:
        if strength <= 0: return img
        h, w = img.shape[:2]
        cx, cy = w/2, h/2
        Y, X = np.ogrid[:h, :w]
        dist = np.clip(np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2), 0, 1)
        mask = np.clip(1.0 - strength * (dist ** shape), 0, 1)
        img_f = to_float(img)
        for c in range(3): img_f[:,:,c] *= mask
        return to_uint8(img_f)
    @staticmethod
    def film_grain(img: np.ndarray, sigma: float = 0.03, color: bool = False) -> np.ndarray:
        if sigma <= 0: return img
        img_f = to_float(img)
        if color:
            noise = np.random.normal(0, sigma, img_f.shape).astype(np.float32)
        else:
            noise_lum = np.random.normal(0, sigma, img_f.shape[:2]).astype(np.float32)
            noise = np.stack([noise_lum]*3, axis=-1)
        return to_uint8(np.clip(img_f + noise, 0, 1))
    @staticmethod
    def halation(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
        if strength <= 0: return img
        img_f = to_float(img)
        lum = 0.299*img_f[:,:,0] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,2]
        highlights = np.clip((lum - 0.75) * 4.0, 0, 1)
        halo = cv2.GaussianBlur((highlights * 255).astype(np.uint8), (0,0), 10).astype(np.float32) / 255.0
        img_f[:,:,0] = np.clip(img_f[:,:,0] + halo * strength * 0.5, 0, 1)
        img_f[:,:,2] = np.clip(img_f[:,:,2] - halo * strength * 0.2, 0, 1)
        return to_uint8(img_f)

class QualityMetrics:
    @staticmethod
    def sharpness(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    @staticmethod
    def contrast(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray.std()
    @staticmethod
    def colorfulness(img: np.ndarray) -> float:
        rg = np.abs(img[:,:,0].astype(np.float32) - img[:,:,1].astype(np.float32))
        yb = np.abs(0.5*(img[:,:,0].astype(np.float32) + img[:,:,1].astype(np.float32)) - img[:,:,2].astype(np.float32))
        return np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)
    @staticmethod
    def snr(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        signal_power = gray.mean() ** 2
        noise_power = np.var(gray - cv2.GaussianBlur(gray, (5,5), 0))
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
    @staticmethod
    def evaluate(img: np.ndarray) -> Dict:
        return {"sharpness": QualityMetrics.sharpness(img), "contrast": QualityMetrics.contrast(img),
                "colorfulness": QualityMetrics.colorfulness(img), "snr_db": QualityMetrics.snr(img)}

# =============================================================================
# INTERFACE GRAPHIQUE PRINCIPALE
# =============================================================================
class ArchivalRestorationGUI:
    """Application GUI complète de restauration d'images."""

    def __init__(self, root):
        self.root = root
        self.root.title("Archival Restoration Suite v4.0 — Machine à bidouiller les photos")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # État
        self.original_image = None  # numpy RGB original
        self.current_image = None   # numpy RGB courant
        self.preview_image = None   # numpy RGB pour preview
        self.current_file_path = None
        self.is_processing = False
        self.undo_stack = []
        self.redo_stack = []
        self.preview_after_id = None

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2d2d2d')
        style.configure('TLabel', background='#2d2d2d', foreground='#e0e0e0', font=('Segoe UI', 9))
        style.configure('TButton', font=('Segoe UI', 9), padding=4)
        style.configure('TCheckbutton', background='#2d2d2d', foreground='#e0e0e0')
        style.configure('TLabelframe', background='#2d2d2d', foreground='#e0e0e0')
        style.configure('TLabelframe.Label', background='#2d2d2d', foreground='#a0a0a0', font=('Segoe UI', 9, 'bold'))
        style.configure('TNotebook', background='#2d2d2d')
        style.configure('TNotebook.Tab', background='#3d3d3d', foreground='#e0e0e0', padding=[8, 4])
        style.map('TNotebook.Tab', background=[('selected', '#505050')])

        # Widgets de contrôles
        self.controls = {}

        self._build_ui()

    def _build_ui(self):
        """Construction complète de l'interface."""
        self.root.configure(background='#2d2d2d')

        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panneau gauche (contrôles)
        left_pane = ttk.Frame(main_frame)
        left_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_pane.configure(width=380)
        left_pane.pack_propagate(False)

        # Panneau droit (prévisualisation)
        right_pane = ttk.Frame(main_frame)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_toolbar(right_pane)
        self._build_canvas(right_pane)
        self._build_statusbar(right_pane)
        self._build_controls(left_pane)

    def _build_toolbar(self, parent):
        """Barre d'outils avec boutons principaux."""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 5))

        btn_style = {'width': 12}

        ttk.Button(toolbar, text="📂 Ouvrir", command=self.load_image, **btn_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Sauvegarder", command=self.save_image, **btn_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Sauvegarder sous...", command=self.save_image_as, **btn_style).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(toolbar, text="↩ Annuler", command=self.undo, **btn_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="↪ Refaire", command=self.redo, **btn_style).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(toolbar, text="🔄 Réinitialiser", command=self.reset_image, **btn_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⚡ Appliquer", command=self.apply_all_changes, **btn_style).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(toolbar, text="📊 Métriques", command=self.show_metrics, **btn_style).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔬 Analyse", command=self.run_analysis, **btn_style).pack(side=tk.LEFT, padx=2)

    def _build_canvas(self, parent):
        """Zone de prévisualisation avec canvas et scrollbars."""
        canvas_frame = ttk.LabelFrame(parent, text="Prévisualisation")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='#1a1a1a', highlightthickness=0)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Zoom controls
        zoom_frame = ttk.Frame(canvas_frame)
        zoom_frame.pack(fill=tk.X, pady=2)

        ttk.Button(zoom_frame, text="➖", command=lambda: self.zoom(-0.1), width=3).pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="➕", command=lambda: self.zoom(0.1), width=3).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Ajuster", command=self.fit_to_window, width=8).pack(side=tk.RIGHT)

        self.zoom_factor = 1.0
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def _build_statusbar(self, parent):
        """Barre d'état."""
        self.status_bar = ttk.Frame(parent)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))

        self.status_label = ttk.Label(self.status_bar, text="Prêt — Chargez une image pour commencer")
        self.status_label.pack(side=tk.LEFT)

        self.info_label = ttk.Label(self.status_bar, text="")
        self.info_label.pack(side=tk.RIGHT)

    def _build_controls(self, parent):
        """Panneau de contrôles avec notebook."""
        # Zone de scroll pour les contrôles
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(controls_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._tab_geometry()
        self._tab_superres()
        self._tab_damage()
        self._tab_denoise()
        self._tab_contrast()
        self._tab_color()
        self._tab_sharpen()
        self._tab_spectral()
        self._tab_vintage()

        # Frame pour les actions rapides en bas
        actions_frame = ttk.LabelFrame(controls_frame, text="⚡ Actions rapides")
        actions_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(actions_frame, text="🎯 Mode Archival Complet",
                  command=self.apply_archival_mode, width=30).pack(pady=5, padx=5, fill=tk.X)
        ttk.Button(actions_frame, text="🔄 Appliquer tous les changements",
                  command=self.apply_all_changes, width=30).pack(pady=5, padx=5, fill=tk.X)

    def _add_control(self, parent, name, ctrl_type, **kwargs):
        """Ajoute un contrôle et le stocke."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        label = ttk.Label(frame, text=kwargs.get('label', name))
        label.pack(side=tk.LEFT)

        if ctrl_type == 'check':
            var = tk.BooleanVar(value=kwargs.get('default', False))
            ctrl = ttk.Checkbutton(frame, variable=var)
            ctrl.pack(side=tk.RIGHT)
            self.controls[name] = {'type': 'check', 'var': var, 'ctrl': ctrl}

        elif ctrl_type == 'slider':
            var = tk.DoubleVar(value=kwargs.get('default', 0.0))
            ctrl = ttk.Scale(frame, from_=kwargs.get('from', 0.0), to=kwargs.get('to', 1.0),
                           variable=var, orient=tk.HORIZONTAL)
            ctrl.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
            self.controls[name] = {'type': 'slider', 'var': var, 'ctrl': ctrl}
            var.trace_add('write', lambda *args: self.schedule_preview())

        elif ctrl_type == 'spinbox':
            var = tk.DoubleVar(value=kwargs.get('default', 0.0))
            ctrl = ttk.Spinbox(frame, from_=kwargs.get('from', 0), to=kwargs.get('to', 100),
                             increment=kwargs.get('step', 1), textvariable=var, width=8)
            ctrl.pack(side=tk.RIGHT)
            self.controls[name] = {'type': 'spinbox', 'var': var, 'ctrl': ctrl}
            var.trace_add('write', lambda *args: self.schedule_preview())

        elif ctrl_type == 'combobox':
            var = tk.StringVar(value=kwargs.get('default', ''))
            ctrl = ttk.Combobox(frame, textvariable=var, values=kwargs.get('values', []),
                              state='readonly', width=12)
            ctrl.pack(side=tk.RIGHT)
            self.controls[name] = {'type': 'combobox', 'var': var, 'ctrl': ctrl}
            var.trace_add('write', lambda *args: self.schedule_preview())

        return frame

    def _tab_geometry(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📐 Géométrie")

        self._add_control(tab, 'auto_straighten', 'check', label="Redressement auto", default=False)
        self._add_control(tab, 'lens_distortion_k1', 'slider', label="Distorsion lentille", from_=-1.0, to=1.0, default=0.0)
        self._add_control(tab, 'rotate', 'spinbox', label="Rotation (°)", from_=-180, to=180, step=1, default=0)
        self._add_control(tab, 'scale', 'spinbox', label="Échelle", from_=0.1, to=4.0, step=0.1, default=1.0)

    def _tab_superres(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔍 Super-résolution")

        self._add_control(tab, 'superres_scale', 'combobox', label="Facteur", values=['1', '2', '3', '4'], default='1')
        self._add_control(tab, 'superres_method', 'combobox', label="Méthode", values=['lanczos', 'ibp', 'edge'], default='lanczos')

    def _tab_damage(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🏛️ Dommages")

        self._add_control(tab, 'restore_foxing', 'check', label="Supprimer foxing", default=False)
        self._add_control(tab, 'foxing_aggressiveness', 'slider', label="Aggressivité foxing", from_=0.0, to=1.0, default=0.7)
        self._add_control(tab, 'restore_mold', 'check', label="Supprimer moisissure", default=False)
        self._add_control(tab, 'restore_cracks', 'check', label="Réparer craquelures", default=False)
        self._add_control(tab, 'restore_tears', 'check', label="Réparer déchirures", default=False)
        self._add_control(tab, 'restore_water', 'check', label="Corriger dégâts d'eau", default=False)
        self._add_control(tab, 'restore_silvering', 'check', label="Supprimer mirage argentique", default=False)
        self._add_control(tab, 'daguerreotype_restore', 'check', label="Restauration daguerréotype", default=False)

    def _tab_denoise(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔇 Débruitage")

        self._add_control(tab, 'denoise_enable', 'check', label="Activer débruitage", default=False)
        self._add_control(tab, 'denoise_method', 'combobox', label="Méthode", values=['bm3d', 'nlm', 'wavelet', 'bilateral'], default='bm3d')
        self._add_control(tab, 'denoise_sigma', 'spinbox', label="Sigma bruit", from_=1, to=100, step=1, default=25)

    def _tab_contrast(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🌓 Contraste")

        self._add_control(tab, 'contrast_method', 'combobox', label="Méthode", values=['clahe', 'retinex', 'reinhard', 'none'], default='none')
        self._add_control(tab, 'clahe_clip', 'slider', label="Limite CLAHE", from_=0.1, to=10.0, default=2.0)
        self._add_control(tab, 'clahe_grid', 'spinbox', label="Grille CLAHE", from_=2, to=16, step=1, default=8)
        self._add_control(tab, 'gamma', 'slider', label="Gamma", from_=0.1, to=5.0, default=1.0)
        self._add_control(tab, 'brightness', 'slider', label="Luminosité", from_=0.1, to=3.0, default=1.0)
        self._add_control(tab, 'shadows_lift', 'slider', label="Relever ombres", from_=0.0, to=1.0, default=0.0)
        self._add_control(tab, 'highlights_compress', 'slider', label="Comprimer HL", from_=0.0, to=1.0, default=0.0)

    def _tab_color(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎨 Couleur")

        self._add_control(tab, 'color_constancy', 'combobox', label="Constance couleur", values=['gray_world', 'white_patch', 'shades_gray', 'none'], default='none')
        self._add_control(tab, 'color_fade_recovery', 'check', label="Récupération couleurs", default=False)
        self._add_control(tab, 'remove_yellowing', 'check', label="Supprimer jaunissement", default=False)
        self._add_control(tab, 'yellowing_strength', 'slider', label="Force anti-jaunissement", from_=0.0, to=2.0, default=1.0)
        self._add_control(tab, 'saturation', 'slider', label="Saturation", from_=0.0, to=3.0, default=1.0)
        self._add_control(tab, 'warmth', 'slider', label="Chaleur", from_=-1.0, to=1.0, default=0.0)
        self._add_control(tab, 'sepia', 'slider', label="Sépia", from_=0.0, to=1.0, default=0.0)

    def _tab_sharpen(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔎 Netteté")

        self._add_control(tab, 'sharpen_enable', 'check', label="Activer netteté", default=False)
        self._add_control(tab, 'sharpen_method', 'combobox', label="Méthode", values=['unsharp', 'phase_congruency', 'deconvolution'], default='unsharp')
        self._add_control(tab, 'sharpen_amount', 'slider', label="Intensité", from_=0.0, to=2.0, default=0.6)
        self._add_control(tab, 'sharpen_radius', 'slider', label="Rayon", from_=0.1, to=5.0, default=1.2)
        self._add_control(tab, 'deconv_iterations', 'spinbox', label="Itérations R-L", from_=1, to=20, step=1, default=8)

    def _tab_spectral(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔬 Spectral")

        self._add_control(tab, 'homomorphic_filter', 'check', label="Filtre homomorphique", default=False)
        self._add_control(tab, 'homomorphic_cutoff', 'spinbox', label="Cutoff", from_=1, to=100, step=1, default=30)
        self._add_control(tab, 'anisotropic_diffusion', 'check', label="Diffusion anisotrope", default=False)
        self._add_control(tab, 'anisotropic_iterations', 'spinbox', label="Itérations", from_=1, to=50, step=1, default=15)
        self._add_control(tab, 'anisotropic_kappa', 'spinbox', label="Kappa", from_=1, to=100, step=1, default=30)

    def _tab_vintage(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎞️ Vintage")

        self._add_control(tab, 'vignette', 'slider', label="Vignettage", from_=0.0, to=1.0, default=0.0)
        self._add_control(tab, 'vignette_shape', 'slider', label="Forme vignette", from_=0.5, to=4.0, default=2.0)
        self._add_control(tab, 'film_grain', 'slider', label="Grain film", from_=0.0, to=0.1, default=0.0)
        self._add_control(tab, 'grain_color', 'check', label="Grain coloré", default=False)
        self._add_control(tab, 'halation', 'slider', label="Halation", from_=0.0, to=1.0, default=0.0)
        self._add_control(tab, 'fade', 'slider', label="Effet délavé", from_=0.0, to=0.5, default=0.0)

    # =============================================================================
    # GESTION DES IMAGES
    # =============================================================================
    def load_image(self):
        """Charge une image depuis un fichier."""
        filepath = filedialog.askopenfilename(
            title="Charger une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("Tous fichiers", "*.*")]
        )
        if not filepath:
            return

        try:
            self.status_label.config(text="Chargement en cours...")
            img_pil = Image.open(filepath).convert('RGB')
            img_np = np.array(img_pil)

            self.original_image = img_np.copy()
            self.current_image = img_np.copy()
            self.current_file_path = filepath
            self.undo_stack = []
            self.redo_stack = []

            self.update_preview()
            self.fit_to_window()

            h, w = img_np.shape[:2]
            self.info_label.config(text=f"{w}×{h}px | {os.path.basename(filepath)}")
            self.status_label.config(text=f"Image chargée : {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image :\n{str(e)}")
            self.status_label.config(text="Erreur de chargement")

    def save_image(self):
        """Sauvegarde l'image courante."""
        if self.current_image is None:
            messagebox.showwarning("Attention", "Aucune image à sauvegarder")
            return

        if self.current_file_path is None:
            self.save_image_as()
            return

        self._save_to_path(self.current_file_path)

    def save_image_as(self):
        """Sauvegarde l'image courante sous un nouveau nom."""
        if self.current_image is None:
            messagebox.showwarning("Attention", "Aucune image à sauvegarder")
            return

        filepath = filedialog.asksaveasfilename(
            title="Sauvegarder l'image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("TIFF", "*.tiff *.tif"), ("BMP", "*.bmp")]
        )
        if not filepath:
            return

        self._save_to_path(filepath)

    def _save_to_path(self, filepath):
        """Sauvegarde effective."""
        try:
            self.status_label.config(text="Sauvegarde en cours...")
            pil = Image.fromarray(self.current_image)

            if filepath.lower().endswith(('.jpg', '.jpeg')):
                pil.save(filepath, quality=95)
            else:
                pil.save(filepath)

            self.current_file_path = filepath
            self.status_label.config(text=f"Sauvegardé : {os.path.basename(filepath)}")
            messagebox.showinfo("Succès", f"Image sauvegardée :\n{filepath}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de sauvegarde :\n{str(e)}")
            self.status_label.config(text="Erreur de sauvegarde")

    def reset_image(self):
        """Réinitialise à l'image originale."""
        if self.original_image is None:
            return

        if messagebox.askyesno("Réinitialiser", "Réinitialiser tous les paramètres ?"):
            self.current_image = self.original_image.copy()
            self.undo_stack = []
            self.redo_stack = []
            self._reset_all_controls()
            self.update_preview()
            self.status_label.config(text="Image réinitialisée")

    def _reset_all_controls(self):
        """Réinitialise tous les contrôles à leurs valeurs par défaut."""
        defaults = {
            'auto_straighten': False, 'lens_distortion_k1': 0.0, 'rotate': 0, 'scale': 1.0,
            'superres_scale': '1', 'superres_method': 'lanczos',
            'restore_foxing': False, 'foxing_aggressiveness': 0.7, 'restore_mold': False,
            'restore_cracks': False, 'restore_tears': False, 'restore_water': False,
            'restore_silvering': False, 'daguerreotype_restore': False,
            'denoise_enable': False, 'denoise_method': 'bm3d', 'denoise_sigma': 25,
            'contrast_method': 'none', 'clahe_clip': 2.0, 'clahe_grid': 8,
            'gamma': 1.0, 'brightness': 1.0, 'shadows_lift': 0.0, 'highlights_compress': 0.0,
            'color_constancy': 'none', 'color_fade_recovery': False, 'remove_yellowing': False,
            'yellowing_strength': 1.0, 'saturation': 1.0, 'warmth': 0.0, 'sepia': 0.0,
            'sharpen_enable': False, 'sharpen_method': 'unsharp', 'sharpen_amount': 0.6,
            'sharpen_radius': 1.2, 'deconv_iterations': 8,
            'homomorphic_filter': False, 'homomorphic_cutoff': 30,
            'anisotropic_diffusion': False, 'anisotropic_iterations': 15, 'anisotropic_kappa': 30,
            'vignette': 0.0, 'vignette_shape': 2.0, 'film_grain': 0.0, 'grain_color': False,
            'halation': 0.0, 'fade': 0.0
        }
        for name, val in defaults.items():
            if name in self.controls:
                self.controls[name]['var'].set(val)

    # =============================================================================
    # PRÉVISUALISATION
    # =============================================================================
    def schedule_preview(self):
        """Planifie une prévisualisation après un délai (debounce)."""
        if self.preview_after_id:
            self.root.after_cancel(self.preview_after_id)
        self.preview_after_id = self.root.after(300, self.update_preview)

    def update_preview(self):
        """Met à jour la prévisualisation."""
        if self.original_image is None or self.is_processing:
            return

        self.preview_after_id = None

        # Utiliser un thread pour le traitement lourd
        self.is_processing = True
        self.status_label.config(text="Traitement en cours...")

        def process_thread():
            try:
                result = self._apply_pipeline(self.original_image)
                self.root.after(0, lambda: self._on_preview_ready(result))
            except Exception as e:
                self.root.after(0, lambda: self._on_preview_error(e))

        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()

    def _on_preview_ready(self, result):
        """Callback quand le traitement est terminé."""
        self.current_image = result
        self.is_processing = False

        self._display_image()

        h, w = result.shape[:2]
        self.info_label.config(text=f"{w}×{h}px")
        self.status_label.config(text="Prévisualisation mise à jour")

    def _on_preview_error(self, error):
        """Callback en cas d'erreur."""
        self.is_processing = False
        self.status_label.config(text=f"Erreur : {str(error)}")
        print(f"Erreur de traitement: {error}")
        traceback.print_exc()

    def _display_image(self):
        """Affiche l'image courante sur le canvas."""
        if self.current_image is None:
            return

        # Créer une version pour l'affichage (ajustée au zoom)
        h, w = self.current_image.shape[:2]
        display_w = int(w * self.zoom_factor)
        display_h = int(h * self.zoom_factor)

        # Limiter la taille d'affichage pour les performances
        max_display = 2000
        if display_w > max_display or display_h > max_display:
            scale = min(max_display / display_w, max_display / display_h)
            display_w = int(display_w * scale)
            display_h = int(display_h * scale)
            self.preview_image = cv2.resize(self.current_image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        else:
            self.preview_image = self.current_image.copy()

        pil_img = Image.fromarray(self.preview_image)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, display_w, display_h))

    def on_canvas_resize(self, event):
        """Gère le redimensionnement du canvas."""
        pass

    def zoom(self, delta):
        """Zoom avant/arrière."""
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor + delta))
        self.zoom_label.config(text=f"{int(self.zoom_factor*100)}%")
        self._display_image()

    def fit_to_window(self):
        """Ajuste l'image à la fenêtre."""
        if self.current_image is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 100 or canvas_h < 100:
            return

        h, w = self.current_image.shape[:2]
        self.zoom_factor = min(canvas_w / w, canvas_h / h, 1.0)
        self.zoom_label.config(text=f"{int(self.zoom_factor*100)}%")
        self._display_image()

    # =============================================================================
    # UNDO / REDO
    # =============================================================================
    def undo(self):
        """Annule la dernière modification."""
        if not self.undo_stack:
            return
        self.redo_stack.append(self.current_image.copy())
        self.current_image = self.undo_stack.pop()
        self.update_preview()
        self.status_label.config(text="Action annulée")

    def redo(self):
        """Refait la dernière modification annulée."""
        if not self.redo_stack:
            return
        self.undo_stack.append(self.current_image.copy())
        self.current_image = self.redo_stack.pop()
        self.update_preview()
        self.status_label.config(text="Action refaite")

    def push_undo(self):
        """Sauvegarde l'état actuel pour undo."""
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            if len(self.undo_stack) > 20:
                self.undo_stack.pop(0)
            self.redo_stack.clear()

    # =============================================================================
    # PIPELINE DE TRAITEMENT
    # =============================================================================
    def _get_param(self, name, default=None):
        """Récupère la valeur d'un paramètre."""
        if name not in self.controls:
            return default
        ctrl = self.controls[name]
        if ctrl['type'] == 'check':
            return ctrl['var'].get()
        elif ctrl['type'] == 'slider':
            return ctrl['var'].get()
        elif ctrl['type'] == 'spinbox':
            val = ctrl['var'].get()
            try:
                return float(val) if '.' in str(val) else int(val)
            except:
                return default
        elif ctrl['type'] == 'combobox':
            return ctrl['var'].get()
        return default

    def _apply_pipeline(self, img):
        """Applique le pipeline complet à partir des paramètres GUI."""
        result = img.copy()
        h, w = result.shape[:2]

        # 1. Géométrie
        if self._get_param('auto_straighten'):
            result = GeometryCorrector.auto_straighten(result)

        rotate = self._get_param('rotate', 0)
        if rotate != 0:
            h, w = result.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), rotate, 1.0)
            result = cv2.warpAffine(result, M, (w, h), flags=cv2.INTER_CUBIC)

        scale = self._get_param('scale', 1.0)
        if scale != 1.0:
            result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        k1 = self._get_param('lens_distortion_k1', 0.0)
        if k1 != 0.0:
            result = GeometryCorrector.lens_distortion(result, k1)

        # 2. Super-résolution
        sr_scale = int(self._get_param('superres_scale', '1'))
        if sr_scale > 1:
            method = self._get_param('superres_method', 'lanczos')
            if method == 'lanczos':
                result = SuperResolution.lanczos_upscale(result, sr_scale)
            elif method == 'ibp':
                result = SuperResolution.iterative_back_projection(result, sr_scale)
            elif method == 'edge':
                result = SuperResolution.edge_directed(result, sr_scale)

        # 3. Dommages
        if self._get_param('restore_foxing'):
            result = DamageRestorer.remove_foxing(result, self._get_param('foxing_aggressiveness', 0.7))
        if self._get_param('restore_mold'):
            result = DamageRestorer.remove_mold(result)
        if self._get_param('restore_cracks'):
            result = DamageRestorer.repair_cracks(result)
        if self._get_param('restore_tears'):
            result = DamageRestorer.repair_tears(result)
        if self._get_param('restore_water'):
            result = DamageRestorer.remove_water_damage(result)
        if self._get_param('restore_silvering'):
            result = DamageRestorer.remove_silvering(result)
        if self._get_param('daguerreotype_restore'):
            result = DamageRestorer.daguerreotype_restore(result)

        # 4. Débruitage
        if self._get_param('denoise_enable'):
            method = self._get_param('denoise_method', 'bm3d')
            sigma = self._get_param('denoise_sigma', 25)
            if method == 'bm3d':
                result = NoiseReduction.bm3d(result, sigma)
            elif method == 'nlm':
                result = NoiseReduction.nlm(result, sigma)
            elif method == 'wavelet':
                result = NoiseReduction.wavelet(result, sigma)
            elif method == 'bilateral':
                result = NoiseReduction.bilateral_stack(result)

        # 5. Contraste
        contrast_method = self._get_param('contrast_method', 'none')
        if contrast_method == 'clahe':
            result = ContrastEnhancer.clahe(result, self._get_param('clahe_clip', 2.0), int(self._get_param('clahe_grid', 8)))
        elif contrast_method == 'retinex':
            result = ContrastEnhancer.retinex_msrcr(result)
        elif contrast_method == 'reinhard':
            result = ContrastEnhancer.reinhard_tmo(result)

        # 6. Couleur
        color_constancy = self._get_param('color_constancy', 'none')
        if color_constancy == 'gray_world':
            result = ColorScience.gray_world(result)
        elif color_constancy == 'white_patch':
            result = ColorScience.white_patch(result)
        elif color_constancy == 'shades_gray':
            result = ColorScience.shades_of_gray(result)

        if self._get_param('color_fade_recovery'):
            result = ColorScience.recover_faded(result)
        if self._get_param('remove_yellowing'):
            result = ColorScience.remove_yellowing(result, self._get_param('yellowing_strength', 1.0))

        warmth = self._get_param('warmth', 0.0)
        if warmth != 0.0:
            img_f = to_float(result)
            factor = warmth
            if factor > 0:
                img_f[:,:,0] = np.clip(img_f[:,:,0] + factor * 0.1, 0, 1)
                img_f[:,:,2] = np.clip(img_f[:,:,2] - factor * 0.07, 0, 1)
            else:
                img_f[:,:,0] = np.clip(img_f[:,:,0] + factor * 0.07, 0, 1)
                img_f[:,:,2] = np.clip(img_f[:,:,2] - factor * 0.1, 0, 1)
            result = to_uint8(img_f)

        saturation = self._get_param('saturation', 1.0)
        if saturation != 1.0:
            pil = Image.fromarray(result)
            pil = ImageEnhance.Color(pil).enhance(saturation)
            result = np.array(pil)

        sepia = self._get_param('sepia', 0.0)
        if sepia > 0:
            img_f = to_float(result)
            sepia_matrix = np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]])
            sepia_img = np.dot(img_f, sepia_matrix.T)
            sepia_img = np.clip(sepia_img, 0, 1)
            result = to_uint8(img_f * (1 - sepia) + sepia_img * sepia)

        # 7. Netteté
        if self._get_param('sharpen_enable'):
            method = self._get_param('sharpen_method', 'unsharp')
            amount = self._get_param('sharpen_amount', 0.6)
            radius = self._get_param('sharpen_radius', 1.2)

            if method == 'unsharp':
                pil = Image.fromarray(result)
                blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
                arr = result.astype(np.float32)
                arr_b = np.array(blurred).astype(np.float32)
                diff = arr - arr_b
                result = np.clip(arr + amount * diff, 0, 255).astype(np.uint8)
            elif method == 'phase_congruency':
                result = SpectralRestorer.phase_congruency_sharpen(result)
            elif method == 'deconvolution':
                psf = DeconvolutionRestorer.estimate_psf_blind(result)
                iters = int(self._get_param('deconv_iterations', 8))
                result = DeconvolutionRestorer.richardson_lucy(result, psf, iters)

        # 8. Spectral
        if self._get_param('homomorphic_filter'):
            cutoff = self._get_param('homomorphic_cutoff', 30)
            result = SpectralRestorer.homomorphic_filter(result, cutoff=cutoff)
        if self._get_param('anisotropic_diffusion'):
            n_iter = int(self._get_param('anisotropic_iterations', 15))
            kappa = self._get_param('anisotropic_kappa', 30)
            result = SpectralRestorer.anisotropic_diffusion(result, n_iter=n_iter, kappa=kappa)

        # 9. Ajustements finaux
        gamma = self._get_param('gamma', 1.0)
        if gamma != 1.0:
            lut = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)], dtype=np.uint8)
            result = lut[result]

        brightness = self._get_param('brightness', 1.0)
        if brightness != 1.0:
            pil = Image.fromarray(result)
            pil = ImageEnhance.Brightness(pil).enhance(brightness)
            result = np.array(pil)

        shadows_lift = self._get_param('shadows_lift', 0.0)
        if shadows_lift > 0:
            f = to_float(result)
            f += shadows_lift * (1.0 - f)
            result = to_uint8(f)

        highlights_compress = self._get_param('highlights_compress', 0.0)
        if highlights_compress > 0:
            f = to_float(result)
            f -= highlights_compress * f
            result = to_uint8(f)

        # 10. Vintage
        vignette = self._get_param('vignette', 0.0)
        if vignette > 0:
            result = VintageEffects.vignette(result, vignette, self._get_param('vignette_shape', 2.0))

        film_grain = self._get_param('film_grain', 0.0)
        if film_grain > 0:
            result = VintageEffects.film_grain(result, film_grain, self._get_param('grain_color', False))

        halation = self._get_param('halation', 0.0)
        if halation > 0:
            result = VintageEffects.halation(result, halation)

        fade = self._get_param('fade', 0.0)
        if fade > 0:
            result = to_uint8(to_float(result) * (1 - fade) + 0.5 * fade)

        return result

    def apply_all_changes(self):
        """Applique tous les changements et sauvegarde dans l'historique."""
        if self.original_image is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image")
            return

        self.push_undo()
        self.update_preview()

    def apply_archival_mode(self):
        """Applique le mode archival complet."""
        if self.original_image is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image")
            return

        # Configurer les paramètres pour le mode archival
        defaults = {
            'auto_straighten': True, 'denoise_enable': True, 'denoise_method': 'bm3d',
            'contrast_method': 'clahe', 'color_constancy': 'shades_gray',
            'sharpen_enable': True, 'sharpen_method': 'unsharp', 'sharpen_amount': 0.6,
            'restore_foxing': True, 'restore_mold': True, 'restore_cracks': True,
            'remove_yellowing': True, 'color_fade_recovery': True
        }

        for name, val in defaults.items():
            if name in self.controls:
                self.controls[name]['var'].set(val)

        self.push_undo()
        self.update_preview()
        self.status_label.config(text="Mode archival appliqué")

    # =============================================================================
    # ANALYSE & MÉTRIQUES
    # =============================================================================
    def run_analysis(self):
        """Exécute l'analyse des dégradations."""
        if self.current_image is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image")
            return

        analyzer = DegradationAnalyzer(self.current_image)
        degradations = analyzer.analyze_all()

        deg_names = [d.name for d in degradations]
        result_text = "Dégradations détectées :\n\n"
        if deg_names:
            for name in deg_names:
                result_text += f"  • {name}\n"
        else:
            result_text += "  Aucune dégradation majeure détectée\n"

        messagebox.showinfo("Analyse des dégradations", result_text)

    def show_metrics(self):
        """Affiche les métriques de qualité."""
        if self.current_image is None:
            return

        metrics_before = QualityMetrics.evaluate(self.original_image)
        metrics_after = QualityMetrics.evaluate(self.current_image)

        result_text = "Métriques de qualité :\n\n"
        result_text += f"{'Métrique':<20} {'Avant':>10} {'Après':>10}\n"
        result_text += "-" * 42 + "\n"

        for key in ['sharpness', 'contrast', 'colorfulness', 'snr_db']:
            label = key.replace('_', ' ').title()
            result_text += f"{label:<20} {metrics_before[key]:>10.2f} {metrics_after[key]:>10.2f}\n"

        messagebox.showinfo("Métriques de qualité", result_text)


# =============================================================================
# MAIN
# =============================================================================
def main():
    root = tk.Tk()

    # Configuration du style sombre
    root.configure(background='#2d2d2d')

    # Centrer la fenêtre
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{min(1400, screen_w-50)}x{min(900, screen_h-50)}+{max(10, (screen_w-1400)//2)}+{max(10, (screen_h-900)//2)}")

    app = ArchivalRestorationGUI(root)

    # Bindings clavier (corrigés)
    root.bind('<Control-z>', lambda e: app.undo())
    root.bind('<Control-y>', lambda e: app.redo())
    root.bind('<Control-s>', lambda e: app.save_image())
    root.bind('<Control-o>', lambda e: app.load_image())
    root.bind('<Control-0>', lambda e: app.fit_to_window())

    # Zoom : on utilise les formes standards reconnues par Tcl/Tk
    root.bind('<Control-KeyPress-plus>', lambda e: app.zoom(0.1))
    root.bind('<Control-equal>', lambda e: app.zoom(0.1))       # Fallback Ctrl+=
    root.bind('<Control-KeyPress-minus>', lambda e: app.zoom(-0.1))
    root.bind('<Control-minus>', lambda e: app.zoom(-0.1))

    root.bind('<F5>', lambda e: app.apply_all_changes())

    # Message d'accueil si pas d'image
    root.update()

    root.mainloop()

if __name__ == "__main__":
    main()
