#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     ARCHIVAL RESTORATION SUITE — Version Ultra-Étendue v3.0                 ║
║     Laboratoire de Restauration Numérique — Niveau Smithsonian               ║
║                                                                              ║
║     Pipeline complet : Acquisition → Analyse → Restauration → Validation    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Exemples d'utilisation avancée :
  # Restauration archéologique complète avec rapport
  python archival_suite.py photo.png --mode archival --report-json

  # Restauration ciblée : moisissure + décoloration
  python archival_suite.py photo.png --restore-mold --color-fade-recovery

  # Super-résolution + inpainting avancé
  python archival_suite.py photo.png --superres 2 --inpaint-exemplar

  # Analyse spectroscopique simulée
  python archival_suite.py photo.png --spectral-analysis --spectral-bands 16

  # Restauration de daguerréotype oxydé
  python archival_suite.py photo.png --medium daguerreotype --oxidation-removal
"""

import argparse
import os
import sys
import time
import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage, signal, interpolate, optimize
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener, convolve2d
from collections import deque

# Imports optionnels avec fallback
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
    warnings.warn("scikit-image non disponible")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA, NMF
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# =============================================================================
# CONSTANTES
# =============================================================================

class DegradationType(Enum):
    FADING = auto(); YELLOWING = auto(); SILVERING = auto(); FOXING = auto()
    MOLD = auto(); WATER_DAMAGE = auto(); CRACKS = auto(); TEARS = auto()
    SCRATCHES = auto(); DUST = auto(); VIGNETTING = auto(); BLUR = auto()
    NOISE = auto(); COLOR_SHIFT = auto(); BANDING = auto(); HALATION = auto()
    OXIDATION = auto(); SILVER_MIGRATION = auto()


@dataclass
class RestorationReport:
    input_path: str; output_path: str
    original_dimensions: Tuple[int, int]; final_dimensions: Tuple[int, int]
    detected_degradations: List[str] = field(default_factory=list)
    applied_steps: List[Dict] = field(default_factory=list)
    spectral_analysis: Dict = field(default_factory=dict)
    quality_metrics: Dict = field(default_factory=dict)
    processing_time: float = 0.0

    def to_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False, default=str)

    def print_summary(self):
        print("\n" + "═"*70)
        print("  RAPPORT DE RESTAURATION ARCHIVALE")
        print("═"*70)
        print(f"  Source    : {self.input_path}")
        print(f"  Sortie    : {self.output_path}")
        print(f"  Dimensions: {self.original_dimensions} → {self.final_dimensions}")
        print(f"  Temps     : {self.processing_time:.2f}s")
        print(f"\n  Dégradations détectées:")
        for d in self.detected_degradations: print(f"    • {d}")
        print(f"\n  Étapes ({len(self.applied_steps)}):")
        for s in self.applied_steps: print(f"    ✓ {s['name']} ({s['duration']:.2f}s)")
        print("═"*70)


# =============================================================================
# UTILITAIRES
# =============================================================================

def to_float(img): return img.astype(np.float32) / 255.0
def to_uint8(img_f): return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
def pil_to_np(pil): return np.array(pil)
def np_to_pil(arr): return Image.fromarray(arr.astype(np.uint8))


# =============================================================================
# ANALYSE DES DÉGRADATIONS
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

    def get_spectral_signature(self, n_bands: int = 8) -> Dict:
        bands = {}
        for i in range(n_bands):
            low, high = i / n_bands, (i + 1) / n_bands
            f = np.fft.fft2(self.gray)
            fshift = np.fft.fftshift(f)
            rows, cols = self.gray.shape
            crow, ccol = rows//2, cols//2
            mask = np.zeros((rows, cols))
            rl, rh = int(min(rows, cols) * low / 2), int(min(rows, cols) * high / 2)
            Y, X = np.ogrid[:rows, :cols]
            dist = np.sqrt((X-ccol)**2 + (Y-crow)**2)
            mask[(dist >= rl) & (dist < rh)] = 1
            fshift *= mask
            band_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
            bands[f"band_{i}"] = {"mean": float(band_img.mean()), "std": float(band_img.std())}
        return bands


# =============================================================================
# RESTAURATION PAR DÉCOMPOSITION SPECTRALE
# =============================================================================

class SpectralRestorer:
    @staticmethod
    def homomorphic_filter(img: np.ndarray, cutoff: float = 30, order: int = 2,
                          low_gain: float = 0.5, high_gain: float = 2.0) -> np.ndarray:
        """Filtre homomorphique : sépare illumination et réflectance."""
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
        """Netteté basée sur la congruence de phase."""
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
        """Diffusion anisotrope de Perona-Malik."""
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


# =============================================================================
# INPAINTING AVANCÉ
# =============================================================================

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
        """Inpainting par patchs exemplaires."""
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


# =============================================================================
# SCIENCE DES COULEURS
# =============================================================================

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


# =============================================================================
# DÉCONVOLUTION ET NETTETÉ
# =============================================================================

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

    @staticmethod
    def wiener_deconvolution(img: np.ndarray, psf: np.ndarray, k: float = 0.01) -> np.ndarray:
        img_f = to_float(img)
        pad_h = (img.shape[0] - psf.shape[0]) // 2
        pad_w = (img.shape[1] - psf.shape[1]) // 2
        psf_p = np.pad(psf, ((pad_h, pad_h), (pad_w, pad_w)))
        psf_fft = np.fft.fft2(psf_p)
        psf_conj = np.conj(psf_fft)
        denom = np.abs(psf_fft)**2 + k
        result = np.zeros_like(img_f)
        for c in range(3):
            result[:,:,c] = np.real(np.fft.ifft2(np.fft.fft2(img_f[:,:,c]) * psf_conj / denom))
        return to_uint8(np.clip(result, 0, 1))


# =============================================================================
# SUPER-RÉSOLUTION
# =============================================================================

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


# =============================================================================
# RÉDUCTION DE BRUIT
# =============================================================================

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


# =============================================================================
# CONTRASTE AVANCÉ
# =============================================================================

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


# =============================================================================
# CORRECTION GÉOMÉTRIQUE
# =============================================================================

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


# =============================================================================
# RESTAURATION DE DOMMAGES SPÉCIFIQUES
# =============================================================================

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


# =============================================================================
# EFFETS VINTAGE ET FINALISATION
# =============================================================================

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


# =============================================================================
# MÉTRIQUES DE QUALITÉ
# =============================================================================

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
        return {
            "sharpness": QualityMetrics.sharpness(img),
            "contrast": QualityMetrics.contrast(img),
            "colorfulness": QualityMetrics.colorfulness(img),
            "snr_db": QualityMetrics.snr(img)
        }


# =============================================================================
# COMPARAISON AVANT/APRÈS
# =============================================================================

def make_compare(original: np.ndarray, restored: np.ndarray, output_path: str):
    h = max(original.shape[0], restored.shape[0])
    def pad(im):
        if im.shape[0] < h:
            ph = h - im.shape[0]
            im = np.pad(im, ((0, ph), (0, 0), (0, 0)), constant_values=128)
        return im
    sep = np.full((h, 4, 3), 200, dtype=np.uint8)
    canvas = np.concatenate([pad(original), sep, pad(restored)], axis=1)
    pil = Image.fromarray(canvas)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil)
        draw.text((20, 20), "AVANT", fill=(255, 255, 255))
        draw.text((pad(original).shape[1] + 24, 20), "APRES", fill=(255, 255, 255))
    except: pass
    pil.save(output_path)
    print(f"  Comparaison sauvegardée : {output_path}")


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Archival Restoration Suite v3.0", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    # I/O
    io = p.add_argument_group("📁 Entrée / Sortie")
    io.add_argument("input", help="Image source")
    io.add_argument("-o", "--output", default=None, help="Chemin de sortie")
    io.add_argument("--compare", action="store_true", help="Image comparatif avant/après")
    io.add_argument("--compare-output", default=None, help="Chemin comparatif")
    io.add_argument("--quality", type=int, default=95, help="Qualité JPEG (1-100)")
    io.add_argument("--report-json", default=None, help="Sauvegarder le rapport JSON")
    io.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux")

    # Mode
    mode = p.add_argument_group("🎯 Mode de restauration")
    mode.add_argument("--mode", choices=["archival", "quick", "custom"], default="archival",
                     help="archival=maximum qualité, quick=rapide, custom=paramètres manuels")
    mode.add_argument("--medium", choices=["paper_bw", "paper_color", "daguerreotype", "tintype",
                                            "ambrotype", "glass_plate", "nitrate_film", "acetate_film",
                                            "polaroid", "chromogenic"], default="paper_color",
                     help="Support photographique original")

    # Géométrie
    geo = p.add_argument_group("📐 Géométrie")
    geo.add_argument("--correct-geometry", action="store_true", help="Correction géométrique auto")
    geo.add_argument("--auto-straighten", action="store_true", help="Redressement auto")
    geo.add_argument("--lens-distortion", type=float, default=None, help="Correction distorsion k1")
    geo.add_argument("--rotate", type=float, default=0.0, help="Rotation manuelle (degrés)")
    geo.add_argument("--scale", type=float, default=1.0, help="Redimensionnement")
    geo.add_argument("--crop", nargs=4, type=int, metavar=("X","Y","W","H"), help="Recadrage")

    # Super-résolution
    sr = p.add_argument_group("🔍 Super-résolution")
    sr.add_argument("--superres", type=int, choices=[1,2,3,4], default=1, help="Facteur de mise à l'échelle")
    sr.add_argument("--superres-method", choices=["lanczos", "ibp", "edge"], default="ibp",
                   help="lanczos=Lanczos, ibp=rétroprojection, edge=dirigée par contours")

    # Dommages spécifiques
    dmg = p.add_argument_group("🏛️ Restauration de dommages")
    dmg.add_argument("--restore-foxing", action="store_true", help="Supprimer foxing")
    dmg.add_argument("--foxing-aggressiveness", type=float, default=0.7, help="Aggressivité foxing")
    dmg.add_argument("--restore-mold", action="store_true", help="Supprimer moisissure")
    dmg.add_argument("--restore-cracks", action="store_true", help="Réparer craquelures")
    dmg.add_argument("--restore-tears", action="store_true", help="Réparer déchirures")
    dmg.add_argument("--restore-water", action="store_true", help="Corriger dégâts d'eau")
    dmg.add_argument("--restore-silvering", action="store_true", help="Supprimer mirage argentique")
    dmg.add_argument("--oxidation-removal", action="store_true", help="Supprimer oxydation")
    dmg.add_argument("--restore-all-damage", action="store_true", help="Tous les dommages")

    # Débruitage
    noise = p.add_argument_group("🔇 Débruitage")
    noise.add_argument("--denoise-method", choices=["bm3d", "nlm", "wavelet", "bilateral", "none"], default="bm3d")
    noise.add_argument("--denoise-sigma", type=float, default=None, help="Sigma bruit (auto si absent)")
    noise.add_argument("--no-denoise", action="store_true", help="Désactiver débruitage")

    # Contraste
    ct = p.add_argument_group("🌓 Contraste & Luminosité")
    ct.add_argument("--contrast-method", choices=["clahe", "retinex", "reinhard", "none"], default="clahe")
    ct.add_argument("--clahe-clip", type=float, default=2.0, help="Limite CLAHE")
    ct.add_argument("--clahe-grid", type=int, default=8, help="Grille CLAHE")
    ct.add_argument("--retinex-sigmas", nargs="+", type=float, default=[15,80,250], help="Sigmas Retinex")
    ct.add_argument("--gamma", type=float, default=None, help="Correction gamma")
    ct.add_argument("--brightness", type=float, default=1.0, help="Luminosité")
    ct.add_argument("--shadows-lift", type=float, default=0.0, help="Relever ombres")
    ct.add_argument("--highlights-compress", type=float, default=0.0, help="Comprimer hautes lumières")

    # Couleur
    col = p.add_argument_group("🎨 Couleur")
    col.add_argument("--color-constancy", choices=["gray_world", "white_patch", "shades_gray", "none"], default="shades_gray")
    col.add_argument("--color-fade-recovery", action="store_true", help="Récupération couleurs décolorées")
    col.add_argument("--remove-yellowing", action="store_true", help="Supprimer jaunissement")
    col.add_argument("--yellowing-strength", type=float, default=1.0, help="Force anti-jaunissement")
    col.add_argument("--saturation", type=float, default=1.0, help="Saturation")
    col.add_argument("--warmth", type=float, default=0.0, help="Chaleur (-100 froid à +100 chaud)")
    col.add_argument("--sepia", type=float, default=0.0, help="Effet sépia (0-1)")

    # Netteté
    sharp = p.add_argument_group("🔎 Netteté")
    sharp.add_argument("--sharpen-method", choices=["unsharp", "phase_congruency", "deconvolution", "none"], default="unsharp")
    sharp.add_argument("--sharpen-amount", type=float, default=0.6, help="Intensité netteté")
    sharp.add_argument("--sharpen-radius", type=float, default=1.2, help="Rayon unsharp")
    sharp.add_argument("--deconv-iterations", type=int, default=8, help="Itérations Richardson-Lucy")
    sharp.add_argument("--no-sharpen", action="store_true", help="Désactiver netteté")

    # Spectral
    spec = p.add_argument_group("🔬 Analyse spectrale")
    spec.add_argument("--spectral-analysis", action="store_true", help="Activer analyse spectrale")
    spec.add_argument("--spectral-bands", type=int, default=8, help="Nombre de bandes spectrales")
    spec.add_argument("--homomorphic-filter", action="store_true", help="Filtre homomorphique")
    spec.add_argument("--anisotropic-diffusion", action="store_true", help="Diffusion anisotrope")

    # Vintage
    vin = p.add_argument_group("🎞️ Effets vintage")
    vin.add_argument("--vignette", type=float, default=0.0, help="Vignettage (0-1)")
    vin.add_argument("--vignette-shape", type=float, default=2.0, help="Forme vignette")
    vin.add_argument("--film-grain", type=float, default=0.0, help="Grain film (0-0.1)")
    vin.add_argument("--grain-color", action="store_true", help="Grain coloré")
    vin.add_argument("--halation", type=float, default=0.0, help="Halation (0-1)")
    vin.add_argument("--fade", type=float, default=0.0, help="Effet délavé (0-0.5)")

    # Inpainting
    inp = p.add_argument_group("🖌️ Inpainting")
    inp.add_argument("--inpaint-mask", default=None, help="Masque pour inpainting")
    inp.add_argument("--inpaint-method", choices=["telea", "ns", "exemplar"], default="telea")
    inp.add_argument("--inpaint-radius", type=int, default=5, help="Rayon inpainting")

    return p


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

class ArchivalRestorationPipeline:
    def __init__(self, args):
        self.args = args
        self.steps_log = []

    def _timed_step(self, name: str, func, *args, **kwargs):
        t0 = time.time()
        print(f"\n  {name}…")
        result = func(*args, **kwargs)
        dt = time.time() - t0
        print(f"    ✓ ({dt:.2f}s)" if dt > 0.1 else "    ✓")
        self.steps_log.append({"name": name, "duration": dt})
        return result, dt

    def process(self, input_path: str, output_path: str) -> RestorationReport:
        t_start = time.time()

        # Chargement
        print(f"\n{'═'*70}")
        print(f"  ARCHIVAL RESTORATION SUITE v3.0")
        print(f"{'═'*70}")
        print(f"\n📷 Chargement : {input_path}")
        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            raise ValueError(f"Impossible de lire {input_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original = img.copy()
        h, w = img.shape[:2]
        print(f"    Taille : {w}×{h} px")

        # Analyse
        print("\n🔬 Analyse des dégradations…")
        analyzer = DegradationAnalyzer(img)
        degradations = analyzer.analyze_all()
        deg_names = [d.name for d in degradations]
        print(f"    Détecté : {', '.join(deg_names) if deg_names else 'Aucune dégradation majeure'}")

        spectral_sig = {}
        if self.args.spectral_analysis:
            spectral_sig = analyzer.get_spectral_signature(self.args.spectral_bands)

        current = img.copy()

        # === GÉOMÉTRIE ===
        if self.args.correct_geometry or self.args.mode == "archival":
            current, _ = self._timed_step("Correction géométrique", self._step_geometry, current)

        # === SUPER-RÉSOLUTION ===
        if self.args.superres > 1:
            method_name = {"lanczos": "Lanczos", "ibp": "Rétroprojection", "edge": "Dirigée contours"}[self.args.superres_method]
            current, _ = self._timed_step(f"Super-résolution {self.args.superres}x ({method_name})",
                                          self._step_superres, current)

        # === DOMMAGES SPÉCIFIQUES ===
        if self.args.restore_all_damage or self.args.mode == "archival":
            self.args.restore_foxing = self.args.restore_mold = self.args.restore_cracks = True
            self.args.restore_tears = self.args.restore_water = self.args.restore_silvering = True

        if self.args.restore_foxing:
            current, _ = self._timed_step("Suppression foxing", DamageRestorer.remove_foxing, current, self.args.foxing_aggressiveness)
        if self.args.restore_mold:
            current, _ = self._timed_step("Suppression moisissure", DamageRestorer.remove_mold, current)
        if self.args.restore_cracks:
            current, _ = self._timed_step("Réparation craquelures", DamageRestorer.repair_cracks, current)
        if self.args.restore_tears:
            current, _ = self._timed_step("Réparation déchirures", DamageRestorer.repair_tears, current)
        if self.args.restore_water:
            current, _ = self._timed_step("Correction dégâts d'eau", DamageRestorer.remove_water_damage, current)
        if self.args.restore_silvering:
            current, _ = self._timed_step("Suppression mirage argentique", DamageRestorer.remove_silvering, current)
        if self.args.medium == "daguerreotype" or self.args.oxidation_removal:
            current, _ = self._timed_step("Restauration daguerréotype", DamageRestorer.daguerreotype_restore, current)

        # === INPAINTING MANUEL ===
        if self.args.inpaint_mask and os.path.exists(self.args.inpaint_mask):
            mask = cv2.imread(self.args.inpaint_mask, cv2.IMREAD_GRAYSCALE)
            method = {"telea": InpaintingEngine.telea_inpaint,
                     "ns": InpaintingEngine.ns_inpaint,
                     "exemplar": InpaintingEngine.exemplar_based_inpaint}[self.args.inpaint_method]
            current, _ = self._timed_step(f"Inpainting ({self.args.inpaint_method})", method, current, mask, self.args.inpaint_radius)

        # === DÉBRUITAGE ===
        if not self.args.no_denoise and self.args.mode != "quick":
            current, _ = self._timed_step(f"Débruitage ({self.args.denoise_method.upper()})", self._step_denoise, current)

        # === CONTRASTE ===
        if self.args.contrast_method != "none":
            current, _ = self._timed_step(f"Contraste ({self.args.contrast_method})", self._step_contrast, current)

        # === COULEUR ===
        if self.args.color_constancy != "none" or self.args.mode == "archival":
            current, _ = self._timed_step(f"Constance couleur ({self.args.color_constancy})", self._step_color, current)

        # === NETTETÉ ===
        if not self.args.no_sharpen and self.args.sharpen_method != "none":
            current, _ = self._timed_step(f"Netteté ({self.args.sharpen_method})", self._step_sharpen, current)

        # === TRAITEMENTS SPECTRAUX ===
        if self.args.homomorphic_filter:
            current, _ = self._timed_step("Filtre homomorphique", SpectralRestorer.homomorphic_filter, current)
        if self.args.anisotropic_diffusion:
            current, _ = self._timed_step("Diffusion anisotrope", SpectralRestorer.anisotropic_diffusion, current)

        # === EFFETS VINTAGE ===
        if self.args.vignette > 0:
            current = VintageEffects.vignette(current, self.args.vignette, self.args.vignette_shape)
        if self.args.film_grain > 0:
            current = VintageEffects.film_grain(current, self.args.film_grain, self.args.grain_color)
        if self.args.halation > 0:
            current = VintageEffects.halation(current, self.args.halation)
        if self.args.fade > 0:
            current = to_uint8(to_float(current) * (1 - self.args.fade) + 0.5 * self.args.fade)

        # === FINALISATION ===
        if self.args.brightness != 1.0:
            current = pil_to_np(ImageEnhance.Brightness(np_to_pil(current)).enhance(self.args.brightness))
        if self.args.gamma is not None:
            current = np.array([((i/255.0)**(1.0/self.args.gamma))*255 for i in range(256)], dtype=np.uint8)[current]
        if self.args.shadows_lift > 0:
            f = to_float(current)
            f += (self.args.shadows_lift/255.0) * (1.0 - f)
            current = to_uint8(f)
        if self.args.highlights_compress > 0:
            f = to_float(current)
            f -= (self.args.highlights_compress/255.0) * f
            current = to_uint8(f)
        if self.args.saturation != 1.0:
            current = pil_to_np(ImageEnhance.Color(np_to_pil(current)).enhance(self.args.saturation))

        # Métriques finales
        print("\n📊 Évaluation de la qualité…")
        metrics_before = QualityMetrics.evaluate(original)
        metrics_after = QualityMetrics.evaluate(current)
        print(f"    Netteté : {metrics_before['sharpness']:.1f} → {metrics_after['sharpness']:.1f}")
        print(f"    Contraste : {metrics_before['contrast']:.1f} → {metrics_after['contrast']:.1f}")
        print(f"    Colorimétrie : {metrics_before['colorfulness']:.1f} → {metrics_after['colorfulness']:.1f}")
        print(f"    SNR : {metrics_before['snr_db']:.1f}dB → {metrics_after['snr_db']:.1f}dB")

        # Sauvegarde
        print(f"\n💾 Sauvegarde → {output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        out_pil = np_to_pil(current)
        kwargs = {}
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            kwargs['quality'] = self.args.quality
        out_pil.save(output_path, **kwargs)

        # Comparaison
        if self.args.compare:
            comp_path = self.args.compare_output or os.path.splitext(output_path)[0] + "_comparaison.png"
            make_compare(original, current, comp_path)

        # Rapport
        total_time = time.time() - t_start
        report = RestorationReport(
            input_path=input_path,
            output_path=output_path,
            original_dimensions=(w, h),
            final_dimensions=(current.shape[1], current.shape[0]),
            detected_degradations=deg_names,
            applied_steps=self.steps_log,
            spectral_analysis=spectral_sig,
            quality_metrics={"before": metrics_before, "after": metrics_after},
            processing_time=total_time
        )

        if self.args.report_json:
            report.to_json(self.args.report_json)

        print(f"\n{'═'*70}")
        print(f"  ✅ Terminé en {total_time:.2f}s")
        print(f"{'═'*70}")

        return report

    def _step_geometry(self, img: np.ndarray) -> np.ndarray:
        if self.args.rotate != 0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), self.args.rotate, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        if self.args.crop:
            x, y, w, h = self.args.crop
            img = img[y:y+h, x:x+w]
        if self.args.scale != 1.0:
            img = cv2.resize(img, None, fx=self.args.scale, fy=self.args.scale, interpolation=cv2.INTER_LANCZOS4)
        if self.args.auto_straighten:
            img = GeometryCorrector.auto_straighten(img)
        if self.args.lens_distortion is not None:
            img = GeometryCorrector.lens_distortion(img, self.args.lens_distortion)
        return img

    def _step_superres(self, img: np.ndarray) -> np.ndarray:
        method_map = {
            "lanczos": SuperResolution.lanczos_upscale,
            "ibp": SuperResolution.iterative_back_projection,
            "edge": SuperResolution.edge_directed
        }
        return method_map[self.args.superres_method](img, self.args.superres)

    def _step_denoise(self, img: np.ndarray) -> np.ndarray:
        method_map = {
            "bm3d": lambda i: NoiseReduction.bm3d(i, self.args.denoise_sigma),
            "nlm": lambda i: NoiseReduction.nlm(i, self.args.denoise_sigma),
            "wavelet": lambda i: NoiseReduction.wavelet(i, self.args.denoise_sigma),
            "bilateral": lambda i: NoiseReduction.bilateral_stack(i),
            "none": lambda i: i
        }
        return method_map[self.args.denoise_method](img)

    def _step_contrast(self, img: np.ndarray) -> np.ndarray:
        method_map = {
            "clahe": lambda i: ContrastEnhancer.clahe(i, self.args.clahe_clip, self.args.clahe_grid),
            "retinex": lambda i: ContrastEnhancer.retinex_msrcr(i, self.args.retinex_sigmas),
            "reinhard": lambda i: ContrastEnhancer.reinhard_tmo(i),
            "none": lambda i: i
        }
        return method_map[self.args.contrast_method](img)

    def _step_color(self, img: np.ndarray) -> np.ndarray:
        method_map = {
            "gray_world": ColorScience.gray_world,
            "white_patch": ColorScience.white_patch,
            "shades_gray": lambda i: ColorScience.shades_of_gray(i, 6.0),
            "none": lambda i: i
        }
        img = method_map[self.args.color_constancy](img)

        if self.args.color_fade_recovery:
            img = ColorScience.recover_faded(img)
        if self.args.remove_yellowing:
            img = ColorScience.remove_yellowing(img, self.args.yellowing_strength)
        if self.args.warmth != 0:
            img_f = to_float(img)
            factor = self.args.warmth / 100.0
            if factor > 0:
                img_f[:,:,0] = np.clip(img_f[:,:,0] + factor * 0.1, 0, 1)
                img_f[:,:,2] = np.clip(img_f[:,:,2] - factor * 0.07, 0, 1)
            else:
                img_f[:,:,0] = np.clip(img_f[:,:,0] + factor * 0.07, 0, 1)
                img_f[:,:,2] = np.clip(img_f[:,:,2] - factor * 0.1, 0, 1)
            img = to_uint8(img_f)
        if self.args.sepia > 0:
            img_f = to_float(img)
            sepia = np.dot(img_f, np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]]).T)
            sepia = np.clip(sepia, 0, 1)
            img = to_uint8(img_f * (1 - self.args.sepia) + sepia * self.args.sepia)

        return img

    def _step_sharpen(self, img: np.ndarray) -> np.ndarray:
        if self.args.sharpen_method == "unsharp":
            pil = np_to_pil(img)
            blurred = pil.filter(ImageFilter.GaussianBlur(radius=self.args.sharpen_radius))
            arr = img.astype(np.float32)
            arr_b = pil_to_np(blurred).astype(np.float32)
            diff = arr - arr_b
            return np.clip(arr + self.args.sharpen_amount * diff, 0, 255).astype(np.uint8)
        elif self.args.sharpen_method == "phase_congruency":
            return SpectralRestorer.phase_congruency_sharpen(img)
        elif self.args.sharpen_method == "deconvolution":
            psf = DeconvolutionRestorer.estimate_psf_blind(img)
            return DeconvolutionRestorer.richardson_lucy(img, psf, self.args.deconv_iterations)
        return img


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ Fichier introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)

    base, ext = os.path.splitext(args.input)
    output = args.output or base + "_restauree.png"

    pipeline = ArchivalRestorationPipeline(args)

    try:
        report = pipeline.process(args.input, output)
        if args.report_json is None and args.mode == "archival":
            report.to_json(base + "_rapport.json")
            print(f"\n📋 Rapport sauvegardé : {base}_rapport.json")
    except Exception as e:
        print(f"\n❌ Erreur : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
