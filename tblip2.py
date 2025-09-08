"""
Enhanced Difference T-BLIP2 Model
Consistent training/inference pipeline with clustering-based caption generation
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2ForConditionalGeneration, Blip2Processor, AutoTokenizer
from sklearn.cluster import DBSCAN
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from collections import Counter
import math
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import contextlib
warnings.filterwarnings('ignore')

# Robust CUDA availability check
use_cuda = torch.cuda.is_available()

# Global tokenizer instance
GLOBAL_TOKENIZER = None

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_global_tokenizer():
    """Get global tokenizer instance"""
    global GLOBAL_TOKENIZER
    if GLOBAL_TOKENIZER is None:
        GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained('Salesforce/blip2-opt-2.7b')
        if GLOBAL_TOKENIZER.pad_token is None:
            GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
    return GLOBAL_TOKENIZER

# =========================
# Metrics & Visualization
# =========================
def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1)))

def _modified_precision_corpus(ref_lists, hyp_lists, n, add_one=True):
    clipped, total = 0, 0
    for refs, hyp in zip(ref_lists, hyp_lists):
        hyp_ngrams = _ngram_counts(hyp, n)
        max_ref = Counter()
        for r in refs:
            max_ref |= _ngram_counts(r, n)
        for ng, c in hyp_ngrams.items():
            clipped += min(c, max_ref.get(ng, 0))
        total += sum(hyp_ngrams.values())
    if add_one:
        clipped += 1
        total += 1
    return (clipped / total) if total > 0 else 0.0

def _brevity_penalty(ref_lists, hyp_lists):
    ref_len, hyp_len = 0, 0
    for refs, hyp in zip(ref_lists, hyp_lists):
        hyp_len += len(hyp)
        # choose ref length closest to hyp
        rl = min((abs(len(r)-len(hyp)), len(r)) for r in refs)[1] if refs else 0
        ref_len += rl
    if hyp_len == 0:
        return 0.0
    if hyp_len > ref_len:
        return 1.0
    return math.exp(1 - ref_len / max(1, hyp_len))

def compute_bleu_scores(ref_texts, hyp_texts):
    """Corpus BLEU-1 / BLEU-4 with simple smoothing."""
    # to lower + simple whitespace tokenization
    refs_tok = [[r.lower().split()] for r in ref_texts]
    hyps_tok = [h.lower().split() for h in hyp_texts]
    p1 = _modified_precision_corpus(refs_tok, hyps_tok, 1, add_one=True)
    p2 = _modified_precision_corpus(refs_tok, hyps_tok, 2, add_one=True)
    p3 = _modified_precision_corpus(refs_tok, hyps_tok, 3, add_one=True)
    p4 = _modified_precision_corpus(refs_tok, hyps_tok, 4, add_one=True)
    bp = _brevity_penalty(refs_tok, hyps_tok)
    bleu1 = bp * p1
    # geometric mean for BLEU-4
    gm = (p1 * p2 * p3 * p4) ** 0.25
    bleu4 = bp * gm
    return float(bleu1), float(bleu4)

def compute_meteor(ref_texts, hyp_texts):
    """Average METEOR if nltk (+ wordnet/omw-1.4/punkt) is available; else return None."""
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        from nltk.tokenize import word_tokenize
    except Exception:
        print("Note: METEOR requires nltk. Install with: pip install nltk")
        return None
    scores = []
    for ref, hyp in zip(ref_texts, hyp_texts):
        try:
            r_tok = word_tokenize(ref.lower())
            h_tok = word_tokenize(hyp.lower())
            scores.append(meteor_score([r_tok], h_tok))
        except LookupError:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                nltk.download('punkt', quiet=True)
                r_tok = word_tokenize(ref.lower())
                h_tok = word_tokenize(hyp.lower())
                scores.append(meteor_score([r_tok], h_tok))
            except Exception:
                # 仍然失败就退回空格分词，保证不再报错
                r_tok = ref.lower().split()
                h_tok = hyp.lower().split()
                scores.append(meteor_score([r_tok], h_tok))
    return float(sum(scores) / max(1, len(scores)))

def compute_simple_cider(ref_texts, hyp_texts):
    """Simple CIDEr approximation when pycocoevalcap is not available"""
    if not ref_texts or not hyp_texts:
        return 0.0
    
    # Simple n-gram overlap based scoring
    def get_ngrams(text, n):
        words = text.lower().split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)] if len(words) >= n else []
    
    # Compute unigram and bigram overlap
    total_score = 0.0
    for ref, hyp in zip(ref_texts, hyp_texts):
        if not ref.strip() or not hyp.strip():
            continue
        
        ref_1grams = set(get_ngrams(ref, 1))
        ref_2grams = set(get_ngrams(ref, 2))
        hyp_1grams = set(get_ngrams(hyp, 1))
        hyp_2grams = set(get_ngrams(hyp, 2))
        
        # Simple precision and recall
        if ref_1grams:
            p1 = len(hyp_1grams & ref_1grams) / len(hyp_1grams) if hyp_1grams else 0
            r1 = len(hyp_1grams & ref_1grams) / len(ref_1grams)
            f1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0
        else:
            f1 = 0
            
        if ref_2grams:
            p2 = len(hyp_2grams & ref_2grams) / len(hyp_2grams) if hyp_2grams else 0
            r2 = len(hyp_2grams & ref_2grams) / len(ref_2grams)
            f2 = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else 0
        else:
            f2 = 0
        
        total_score += (f1 + f2) / 2
    
    return total_score / len(ref_texts) if ref_texts else 0.0

def compute_cider_spice(ref_texts, hyp_texts, use_cider=True, use_spice=False):
    """
    计算 CIDEr-D / SPICE。依赖 pycocoevalcap；若不可用或报错则返回 None。
    返回: {'cider': float|None, 'spice': float|None}
    """
    results = {'cider': None, 'spice': None}
    # 组装 coco-caption 需要的 dict：{imgId: [refs]} 和 {imgId: [hyp]}
    gts = {}
    res = {}
    for i, (r, h) in enumerate(zip(ref_texts, hyp_texts)):
        img_id = str(i)
        gts[img_id] = [r]
        res[img_id] = [h]
    # CIDEr-D
    if use_cider:
        try:
            # ★ 过滤空字符串，避免 CIDEr 计算异常
            valid_gts = {}
            valid_res = {}
            for img_id, (refs, hyps) in zip(gts.keys(), zip(gts.values(), res.values())):
                ref = refs[0] if refs else ""
                hyp = hyps[0] if hyps else ""
                if ref.strip() and hyp.strip():  # 跳过空字符串
                    valid_gts[img_id] = refs
                    valid_res[img_id] = hyps
            
            if not valid_gts:
                print("Warning: All reference/hypothesis pairs are empty. Setting CIDEr-D to 0.0.")
                results['cider'] = 0.0
            else:
                try:
                    from pycocoevalcap.cider.cider import Cider
                    scorer = Cider()  # 大多数安装里 Cider 实现的是 CIDEr-D 版本
                except Exception:
                    from pycocoevalcap.cider.cider_scorer import CiderScorer  # 兜底
                    scorer = None
                if scorer is not None:
                    score, _ = scorer.compute_score(valid_gts, valid_res)
                    results['cider'] = float(score)
                else:
                    # 低配兜底（很少用到）
                    cs = CiderScorer(n=4, sigma=6.0)
                    for k in valid_gts.keys():
                        cs += (valid_res[k][0], valid_gts[k])
                    score, _ = cs.compute_score()
                    results['cider'] = float(score)
        except Exception as e:
            print(f"Error computing CIDEr-D: {e}")
            print("Note: CIDEr-D requires pycocoevalcap. Install with: pip install git+https://github.com/salaniz/pycocoevalcap")
            # ★ 提供替代实现，即使没有 pycocoevalcap 也能给出近似分数
            results['cider'] = compute_simple_cider(ref_texts, hyp_texts)
    # SPICE（慢，且依赖 Java；默认关闭）
    if use_spice:
        try:
            from pycocoevalcap.spice.spice import Spice
            spice_scorer = Spice()
            score, _ = spice_scorer.compute_score(gts, res)  # 返回平均分与逐样本
            results['spice'] = float(score)
        except Exception:
            results['spice'] = None
            print("Note: SPICE evaluation requires pycocoevalcap + Java(>=8). Install with:")
            print("  pip install pycocoevalcap")
            print("  java -version  # Check Java version")
    return results

# --- History logging (final) ---
class HistoryLogger:
    def __init__(self, path='checkpoints/training_history.json'):
        self.path = path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.history = []

    # 支持既能传 dict 也能传 kwargs
    def append(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            entry = args[0]
        elif not args:
            entry = kwargs
        else:
            raise TypeError("HistoryLogger.append() 只接受一个 dict 或纯 kwargs")
        self.history.append(entry)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    # 允许直接 for h in history: ...
    def __iter__(self):
        return iter(self.history)
    def __len__(self):
        return len(self.history)

    def get(self):
        return self.history

# --- Curves plotting (final) ---
def plot_curves(history, out_dir='plots', show_cider=False, show_spice=False):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]

    def _save(figpath):
        plt.tight_layout(); plt.savefig(figpath); plt.close()
        print(f"[plot] saved -> {os.path.abspath(figpath)}")

    # (1) train/val loss
    try:
        y_tr = [h.get('train_loss', float('nan')) for h in history]
        y_va = [h.get('val_loss',   float('nan')) for h in history]
        if any(x == x for x in y_tr) or any(x == x for x in y_va):
            plt.figure()
            plt.plot(epochs, y_tr, label='Train Loss')
            plt.plot(epochs, y_va, label='Val Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training/Validation Loss')
            plt.legend(); plt.grid(True, alpha=0.3)
            _save(os.path.join(out_dir, 'loss_curve.png'))
    except Exception as e:
        print(f"[plot] skip loss_curve: {e}")

    # (2) loss components
    try:
        has_lm  = any('train_lm_loss'  in h for h in history)
        has_itm = any('train_itm_loss' in h for h in history)
        has_itc = any('train_itc_loss' in h for h in history)
        if has_lm or has_itm or has_itc:
            plt.figure()
            if has_lm:
                plt.plot(epochs, [h.get('train_lm_loss',  float('nan')) for h in history], label='LM Loss')
            if has_itm:
                plt.plot(epochs, [h.get('train_itm_loss', float('nan')) for h in history], label='ITM Loss')
            if has_itc:
                plt.plot(epochs, [h.get('train_itc_loss', float('nan')) for h in history], label='ITC Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss Components')
            plt.legend(); plt.grid(True, alpha=0.3)
            _save(os.path.join(out_dir, 'loss_components.png'))
    except Exception as e:
        print(f"[plot] skip loss_components: {e}")

    # (3) 验证指标
    try:
        has_any_metric = any(('bleu1' in h) or ('bleu4' in h) or ('meteor' in h)
                             or ('cider' in h and show_cider)
                             or ('spice' in h and show_spice) for h in history)
        if has_any_metric:
            plt.figure()
            if any('bleu1' in h for h in history):
                plt.plot(epochs, [h.get('bleu1',  float('nan')) for h in history],
                         label='BLEU-1')
            if any('bleu4' in h for h in history):
                plt.plot(epochs, [h.get('bleu4',  float('nan')) for h in history],
                         label='BLEU-4')
            if any('meteor' in h for h in history):
                plt.plot(epochs, [h.get('meteor', float('nan')) for h in history],
                         label='METEOR')
            if show_cider and any('cider' in h for h in history):
                plt.plot(epochs, [h.get('cider',  float('nan')) for h in history],
                         label='CIDEr-D')
            if show_spice and any('spice' in h for h in history):
                plt.plot(epochs, [h.get('spice',  float('nan')) for h in history],
                         label='SPICE')
            plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Validation Metrics')
            plt.legend(); plt.grid(True, alpha=0.3)
            _save(os.path.join(out_dir, 'metrics_curve.png'))
    except Exception as e:
        print(f"[plot] skip metrics_curve: {e}")

    # (4) ITC temperature
    try:
        if any('itc_temperature' in h for h in history):
            plt.figure()
            plt.plot(epochs, [h.get('itc_temperature', float('nan')) for h in history], label='ITC Temperature')
            plt.xlabel('Epoch'); plt.ylabel('Temperature'); plt.title('ITC Temperature Evolution')
            plt.legend(); plt.grid(True, alpha=0.3)
            _save(os.path.join(out_dir, 'itc_temperature_curve.png'))
    except Exception as e:
        print(f"[plot] skip itc_temperature_curve: {e}")

def save_visual_debug(img1, img2, diff_map, cluster_masks, out_path, max_show=4):
    """
    img1/img2: [3,H,W] (0-1 or -1~1 都可) ; diff_map: [H,W]; cluster_masks: [K,H,W]
    """
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    def to_np(x):
        x = x.detach().float().cpu()
        if x.dim()==3 and x.size(0)==3:
            x = x.clamp(0,1).permute(1,2,0).numpy()
        else:
            x = x.numpy()
        return x
    I1, I2 = to_np(img1), to_np(img2)
    D = to_np(diff_map)
    K = min(cluster_masks.size(0), max_show)
    fig, axes = plt.subplots(2, K+1, figsize=(3*(K+1), 6))
    axes[0,0].imshow(I1); axes[0,0].set_title("Image A"); axes[0,0].axis('off')
    axes[1,0].imshow(I2); axes[1,0].set_title("Image B"); axes[1,0].axis('off')
    for i in range(K):
        axes[0,i+1].imshow(D, cmap='gray'); axes[0,i+1].imshow(to_np(cluster_masks[i]), alpha=0.4)
        axes[0,i+1].set_title(f"Cluster {i+1}"); axes[0,i+1].axis('off')
        axes[1,i+1].imshow(I2); axes[1,i+1].imshow(to_np(cluster_masks[i]), alpha=0.4)
        axes[1,i+1].axis('off')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

class DifferenceDetector(nn.Module):
    """
    Difference detector with clustering - consistent with spot-the-diff
    
    Features:
    - Adaptive thresholding using Otsu method with percentile fallback
    - Morphological operations for noise reduction
    - DBSCAN clustering for robust region detection
    """
    def __init__(self, cluster_embed_dim=512, max_clusters=5, eps=8, min_samples=8, 
                 verbose=False, diff_threshold_percentile=98.0, diff_threshold_multiplier=0.3):
        super().__init__()
        self.cluster_embed_dim = cluster_embed_dim
        self.max_clusters = max_clusters
        self.eps = eps
        self.min_samples = min_samples
        self.verbose = verbose
        
        # Store threshold parameters
        self.thres_pct = diff_threshold_percentile
        self.thres_mul = diff_threshold_multiplier
        
        # Difference encoder for cluster features
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, cluster_embed_dim),
            nn.LayerNorm(cluster_embed_dim)
        )

    def forward(self, diff_map):
        """
        Args:
            diff_map: [B, 224, 224] difference map
        Returns:
            cluster_features: [B, max_clusters, cluster_embed_dim]
            cluster_masks: [B, max_clusters, 224, 224]
            cluster_info: detailed cluster information
        """
        batch_size = diff_map.size(0)
        device = diff_map.device
        cluster_features = torch.zeros((batch_size, self.max_clusters, self.cluster_embed_dim), device=device)
        cluster_masks = torch.zeros((batch_size, self.max_clusters, 224, 224), device=device)
        cluster_info = []
        
        # Debug info (optional)
        if self.verbose:
            print(f"DifferenceDetector - batch_size: {batch_size}, max_clusters: {self.max_clusters}")
            print(f"DifferenceDetector - cluster_features shape: {cluster_features.shape}")
        
        for b in range(batch_size):
            single = diff_map[b].cpu().numpy()
            
            # --- Normalise to [0,1] per image for stable thresholding ---
            x = single
            x = (x - x.min()) / max(1e-6, (x.max() - x.min()))
            
            # --- Try Otsu; if fails (e.g., all equal), fallback to high percentile ---
            try:
                import cv2
                x8 = (x * 255).astype('uint8')
                _th, bin_map = cv2.threshold(x8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_map = (bin_map > 0).astype(np.float32)
            except Exception:
                thres = np.percentile(x, self.thres_pct)  # e.g. 99.5
                binary_map = (x > thres).astype(np.float32)
            
            # --- Morphological open: remove salt noise, close small holes ---
            try:
                import cv2
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                binary_map = cv2.morphologyEx((binary_map*255).astype('uint8'), cv2.MORPH_OPEN, k, iterations=1)
                binary_map = (binary_map > 0).astype(np.float32)
            except Exception:
                # Fallback if OpenCV not available
                pass
            
            coords = np.column_stack(np.where(binary_map > 0))
            info_list = []
            
            if len(coords) > 0:
                # DBSCAN clustering with original parameters
                clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(coords)
                labels = clustering.labels_
                unique = set(labels)
                if -1 in unique:
                    unique.remove(-1)  # Remove noise points
                
                # Sort clusters by size; we'll fill into contiguous slots (0..K-1)
                clusters = sorted(unique, key=lambda l: (labels==l).sum(), reverse=True)
                slot = 0
                for label in clusters:
                    if slot >= self.max_clusters:
                        break
                    mask = np.zeros((224, 224), dtype=np.float32)
                    pts = coords[labels == label]
                    mask[pts[:, 0], pts[:, 1]] = 1
                    
                    # Skip very small clusters
                    if len(pts) < 5:
                        continue
                    
                    mask_2d = torch.from_numpy(mask).to(device)                     # [224,224]
                    mask_for_encoder = mask_2d.unsqueeze(0).unsqueeze(0)            # [1,1,224,224]
                    encoder_device = next(self.diff_encoder.parameters()).device
                    feat = self.diff_encoder(mask_for_encoder.to(encoder_device)).squeeze(0)  # [512]
                    cluster_features[b, slot] = feat.to(device)
                    cluster_masks[b, slot] = mask_2d
                    
                    info_list.append({
                        'coords': pts,
                        'mask': mask,
                        'size': len(pts),
                        'cluster_id': slot + 1,
                        'bbox': [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()] if len(pts) > 0 else [0, 0, 0, 0],
                    })
                    slot += 1
            
            cluster_info.append(info_list)
        
        return cluster_features, cluster_masks, cluster_info

class EnhancedDiffBLIP2(nn.Module):
    """
    Enhanced BLIP2 model with difference detection and clustering
    """
    def __init__(self, blip2_model, max_clusters=5, device=None, 
                 diff_threshold_percentile=98.0, diff_threshold_multiplier=0.3,
                 dbscan_eps=8.0, dbscan_min_samples=8):
        super().__init__()
        if device is None:
            device = str(next(blip2_model.vision_model.parameters()).device)
        self.device = device
        self.max_clusters = max_clusters
        self.verbose = False
        self.blip2_model = blip2_model
        
        # Store difference detection parameters
        self.diff_threshold_percentile = diff_threshold_percentile
        self.diff_threshold_multiplier = diff_threshold_multiplier
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
        # Get BLIP2's visual feature dimension
        self.vision_dim = self.blip2_model.vision_model.config.hidden_size
        
        # Difference detector
        self.diff_detector = DifferenceDetector(
            max_clusters=max_clusters, 
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            verbose=False,
            diff_threshold_percentile=self.diff_threshold_percentile,
            diff_threshold_multiplier=self.diff_threshold_multiplier
        )
        
        # Feature fusion layer - updated for enhanced global features
        self.fusion_layer = nn.Linear(self.vision_dim * 3, 2560)  # Map to language model embedding dim
        
        # Geometric encoding MLP for cluster position/size features
        self.geo_mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU())
        
        # Cluster projection with geometric features: global + cluster + geo
        self.cluster_projection = nn.Linear(2560 + 512 + 64, 2560)
        self.ln = nn.LayerNorm(2560)
        
        # Alignment heads (optional) for ITM/ITC
        self.itm_head = nn.Sequential(nn.Linear(2560*2, 512), nn.ReLU(), nn.Linear(512, 1))
        self.itc_proj_v = nn.Linear(2560, 1024)
        self.itc_proj_t = nn.Linear(2560, 1024)
        self.aln_weight_itm = 0.5
        self.aln_weight_itc = 0.5
        
        # Learnable temperature for ITC
        # 初始化为 ln(1/τ0)，默认 τ0=0.07（CLIP 量级，exp≈14.2857），并在训练中围绕该目标做正则
        tau0 = float(getattr(self, "itc_temp_init", 0.07))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / max(1e-6, tau0)), dtype=torch.float32))
        self.itc_temp_target = torch.tensor(math.log(1.0 / max(1e-6, tau0)), dtype=torch.float32)
        self.itc_temp_reg = float(getattr(self, "itc_temp_reg", 1e-4))
        
        # Define prefix length for visual prefix (used in both forward and ITC/ITM)
        self.prefix_len = min(self.max_clusters, 4)
        
        # Move only custom modules to the same device (avoid breaking HF device_map sharding)
        self.diff_detector.to(self.device)
        self.fusion_layer.to(self.device)
        self.geo_mlp.to(self.device)
        self.cluster_projection.to(self.device)
        self.ln.to(self.device)
        self.itm_head.to(self.device)
        self.itc_proj_v.to(self.device)
        self.itc_proj_t.to(self.device)
        # logit_scale is already a parameter, no need to move separately
        
        # Freeze BLIP2 model parameters to save memory and prevent overfitting
        print("Freezing BLIP2 backbone parameters...")
        for param in self.blip2_model.parameters():
            param.requires_grad = False
        
        # Keep Q-Former frozen since we're not using it in forward pass
        # (We directly use vision_model → custom fusion → LLM)
        print("Keeping Q-Former frozen (not used in forward pass)...")
        for param in self.blip2_model.qformer.parameters():
            param.requires_grad = False
        
        # Freeze most language model parameters, only train the last few layers
        for name, param in self.blip2_model.language_model.named_parameters():
            if ("model.decoder.layers.31" in name) or \
               ("model.decoder.layers.30" in name) or \
               ("model.decoder.layers.29" in name) or \
               ("lm_head" in name):
                param.requires_grad = True         # Only change trainability
                # Don't change dtype, keep loaded half/bf16 to avoid mixed precision
            else:
                param.requires_grad = False        # Freeze remaining layers (keep half/bf16)
        
        # ★ Optional: Make trainable layers dtype consistent with chosen precision
        # Note: This requires external precision parameter, temporarily commented out
        # if hasattr(self, 'precision') and self.precision == 'bf16':
        #     for name, p in self.blip2_model.language_model.named_parameters():
        #         if p.requires_grad and p.dtype != torch.bfloat16:
        #             p.data = p.data.to(torch.bfloat16)
        
        # Keep our custom modules trainable
        print("Keeping custom modules trainable...")
        for param in self.diff_detector.parameters():
            param.requires_grad = True
            param.data = param.data.to(dtype=torch.float32)  # Ensure float32 for AMP
        for param in self.fusion_layer.parameters():
            param.requires_grad = True
            param.data = param.data.to(dtype=torch.float32)  # Ensure float32 for AMP
        for param in self.cluster_projection.parameters():
            param.requires_grad = True
            param.data = param.data.to(dtype=torch.float32)  # Ensure float32 for AMP
        for param in self.ln.parameters():
            param.requires_grad = True
            param.data = param.data.to(dtype=torch.float32)  # Ensure float32 for AMP
        for param in [self.logit_scale]:
            param.requires_grad = True
            param.data = param.data.to(dtype=torch.float32)  # Ensure float32 for AMP
        
        # Enable gradient checkpointing to save memory
        print("Enabling gradient checkpointing...")
        # Skip Q-Former since it's frozen and not used in forward pass
        if hasattr(self.blip2_model, 'language_model'):
            self.blip2_model.language_model.gradient_checkpointing_enable()
            # Disable use_cache to avoid warning with gradient checkpointing
            if hasattr(self.blip2_model.language_model, "config"):
                self.blip2_model.language_model.config.use_cache = False
            print("  - Language model gradient checkpointing enabled.")
        
        # Debug device info
        print(f"EnhancedDiffBLIP2 - device: {device}")
        print(f"EnhancedDiffBLIP2 - diff_detector device: {next(self.diff_detector.parameters()).device}")
        print(f"EnhancedDiffBLIP2 - fusion_layer device: {next(self.fusion_layer.parameters()).device}")
        
        # Print trainable parameters count
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"EnhancedDiffBLIP2 - Trainable parameters: {trainable_params:,} / {total_params:,}")

    def encode_images(self, img1, img2):
        """Encode two images and return their difference features"""
        device = next(self.blip2_model.vision_model.parameters()).device
        
        # Ensure correct format
        if img1.dim() == 5:
            img1 = img1.squeeze(0).squeeze(0)   # -> [3,H,W]
        if img2.dim() == 5:
            img2 = img2.squeeze(0).squeeze(0)
        if img1.dim() == 3:                      # Restore batch dimension
            img1 = img1.unsqueeze(0)             # -> [1,3,H,W]
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Handle grayscale images
        if img1.size(1) == 1:
            img1 = img1.repeat(1, 3, 1, 1)
        if img2.size(1) == 1:
            img2 = img2.repeat(1, 3, 1, 1)
        
        # Ensure images are on correct device
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        if self.verbose:
            print(f"encode_images - device: {device}")
            print(f"encode_images - img1 device: {img1.device}")
            print(f"encode_images - img2 device: {img2.device}")
        
        # Encode with BLIP2 vision model
        with torch.no_grad():
            out1 = self.blip2_model.vision_model(pixel_values=img1)
            out2 = self.blip2_model.vision_model(pixel_values=img2)
            if hasattr(out1, "pooler_output") and out1.pooler_output is not None:
                vf1 = out1.pooler_output
            else:
                vf1 = out1.last_hidden_state.mean(dim=1)
            if hasattr(out2, "pooler_output") and out2.pooler_output is not None:
                vf2 = out2.pooler_output
            else:
                vf2 = out2.last_hidden_state.mean(dim=1)
        
        return vf1, vf2

    def compute_difference_map(self, img1, img2):
        """
        Compute difference map between image pair
        """
        # Handle different input shapes
        if img1.dim() == 5:
            img1 = img1.squeeze(1)
            img2 = img2.squeeze(1)
        
        # Ensure we have batch dimension
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        batch_size = img1.size(0)
        
        if self.verbose:
            print(f"compute_difference_map - img1 shape: {img1.shape}, img2 shape: {img2.shape}")
            print(f"compute_difference_map - batch_size: {batch_size}")
        diff_maps = []
        
        for i in range(batch_size):
            img1_single = img1[i]
            img2_single = img2[i]
            
            # Convert to grayscale if needed
            if img1_single.size(0) == 3:
                gray1 = 0.299 * img1_single[0] + 0.587 * img1_single[1] + 0.114 * img1_single[2]
                gray2 = 0.299 * img2_single[0] + 0.587 * img2_single[1] + 0.114 * img2_single[2]
            else:
                gray1, gray2 = img1_single, img2_single
            
            # Compute difference
            diff_map = torch.abs(gray1 - gray2)
            diff_maps.append(diff_map)
        
        # Stack all difference maps
        if batch_size == 1:
            diff_map = diff_maps[0].unsqueeze(0)  # Add batch dimension
        else:
            diff_map = torch.stack(diff_maps, dim=0)
        
        if self.verbose:
            print(f"compute_difference_map - diff_map shape: {diff_map.shape}")
        return diff_map

    def forward(self, img1, img2, captions=None, attention_mask=None, sup_mask=None, return_loss=False):
        """
        Forward pass with enhanced visual token integration
        """
        # Encode images
        vf1, vf2 = self.encode_images(img1, img2)
        
        # Compute difference map and clustering
        diff_map = self.compute_difference_map(img1, img2)
        cluster_feats, cluster_masks, cluster_info = self.diff_detector(diff_map)
        
        # Enhanced global visual features with difference information
        global_feat = torch.cat([vf1, vf2, vf2 - vf1], dim=1)  # [B, vision_dim * 3]
        # Ensure consistent dtype for fusion layer
        global_feat = global_feat.to(dtype=torch.float32)
        global_feat = self.fusion_layer(global_feat)
        global_feat = self.ln(global_feat)
        
        # === Build geometric encoding tensor [B, max_clusters, 5] ===
        B = global_feat.size(0)
        geo_feats = torch.zeros(B, self.max_clusters, 5, device=global_feat.device, dtype=global_feat.dtype)
        for b in range(B):
            info_list = cluster_info[b] if b < len(cluster_info) else []
            for i, info in enumerate(info_list[:self.max_clusters]):
                y1, x1, y2, x2 = info['bbox']  # note: your bbox is (row_min, col_min, row_max, col_max)
                cy = (y1 + y2) / 2.0 / 224.0
                cx = (x1 + x2) / 2.0 / 224.0
                h = max(y2 - y1, 1) / 224.0
                w = max(x2 - x1, 1) / 224.0
                area = float(info['size']) / (224.0 * 224.0)
                geo_feats[b, i] = torch.tensor([cx, cy, w, h, area], device=global_feat.device, dtype=global_feat.dtype)

        fused_feats_list = []
        for i in range(self.max_clusters):
            if i < cluster_feats.size(1):
                cf = cluster_feats[:, i].to(global_feat.device)        # [B, 512]
                pe = self.geo_mlp(geo_feats[:, i])                     # [B, 64]
                cluster_enhanced = torch.cat([global_feat, cf, pe], dim=1)  # [B, 2560+512+64]
                cluster_enhanced = self.cluster_projection(cluster_enhanced)
                cluster_enhanced = self.ln(cluster_enhanced)
                cluster_enhanced = torch.nan_to_num(cluster_enhanced, nan=0.0, posinf=100.0, neginf=-100.0)
            else:
                cluster_enhanced = global_feat
            fused_feats_list.append(cluster_enhanced.unsqueeze(1))
        
        fused_feats = torch.cat(fused_feats_list, dim=1)  # [B, max_clusters, 2560]
        
        if return_loss and captions is not None:
            # === Multi-cluster visual prefix ===
            visual_prefix = fused_feats[:, :self.prefix_len, :]  # [B, K, hidden]
            if visual_prefix.size(1) < self.prefix_len:
                pad = global_feat.unsqueeze(1).repeat(1, self.prefix_len - visual_prefix.size(1), 1)
                visual_prefix = torch.cat([visual_prefix, pad], dim=1)
            # replace invalid clusters (area<=0) with global_feat
            areas = geo_feats[:, :self.prefix_len, 4].unsqueeze(-1)  # [B,K,1]
            invalid = (areas <= 0)
            if invalid.any():
                visual_prefix = torch.where(invalid, global_feat.unsqueeze(1).expand_as(visual_prefix), visual_prefix)
            
            # Get text embeddings (captions contain prompt+caption)
            lm_emb = self.blip2_model.language_model.get_input_embeddings()
            lm_device = next(lm_emb.parameters()).device
            lm_dtype  = next(lm_emb.parameters()).dtype
            # ★ Ensure embedding indices are integers and on LM device
            if captions.dtype != torch.long:
                captions = captions.long()
            captions = captions.to(device=lm_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=lm_device, dtype=torch.long)
            text_emb = lm_emb(captions).to(device=lm_device, dtype=lm_dtype)            # [B, L, H]
            
            # Align visual prefix to LM device and align distribution to LM token embeddings (mean/variance)
            lm_w = lm_emb.weight.detach()
            w_mean = lm_w.mean().to(lm_device, dtype=torch.float32)
            w_std  = lm_w.std().clamp_min(1e-6).to(lm_device, dtype=torch.float32)
            vp = torch.nn.functional.layer_norm(visual_prefix.float(), (visual_prefix.size(-1),))
            vp = (vp * w_std + w_mean)                                  # Make prefix distribution close to LM word vectors
            # Tighter clamping: around w_mean ±3σ
            lo = (w_mean - 3.0 * w_std).to(vp.dtype)
            hi = (w_mean + 3.0 * w_std).to(vp.dtype)
            vp = torch.clamp(vp, min=lo, max=hi)
            # Linear warmup: 0→prefix_scale
            base = float(getattr(self, 'prefix_scale', 0.25))
            warm = int(getattr(self, 'prefix_warmup_steps', 200))
            factor = 1.0
            if self.training and warm > 0 and hasattr(self, '_train_step'):
                factor = min(1.0, float(self._train_step) / float(warm))
            visual_prefix = (vp * (base * factor)).to(device=lm_device, dtype=lm_dtype)

            # Concatenate
            input_embeds = torch.cat([visual_prefix, text_emb], dim=1)
            # ★ 最终消毒：把 NaN/Inf 变为有限数，并限制范围（半精度更稳）
            input_embeds = torch.nan_to_num(input_embeds, nan=0.0, posinf=100.0, neginf=-100.0)
            input_embeds = torch.clamp(input_embeds, -100.0, 100.0)
            
            # === 构造 attention_mask（前缀视作可见）===
            B, L = captions.size()
            if attention_mask is not None:
                prefix_mask = torch.ones((B, self.prefix_len), dtype=attention_mask.dtype, device=attention_mask.device)
                full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1).to(lm_device)
            else:
                full_attention_mask = None
            
            # === 前向（不传 labels，取 logits 自己算 CE）===
            outputs = self.blip2_model.language_model(
                input_ids=None,
                inputs_embeds=input_embeds,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            logits = outputs.logits   # [B, self.prefix_len+L, V]

            # === 使用数据集提供的监督掩码；若缺失则按旧逻辑推断 ===
            if sup_mask is not None:
                sup_mask = sup_mask.to(device=captions.device, dtype=torch.bool)
                # 对齐长度
                if sup_mask.dim()==1: sup_mask = sup_mask.unsqueeze(0)
                if sup_mask.size(1) != L:
                    # 以实际 L 为准截断/补齐（极少发生）
                    if sup_mask.size(1) > L:
                        sup_mask = sup_mask[:, :L]
                    else:
                        pad = torch.zeros((sup_mask.size(0), L - sup_mask.size(1)),
                                          dtype=torch.bool, device=sup_mask.device)
                        sup_mask = torch.cat([sup_mask, pad], dim=1)
                # 每条样本至少 1 个监督 token
                for i in range(B):
                    if sup_mask[i].sum() == 0:
                        sup_mask[i, max(0, L-1)] = True
            else:
                # 回退：按 prompt_len + attention_mask 构造
                prompt_len = self._cached_prompt_len if hasattr(self,'_cached_prompt_len') else torch.zeros(B, dtype=torch.long, device=captions.device)
                sup_mask = torch.ones_like(captions, dtype=torch.bool)
                for i in range(B):
                    pl = int(prompt_len[i].item())
                    pl = max(0, min(pl, L-1))
                    if pl > 0:
                        sup_mask[i, :pl] = False
                if attention_mask is not None:
                    sup_mask &= (attention_mask > 0)
                for i in range(B):
                    if sup_mask[i].sum() == 0:
                        j = (int(attention_mask[i].sum().item()) - 1) if attention_mask is not None and attention_mask[i].sum() > 0 else (L - 1)
                        sup_mask[i, j] = True

            # === Causal shift：用位置 (self.prefix_len-1+t) 的 logits 预测 captions[:, t] ===
            idx_b, idx_t = torch.nonzero(sup_mask, as_tuple=True)             # 选出需要监督的 (b,t)
            
            # ★ 跳过空 caption 样本的 LM 损失计算
            if idx_b.shape[0] == 0:
                # 所有样本都没有有效监督位置，跳过 LM 损失
                lm_loss = torch.zeros((), device=lm_device, dtype=torch.float32)
                if self.training and not hasattr(self, '_warn_empty_sup'):
                    print("[Warn] All samples have empty captions. Skipping LM loss calculation.")
                    self._warn_empty_sup = True
            else:
                # 取对应位置的 logits：注意要用 t 的"前一位"作为预测源
                logits_text_shift = logits[idx_b, (self.prefix_len - 1) + idx_t, :]    # [N, V]
                targets = captions[idx_b, idx_t]                                  # [N]
                
                # ★ Debug 打印：监控监督信号
                if self.training and hasattr(self, '_debug_step') and self._debug_step % 100 == 0:
                    print(f"[DEBUG] LM supervision:")
                    print(f"  - sup_mask shape: {sup_mask.shape}")
                    print(f"  - sup_mask sum per sample: {sup_mask.sum(dim=1)}")
                    print(f"  - valid positions: {idx_b.shape[0]}")
                    print(f"  - logits_text_shift shape: {logits_text_shift.shape}")
                    print(f"  - targets shape: {targets.shape}")
                    print(f"  - targets range: [{targets.min()}, {targets.max()}]")
                    self._debug_step = getattr(self, '_debug_step', 0) + 1
                
                # ★ 在 FP32 下算 CE，避免半精度下的下溢问题
                with torch.cuda.amp.autocast(enabled=False):
                    lm_loss = F.cross_entropy(
                        logits_text_shift.float(),  # 强制 FP32
                        targets,
                        ignore_index=-100,
                        reduction="mean"
                    )
            
            # ★ 数值稳定保护：仅对该 batch 生效，别全局关掉 LM
            if not torch.isfinite(lm_loss):
                if not hasattr(self, "_warn_lm_nan"):
                    print("[Warn] LM loss non-finite on this batch. Zeroing *this* term.")
                    self._warn_lm_nan = True
                lm_loss = logits_text_shift.new_zeros(())
            # === 基于 sup_mask 的聚合：对有监督的 token 做加权平均（无 sup_mask 则回退到最后 token）===
            # 注意：sup_mask 已在数据集构建时跳过特殊符（BOS/EOS/PAD/UNK），只标记有意义的 caption tokens
            last_hid = outputs.hidden_states[-1]  # [B, T, H]
            B, T, H = last_hid.shape
            
            use_sup_agg = (sup_mask is not None) and (sup_mask.any().item())
            if use_sup_agg:
                # 对齐维度：sup_mask 只覆盖 caption 段，需在左侧补上 self.prefix_len 的 False
                sup_mask_ = sup_mask.to(last_hid.device, dtype=last_hid.dtype)  # [B, L]
                if sup_mask_.dim() == 1:
                    sup_mask_ = sup_mask_.unsqueeze(0)
                # caption 段在 hidden 里的起点索引是 self.prefix_len
                if sup_mask_.size(1) > (T - self.prefix_len):
                    sup_mask_ = sup_mask_[:, : (T - self.prefix_len)]
                elif sup_mask_.size(1) < (T - self.prefix_len):
                    pad_len = (T - self.prefix_len) - sup_mask_.size(1)
                    sup_mask_ = torch.cat([sup_mask_, torch.zeros(B, pad_len, device=sup_mask_.device, dtype=sup_mask_.dtype)], dim=1)
                pad_prefix = torch.zeros(B, self.prefix_len, device=sup_mask_.device, dtype=sup_mask_.dtype)
                weights = torch.cat([pad_prefix, sup_mask_], dim=1)            # [B, T]
                # 归一化权重
                weights_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6) # [B, 1]
                weights = weights / weights_sum
                # 加权平均得到 pooled_text
                pooled_text = (last_hid * weights.unsqueeze(-1)).sum(dim=1)    # [B, H]
                
                # Debug: Log when sup_mask aggregation is used
                if self.training and not hasattr(self, '_log_sup_agg'):
                    print(f"[Info] Using supervised mask aggregation for pooled_text (B={B}, T={T}, prefix_len={self.prefix_len})")
                    self._log_sup_agg = True
            else:
                # 回退：使用最后一个有效 token
                if full_attention_mask is None:
                    eff_mask = torch.ones(B, T, device=last_hid.device, dtype=torch.long)
                else:
                    eff_mask = full_attention_mask.to(last_hid.device)
                last_idx = eff_mask.sum(dim=1).clamp(min=1) - 1
                pooled_text = last_hid[torch.arange(B, device=last_hid.device), last_idx]
                
                # Debug: Log when fallback is used
                if self.training and not hasattr(self, '_log_fallback'):
                    print(f"[Info] Using last token fallback for pooled_text (B={B}, T={T})")
                    self._log_fallback = True

            # ITM: positive pairs vs. hard negatives (shuffled captions + weakened visual cues)
            B = global_feat.size(0)
            aln_device = next(self.itm_head.parameters()).device
            aln_dtype  = next(self.itm_head.parameters()).dtype
            pooled_text = pooled_text.to(device=aln_device, dtype=aln_dtype)
            # 用聚类增强后的视觉 prefix 做图文对齐（平均聚合）
            gfeat_aln = fused_feats[:, :self.prefix_len, :].mean(dim=1).to(device=aln_device, dtype=aln_dtype)
            # --- 数值消毒，防止极端 batch 里出现 NaN/Inf 级联到对齐损失 ---
            pooled_text = torch.nan_to_num(pooled_text, nan=0.0, posinf=1e3, neginf=-1e3)
            gfeat_aln   = torch.nan_to_num(gfeat_aln,   nan=0.0, posinf=1e3, neginf=-1e3)
            # Skip ITM/ITC when B=1 (maintain numerical stability, avoid pseudo-negative = positive)
            if B > 1:
                idx_perm = torch.randperm(B, device=aln_device)
                neg_text = pooled_text[idx_perm]

                # Harden negatives a bit: drop some visual cues
                drop_mask = (torch.rand_like(gfeat_aln[..., :1]) < 0.3).float()  # 30% drop
                gfeat_pos = gfeat_aln
                gfeat_neg = gfeat_aln * (1.0 - drop_mask)  # weaken visuals for neg

                pos_logit = self.itm_head(torch.cat([gfeat_pos, pooled_text], dim=1)).squeeze(-1)
                neg_logit = self.itm_head(torch.cat([gfeat_neg, neg_text],   dim=1)).squeeze(-1)

                pos_logit = torch.nan_to_num(pos_logit, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
                neg_logit = torch.nan_to_num(neg_logit, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

                itm_logits = torch.cat([pos_logit, neg_logit], dim=0)
                itm_labels = torch.cat([
                    torch.ones(B, device=aln_device),
                    torch.zeros(B, device=aln_device)
                ])

                with torch.cuda.amp.autocast(enabled=False):
                    # Tiny label smoothing (p=0.05) helps escape 0.693 plateau early
                    # 0.693 ≈ ln(2) is the theoretical random binary classification loss
                    eps = 0.05
                    targets = itm_labels.float()
                    targets = targets * (1 - eps) + 0.5 * eps
                    itm_loss = F.binary_cross_entropy_with_logits(itm_logits.float(), targets)
            else:
                itm_loss = torch.zeros(1, device=aln_device)
                if self.training and not hasattr(self, '_warn_batch_size_1'):
                    print("[Warn] Batch size = 1, skipping ITM/ITC alignment losses")
                    self._warn_batch_size_1 = True

            # ITC: InfoNCE
            if B > 1:
                v = self.itc_proj_v(gfeat_aln)
                t = self.itc_proj_t(pooled_text)
                v = torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)
                t = torch.nan_to_num(t, nan=0.0, posinf=1e3, neginf=-1e3)
                v = F.normalize(v, dim=1, eps=1e-6)
                t = F.normalize(t, dim=1, eps=1e-6)
                
                # ★ 在 FP32 下计算 logits，避免半精度下的数值不稳定
                with torch.cuda.amp.autocast(enabled=False):
                    logits_vt = self.logit_scale.exp().clamp(1.0, 100.0) * (v.float() @ t.float().t())
                
                # row-wise stabilizer: subtract max (softmax-invariant)
                logits_vt = logits_vt - logits_vt.detach().max(dim=1, keepdim=True)[0]
                labels_itc = torch.arange(B, device=aln_device)
                itc_loss = (F.cross_entropy(logits_vt.float(), labels_itc) + 
                            F.cross_entropy(logits_vt.t().float(), labels_itc)) * 0.5
            else:
                itc_loss = torch.zeros(1, device=aln_device)
                # Warning already printed for ITM above
            
            # —— 1) clamp logit_scale（等价于夹住温度范围）——
            # 温度 T 的允许范围：T_min=0.03, T_max=0.20 （可按需改）
            T_min, T_max = 0.03, 0.20
            logit_scale_min = math.log(1.0 / T_max)  # ≈ log(5)  ≈ 1.609
            logit_scale_max = math.log(1.0 / T_min)  # ≈ log(33) ≈ 3.497
            with torch.no_grad():
                self.logit_scale.clamp_(logit_scale_min, logit_scale_max)
            
            # —— 2) ITC 正则（朝向目标温度 T0=itc_temp_init）——
            if self.itc_temp_reg > 0:
                target_logit_scale = math.log(1.0 / max(1e-6, self.itc_temp_init))
                itc_temp_reg_loss = self.itc_temp_reg * (self.logit_scale - target_logit_scale) ** 2
                itc_loss = itc_loss + itc_temp_reg_loss

            # --- 兜底：若某分量出现 non-finite（极少数 batch），将其置 0 并给出一次性提示 ---
            if not torch.isfinite(itm_loss): 
                if not hasattr(self, "_warn_itm_nan"):
                    print("[Warn] ITM loss became non-finite on a batch. Zeroing this term for this step.")
                    self._warn_itm_nan = True
                itm_loss = torch.zeros_like(itm_loss)
            if not torch.isfinite(itc_loss):
                if not hasattr(self, "_warn_itc_nan"):
                    print("[Warn] ITC loss became non-finite on a batch. Zeroing this term for this step.")
                    self._warn_itc_nan = True
                itc_loss = torch.zeros_like(itc_loss)
            loss = lm_loss + self.aln_weight_itm * itm_loss + self.aln_weight_itc * itc_loss
            return loss, lm_loss.detach(), itm_loss.detach(), itc_loss.detach(), cluster_info, cluster_masks
        else:
            # Inference: return fused features for generation
            return fused_feats, cluster_info, cluster_masks

    def generate(self, img1, img2, prompt="Diff: ", 
                tokenizer=None, max_new_tokens=50):
        """
        Generate a single high-quality difference caption for the whole image pair.
        """
        self.eval()
        with torch.no_grad():
            # Get fused features and cluster info
            fused_feats, cluster_info, cluster_masks = self.forward(img1, img2, return_loss=False)
            
            if tokenizer is None:
                tokenizer = get_global_tokenizer()
            
            # Use top-K clusters as visual prefix (一致于训练)
            visual_prefix = fused_feats[:, :self.prefix_len, :]
            if visual_prefix.size(1) < self.prefix_len:
                pad = visual_prefix[:, :1, :].repeat(1, self.prefix_len - visual_prefix.size(1), 1)
                visual_prefix = torch.cat([visual_prefix, pad], dim=1)
            # 同训练：无效簇回填
            # 这里需要与 forward 里一致的 geo_feats；简化做法：若出现全零行则回填
            zero_rows = (visual_prefix.abs().sum(dim=-1, keepdim=True) == 0)
            if zero_rows.any():
                # 取 batch 内的 global proxy：用可见 token 的均值近似
                global_proxy = visual_prefix.mean(dim=1, keepdim=True)
                visual_prefix = torch.where(zero_rows, global_proxy.expand_as(visual_prefix), visual_prefix)
            
            # Tokenize the prompt, exclude special tokens to avoid double [BOS]
            lm_emb = self.blip2_model.language_model.get_input_embeddings()
            lm_device = next(lm_emb.parameters()).device
            lm_dtype  = next(lm_emb.parameters()).dtype
            input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(lm_device)
            text_emb  = lm_emb(input_ids).to(dtype=lm_dtype)                              # [1, L, H]
            
            # 分布对齐 + 消毒 + 裁剪
            lm_w = lm_emb.weight.detach()
            w_mean = lm_w.mean().to(lm_device, dtype=torch.float32)
            w_std  = lm_w.std().clamp_min(1e-6).to(lm_device, dtype=torch.float32)
            vp = torch.nn.functional.layer_norm(visual_prefix.float(), (visual_prefix.size(-1),))
            # 与训练一致：先仿射到 LM 词向量统计，再以 w_mean 为中心限幅
            vp = (vp * w_std + w_mean).to(dtype=lm_dtype)
            lo = (w_mean - 3.0 * w_std).to(vp.dtype)
            hi = (w_mean + 3.0 * w_std).to(vp.dtype)
            vp = torch.clamp(vp, min=lo, max=hi)
            scale = float(getattr(self, 'prefix_scale', 0.25))
            vp = vp * scale
            inputs_embeds = torch.cat([vp, text_emb], dim=1)
            inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0, posinf=100.0, neginf=-100.0)
            inputs_embeds = torch.clamp(inputs_embeds, -100.0, 100.0)
            
            # Build attention mask (all ones)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=lm_device)
            
            # Generate only one description for the difference
            generated_ids = self.blip2_model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # 先关采样以提升贴图度
                num_beams=3,           # Reduced from 5 for faster evaluation
                length_penalty=0.7,    # Reduced from 0.9 for better length balance
                no_repeat_ngram_size=3,
                min_new_tokens=4,      # Ensure minimum meaningful output
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # Remove prefix part from output for clarity
            output_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up the output
            if prompt in output_caption:
                output_caption = output_caption.replace(prompt, "").strip()
            
            # Remove common template phrases
            import re
            patterns = [
                r"^Describe.*?:", r"^The main difference is", r"^In the highlighted region,?", 
                r"^\d+\.", r"^Region \d+:?", r"^Cluster \d+:?", r"^Visual difference", r"^Difference in",
                r"^This region", r"^The highlighted", r"^Between two images",
                r"^A difference", r"^There is", r"^Here is",
                r"main difference.*?:?", r"the main difference", r"main difference",
                r"what changed\?", r"what is different\?", r"what changed between these images\?",
                r"describe the difference", r"describe the main difference", r"what is the difference",
                r"what has changed", r"describe the main change", r"what is the key difference",
                r"in this region", r"in this area", r"in this place", r"in this country",
                r"this region", r"this area", r"this place", r"this country",
                r"the main difference in this region", r"the main difference in this area"
            ]
            for p in patterns:
                output_caption = re.sub(p, "", output_caption, flags=re.IGNORECASE)
            
            # Clean up punctuation and whitespace
            output_caption = re.sub(r'[:\s]*$', '', output_caption)
            output_caption = output_caption.strip(" :,\n\t")
            output_caption = output_caption.replace("\n", " ")
            
            # Remove excessive repetition
            output_caption = re.sub(r'\s+', ' ', output_caption)  # Normalize whitespace
            output_caption = re.sub(r'([^\s])\1{2,}', r'\1', output_caption)  # Remove character repetition
            
            # Fallback if no meaningful output
            if not output_caption or len(output_caption.strip()) < 3:
                output_caption = "A significant difference was detected between the images."
            
            return [output_caption.strip()], cluster_masks, cluster_info

class EnhancedDiffDataset(Dataset):
    """
    Dataset for difference detection training
    """
    def __init__(self, data_dir, split='train', max_samples=None, max_length=24):
        self.data_dir = data_dir
        self.split = split  # 'train', 'val', or 'test'
        self.max_samples = max_samples
        self.max_length = max_length
        self.tokenizer = get_global_tokenizer()
        self.processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
        
        # Load data
        self.samples = self.load_data()
        
    def load_data(self):
        """Load dataset based on split"""
        samples = []
        
        # Find all image pairs and annotations
        image_dir = os.path.join(self.data_dir, 'resized_images')
        annotation_file = os.path.join(self.data_dir, 'annotations', f'{self.split}.json')
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                
            # Apply max_samples limit based on total available data
            if self.max_samples:
                # Calculate the proportion to use based on total available data
                total_available = len(annotations)
                if self.max_samples > total_available:
                    print(f"Warning: max_samples ({self.max_samples}) > total available ({total_available}) for {self.split} split. Using all available data.")
                    max_to_use = total_available
                else:
                    max_to_use = self.max_samples
                annotations = annotations[:max_to_use]
                print(f"Using {len(annotations)} samples from {self.split} split (requested: {self.max_samples}, available: {total_available})")
            else:
                print(f"Using all {len(annotations)} samples from {self.split} split")
                
            for annotation in annotations:
                img_id = annotation['img_id']
                sentences = annotation['sentences']
                
                # Try different image naming patterns
                possible_img1_paths = [
                    os.path.join(image_dir, f"{img_id}.png"),
                    os.path.join(image_dir, f"{img_id}_2.png"),
                    os.path.join(image_dir, f"{img_id}_before.jpg"),
                    os.path.join(image_dir, f"{img_id}_before.png")
                ]
                
                possible_img2_paths = [
                    os.path.join(image_dir, f"{img_id}_diff.jpg"),
                    os.path.join(image_dir, f"{img_id}_after.jpg"),
                    os.path.join(image_dir, f"{img_id}_after.png")
                ]
                
                # Find existing image pair
                img1_path = None
                img2_path = None
                
                for path in possible_img1_paths:
                    if os.path.exists(path):
                        img1_path = path
                        break
                        
                for path in possible_img2_paths:
                    if os.path.exists(path):
                        img2_path = path
                        break
                
                # Check if both images exist
                if img1_path and img2_path:
                    # Use the first sentence as caption, or join multiple sentences
                    if sentences and any(s.strip() for s in sentences):
                        caption = ' '.join(sentences)
                    else:
                        # 用内容相关描述替代指令性提示，避免混淆模型
                        caption = "There are no notable differences between these images."
                    
                    samples.append({
                        'img1_path': img1_path,
                        'img2_path': img2_path,
                        'caption': caption
                    })
        else:
            print(f"Warning: Annotation file {annotation_file} not found for {self.split} split")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        img1 = Image.open(sample['img1_path']).convert('RGB')
        img2 = Image.open(sample['img2_path']).convert('RGB')
        
        # Preprocessing
        processed1 = self.processor(images=img1, return_tensors='pt')
        processed2 = self.processor(images=img2, return_tensors='pt')
        
        # Prompt / Caption 分开编码，确保 caption 至少保留 1 个 token
        prompt = "Diff: "
        toks_prompt = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        prompt_len = int(toks_prompt.size(0))
        max_cap_len = max(1, self.max_length - prompt_len)
        toks_cap = self.tokenizer(sample['caption'],
                                  add_special_tokens=False,
                                  truncation=True,
                                  max_length=max_cap_len,
                                  return_tensors='pt')['input_ids'][0]
        # 拼接并补齐
        full_ids = torch.cat([toks_prompt, toks_cap], dim=0)
        attn = torch.ones_like(full_ids)
        if full_ids.size(0) < self.max_length:
            pad_len = self.max_length - full_ids.size(0)
            full_ids = torch.cat([full_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=full_ids.dtype)], dim=0)
            attn = torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)], dim=0)
        # 监督掩码：仅 caption 段为 True（跳过特殊符如 BOS/EOS 和填充符）
        sup_mask = torch.zeros(self.max_length, dtype=torch.bool)
        cap_len = min(int(toks_cap.size(0)), self.max_length - prompt_len)
        if cap_len > 0:
            # 标记 caption 段为监督区域，但跳过特殊符
            for i in range(prompt_len, prompt_len + cap_len):
                if i < len(full_ids):
                    token_id = full_ids[i].item()
                    # 跳过特殊符：BOS, EOS, PAD, UNK 等
                    if token_id not in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, 
                                       self.tokenizer.pad_token_id, self.tokenizer.unk_token_id]:
                        sup_mask[i] = True
        else:
            # 极端兜底：至少留 1 个 token 监督（若 prompt_len 已达上限）
            sup_mask[max(0, min(self.max_length - 1, prompt_len))] = True
        encoded = {'input_ids': full_ids.unsqueeze(0), 'attention_mask': attn.unsqueeze(0)}
        
        return {
            'image1': processed1['pixel_values'].squeeze(0),
            'image2': processed2['pixel_values'].squeeze(0),
            'captions': encoded['input_ids'].squeeze(0),             # [L]
            'attention_mask': encoded['attention_mask'].squeeze(0),  # [L]
            'prompt_len': torch.tensor(prompt_len, dtype=torch.long),# 仅做日志/可视化需要
            'sup_mask': sup_mask,                                    # ★ 明确的监督掩码
            'original_caption': sample['caption']
        }

def train_model(model, train_loader, val_loader, args):
    """
    Training function with validation and early stopping
    """
    import gc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # decide autocast dtype per args
    if torch.cuda.is_available() and args.precision != 'fp32':
        if args.precision == 'bf16' and torch.cuda.is_bf16_supported():
            amp_ctx = lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            amp_ctx = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        amp_ctx = contextlib.nullcontext
    
    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.set_per_process_memory_fraction(0.7)  # Increase memory fraction
            print(f"CUDA memory fraction set to 0.7")
        except Exception as e:
            print(f"Warning: Could not set CUDA memory fraction: {e}")
            print("Continuing with default memory settings...")
        # Enable Flash Attention for better memory efficiency if available
        # 可选：按需启用 Flash/Memory-efficient SDP
        try:
            torch.backends.cuda.enable_flash_sdp(args.enable_flash_sdp)
            torch.backends.cuda.enable_mem_efficient_sdp(args.enable_flash_sdp)
        except Exception:
            pass
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        
        # enable TF32 for extra stability/speed on Ampere+
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        
        # Print memory info
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
    
    # If using HF device_map dispatch, skip full model migration to avoid breaking device_map
    if not (hasattr(model, "blip2_model") and hasattr(model.blip2_model, "hf_device_map")):
        model = model.to(device)
    else:
        print("Detected HF device_map dispatch; skip model.to(device).")
    
    # Optimizer with separate parameter groups for different learning rates
    # Head/alignment layers need higher LR to escape random alignment plateau
    # Collect parameters
    head_names = ['itm_head', 'itc_proj_v', 'itc_proj_t', 'fusion_layer',
                  'cluster_projection', 'geo_mlp', 'ln', 'diff_detector', 'logit_scale']
    head_params = []
    lm_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if any(hn in n for hn in head_names):
            head_params.append(p)
        else:
            lm_params.append(p)
    
    optimizer = torch.optim.AdamW(
        [
            {'params': head_params, 'lr': args.learning_rate * 10.0},
            {'params': lm_params,   'lr': args.learning_rate},
        ],
        weight_decay=args.weight_decay
    )
    
    print(f"Optimizer parameter groups:")
    print(f"  - Head/alignment params: {len(head_params)} params, LR: {args.learning_rate * 10.0:.2e}")
    print(f"  - Language model params: {len(lm_params)} params, LR: {args.learning_rate:.2e}")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.lr_scheduler_step_size,
        gamma=args.lr_scheduler_gamma
    )
    
    # Initialize AMP scaler for mixed precision training
    # 如果优化器里含有 FP16 参数（如 HF 半精度 LM 的可训练层），则禁用 GradScaler，
    # 否则 GradScaler.unscale_ 会报 "Attempting to unscale FP16 gradients."
    any_fp16_params = False
    for g in optimizer.param_groups:
        for p in g["params"]:
            if isinstance(p, torch.Tensor) and p.dtype == torch.float16:
                any_fp16_params = True
                break
        if any_fp16_params:
            break
    use_scaler = (torch.cuda.is_available() and args.precision=='fp16' and not any_fp16_params)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    if not scaler.is_enabled():
        print("AMP GradScaler disabled: optimizer contains FP16 parameters (using autocast only).")
    
    # ★ 正确根据 --precision 选择 autocast 的 dtype
    def get_amp_ctx(precision: str):
        if not torch.cuda.is_available():
            return contextlib.nullcontext
        if precision.lower() in ["bf16", "bfloat16", "bfloat"]:
            return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
        elif precision.lower() in ["fp16", "half", "float16"]:
            return lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            # fp32
            return lambda: contextlib.nullcontext()
    
    amp_ctx = get_amp_ctx(args.precision)
    
    # ★ Sanity check：根据 precision 检查权重 dtype
    num_fp16 = sum(int(p.dtype == torch.float16) for p in model.parameters())
    num_bf16 = sum(int(p.dtype == torch.bfloat16) for p in model.parameters())
    if args.precision == 'fp32' and num_fp16 + num_bf16 > 0:
        raise AssertionError(f"Expected FP32 params, but found FP16({num_fp16})/BF16({num_bf16}).")
    elif num_fp16 > 0 and args.precision != 'fp16':
        print(f"[Warn] Found {num_fp16} FP16 params while precision={args.precision}. "
              f"Consider loading with torch_dtype={args.precision}.")
    
    # Training loop
    print(f"Starting training, device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Precision: {args.precision} | autocast: {'on' if args.precision!='fp32' and torch.cuda.is_available() else 'off'} | GradScaler: {'on' if scaler.is_enabled() else 'off'}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = args.early_stopping_patience
    history = HistoryLogger(path='checkpoints/training_history.json')
    
    # init global step for prefix warmup
    model._train_step = 0
    # Initialize debug counter for LM supervision monitoring
    model._debug_step = 0
    # Initialize temperature monitoring for ITC
    model._logit_scale_hist = []
    
    for epoch in range(args.num_epochs):
        # Light cleanup per epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_lm_loss = 0.0
        train_itm_loss = 0.0
        train_itc_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
        for batch_idx, batch in enumerate(progress_bar):
            # advance global step first (即使该 batch 后面被跳过)
            model._train_step += 1
            # (Optional) very frequent cleanup会影响性能；必要时再打开
            # if torch.cuda.is_available(): torch.cuda.empty_cache()
            # gc.collect()
            # Move data to device
            image1 = batch['image1'].to(device)
            image2 = batch['image2'].to(device)
            captions = batch['captions'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            prompt_len = batch['prompt_len'].to(device)  # [B]
            
            # 将 prompt_len 暂存到模型实例（避免改 forward 签名）
            model._cached_prompt_len = prompt_len
            
            # Forward pass with AMP
            sup_mask = batch.get('sup_mask', None)
            if sup_mask is not None and not hasattr(model, '_log_sup_mask_present'):
                print(f"[Info] sup_mask present in batch: shape={sup_mask.shape}, sum={sup_mask.sum().item()}")
                model._log_sup_mask_present = True
            
            with amp_ctx():
                loss, lm_l, itm_l, itc_l, _, _ = model(image1, image2, captions, attention_mask, sup_mask, return_loss=True)
            
            # Check for non-finite (NaN/Inf) loss
            if not torch.isfinite(loss):
                print(f"[ERROR] Loss is non-finite at batch {batch_idx}. Skipping this batch.")
                continue
            
            # Keep a copy for logging (no grad)
            raw_loss = loss.detach()
            # Use the graph-bearing loss for backward, normalized for grad accumulation
            loss_to_backward = loss / args.gradient_accumulation_steps
            
            # Backward pass with AMP
            if scaler.is_enabled():
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
                # Light cleanup after step（频率可控）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # (Optional) 每个 batch 都清理会影响性能，建议关闭
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # gc.collect()
            
            # Update loss
            train_loss += raw_loss.item()
            train_lm_loss += lm_l.item()
            train_itm_loss += itm_l.item()
            train_itc_loss += itc_l.item()
            train_steps += 1
            
            # Force delete variables to free memory
            del image1, image2, captions, attention_mask, prompt_len
            
            # Update progress bar - occasionally show loss components for monitoring alignment
            if batch_idx % 10 == 0:  # Show components every 10 steps
                progress_bar.set_postfix({
                    'loss': f'{raw_loss.item():.4f}',
                    'avg_loss': f'{train_loss/(batch_idx+1):.4f}',
                    'LM': f'{lm_l.item():.4f}',
                    'ITM': f'{itm_l.item():.4f}',
                    'ITC': f'{itc_l.item():.4f}'
                })
            else:
                progress_bar.set_postfix({
                    'loss': f'{raw_loss.item():.4f}',
                    'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
                })
            
            # Log ITC temperature every 20 steps for monitoring
            if batch_idx % 20 == 0:
                model._logit_scale_hist.append(float(model.logit_scale.exp().detach().cpu()))
        
        last_batch_idx = batch_idx
        
        # ---- flush pending grads if last micro-batch didn't hit the accumulation boundary ----
        if ((last_batch_idx + 1) % args.gradient_accumulation_steps) != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Force cleanup after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Learning rate scheduling
        scheduler.step()
        
        denom = max(1, train_steps)
        avg_train_loss = train_loss / denom
        avg_train_lm_loss = train_lm_loss / denom
        avg_train_itm_loss = train_itm_loss / denom
        avg_train_itc_loss = train_itc_loss / denom
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        print(f'  - LM Loss: {avg_train_lm_loss:.4f}, ITM Loss: {avg_train_itm_loss:.4f}, ITC Loss: {avg_train_itc_loss:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]')
            for batch_idx, batch in enumerate(val_progress_bar):
                image1 = batch['image1'].to(device)
                image2 = batch['image2'].to(device)
                captions = batch['captions'].to(device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
                prompt_len = batch['prompt_len'].to(device)
                model._cached_prompt_len = prompt_len
                
                with amp_ctx():
                    loss, lm_l, itm_l, itc_l, _, _ = model(image1, image2, captions, attention_mask, batch.get('sup_mask', None), return_loss=True)
                if not torch.isfinite(loss):
                    print(f"[VAL] Non-finite loss at batch {batch_idx}. Skipping this batch.")
                    continue
                val_loss += loss.item(); val_steps += 1
                del image1, image2, captions, attention_mask, prompt_len
                
                # Update validation progress bar
                val_progress_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'avg_val_loss': f'{val_loss/(batch_idx+1):.4f}'
                })
        
        avg_val_loss = val_loss / max(1, val_steps)
        print(f'Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}')

        # ====== Evaluation on validation set (generation metrics) ======
        do_eval = ((epoch + 1) % args.eval_every == 0)
        metrics = {}
        if do_eval:
            print("Running validation generation for metrics...")
            # Use a copy of val_loader but with batch_size=1 to simplify generation loop
            val_eval_loader = DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=0)
            metrics = evaluate_on_loader(
                model, val_eval_loader,
                max_samples=args.eval_samples,
                use_cider=args.eval_cider,
                use_spice=args.eval_spice
            )
            msg = (f"Val Metrics @Epoch {epoch+1}: "
                   f"BLEU-1={metrics['bleu1']:.4f}, BLEU-4={metrics['bleu4']:.4f}")
            
            # ★ 只显示启用的指标，避免 N/A 混淆
            if args.eval_cider and metrics.get('cider') is not None:
                msg += f", CIDEr-D={metrics['cider']:.4f}"
            if args.eval_spice and metrics.get('spice') is not None:
                msg += f", SPICE={metrics['spice']:.4f}"
            
            # METEOR 总是显示（nltk 已安装）
            if metrics.get('meteor') is not None:
                msg += f", METEOR={metrics['meteor']:.4f}"
            
            print(msg)

        # log history and plot
        hist_entry = {
            'epoch': epoch+1,
            'train_loss': float(avg_train_loss),
            'train_lm_loss': float(avg_train_lm_loss),
            'train_itm_loss': float(avg_train_itm_loss),
            'train_itc_loss': float(avg_train_itc_loss),
            'val_loss': float(avg_val_loss),
            'lr': float(scheduler.get_last_lr()[0]),
            # True temperature for logging/plotting (T = 1/exp(logit_scale))
            'itc_temperature': float(torch.exp(-model.logit_scale.detach()).cpu()),
        }
        # 合并本轮指标（只包含启用的指标）
        for k, v in metrics.items():
            if k in ['bleu1', 'bleu4', 'meteor'] or \
               (k == 'cider' and args.eval_cider and v is not None) or \
               (k == 'spice' and args.eval_spice and v is not None):
                hist_entry[k] = float(v) if isinstance(v, (int, float)) else v
        history.append(**hist_entry)  # 追加字典
        if args.save_plots:
            plot_curves(
                history,
                out_dir=args.plots_dir,
                show_cider=args.eval_cider,
                show_spice=args.eval_spice
            )

        # Dynamic ITC weight scheduling (prevent overfitting to alignment)
        if epoch < 4:
            model.aln_weight_itc = 1.5
        elif epoch < 8:
            model.aln_weight_itc = 1.0
        else:
            model.aln_weight_itc = 0.7
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"✓ Validation loss improved to {best_val_loss:.4f}. Saving best model.")
            checkpoint_path = f'checkpoints/tblip2.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f'Best checkpoint saved: {checkpoint_path}')
        else:
            patience_counter += 1
            print(f"✗ Validation loss did not improve. Patience: {patience_counter}/{args.early_stopping_patience}")
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered. Validation loss did not improve for {args.early_stopping_patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        # Save checkpoint every 5 epochs (in addition to best model)
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoints/enhanced_diff_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Regular checkpoint saved: {checkpoint_path}')
    
    # Final plotting to ensure all curves are generated
    if args.save_plots:
        print("[plot] final dump of curves...")
        plot_curves(
            history,
            out_dir=args.plots_dir,
            show_cider=args.eval_cider,
            show_spice=args.eval_spice
        )
    
    print('Training completed!')

def load_best_model(model, checkpoint_path='checkpoints/best_enhanced_diff_model.pth'):
    """
    Load the best model from checkpoint
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New checkpoint format with additional info
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            bvl = checkpoint.get('best_val_loss', None)
            if isinstance(bvl, (float, int)):
                print(f"Best validation loss: {bvl:.4f}")
            else:
                print("Best validation loss: unknown")
        else:
            # Old checkpoint format
            model.load_state_dict(checkpoint)
            print("Loaded model (old format)")
        
        return True
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
        return False

def test_model(model, test_loader, args):
    """
    Test function
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best model if available
    load_best_model(model)
    
    # If using HF device_map dispatch, skip full model migration to avoid breaking device_map
    if not (hasattr(model, "blip2_model") and hasattr(model.blip2_model, "hf_device_map")):
        model = model.to(device)
    else:
        print("Detected HF device_map dispatch; skip model.to(device).")
    
    model.eval()
    import contextlib
    no_grad_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    print("Starting testing...")
    with no_grad_ctx():
        for i, batch in enumerate(tqdm(test_loader)):
            if args.max_test_samples is not None and i >= args.max_test_samples:
                break
                
            image1 = batch['image1'].to(device)
            image2 = batch['image2'].to(device)
            original_caption = batch['original_caption']
            
            # No labels needed before generation, but set default 0 for consistency
            model._cached_prompt_len = torch.zeros(1, dtype=torch.long, device=device)
            
            # Generate descriptions
            descriptions, cluster_masks, cluster_info = model.generate(
                image1.unsqueeze(0), 
                image2.unsqueeze(0)
            )
            # Visualisation
            # save_visual_debug(image1[0], image2[0], model.compute_difference_map(image1, image2)[0], cluster_masks[0], f"plots/debug_sample_{i+1}.png")
            
            print(f"\nSample {i+1}:")
            print(f"  Original Caption: {original_caption}")
            print(f"  Generated Descriptions:")
            for j, description in enumerate(descriptions):
                print(f"    {description}")
            
            if i >= 5:  # Only show first 5 samples
                break

def evaluate_on_loader(model, loader, max_samples=None, use_cider=True, use_spice=False):
    """
    Run generation on a loader and compute BLEU-1/BLEU-4/METEOR/(CIDEr/SPICE).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    refs, hyps = [], []
    n_done = 0
    no_grad_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with no_grad_ctx():
        for batch in loader:
            bs = batch['image1'].size(0)
            for j in range(bs):
                img1 = batch['image1'][j].to(device)
                img2 = batch['image2'][j].to(device)
                oc = batch['original_caption']
                ref = oc[j] if isinstance(oc, (list, tuple)) else oc
                model._cached_prompt_len = torch.zeros(1, dtype=torch.long, device=device)
                hyp_list, _, _ = model.generate(img1.unsqueeze(0), img2.unsqueeze(0))
                hyp = hyp_list[0]
                refs.append(ref)
                hyps.append(hyp)
                n_done += 1
                if max_samples is not None and n_done >= max_samples:
                    break
            if max_samples is not None and n_done >= max_samples:
                break
    bleu1, bleu4 = compute_bleu_scores(refs, hyps)
    meteor = compute_meteor(refs, hyps)
    extra = compute_cider_spice(
        refs, hyps,
        use_cider=use_cider,
        use_spice=use_spice
    )
    out = {'bleu1': bleu1, 'bleu4': bleu4, 'meteor': meteor, 'n_samples': n_done}
    out.update(extra)
    return out

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced Difference Detection Model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--data_dir', type=str, default='data/diff/spot-the-diff-harsh19', help='Data directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--max_test_samples', type=int, default=None, help='Maximum number of test samples')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_clusters', type=int, default=5, help='Maximum number of clusters')
    parser.add_argument('--lr_scheduler_step_size', type=int, default=5, help='Learning rate scheduler step size')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.7, help='Learning rate scheduler gamma')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate metrics every N epochs')
    parser.add_argument('--eval_samples', type=int, default=200, help='Max samples for metric evaluation (None=all, default: 200)')
    parser.add_argument('--save_plots', action='store_true', default=True, help='Save loss/metric plots during training (default: True)')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--text_max_length', type=int, default=64,
                        help='Tokenizer max_length for training captions (+prompt). Default: 64')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
# --- evaluation flags ---
# 需要 Python 3.9+ 的 argparse.BooleanOptionalAction
    parser.add_argument(
        "--eval_cider",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute CIDEr-D (default: True). Use --no-eval_cider to disable."
    )
    parser.add_argument(
        "--eval_spice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute SPICE (default: True). Use --no-eval_spice to disable."
    )
    parser.add_argument('--enable_flash_sdp', action='store_true', default=False,
                         help='Enable Flash/Memory-efficient SDP attention (may be unstable on some GPUs)')
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp16','bf16','fp32'],
                         help='Computation precision for autocast (default: bf16)')
    parser.add_argument('--prefix_scale', type=float, default=1.0,
                        help='Scale factor applied to visual prefix to stabilize LM (default: 1.0)')
    parser.add_argument('--prefix_warmup_steps', type=int, default=300,
                        help='Warmup steps for visual prefix scale (linearly 0→prefix_scale, default: 300)')
    # —— Alignment loss weights (exposed to CLI) —
    parser.add_argument('--aln_weight_itm', type=float, default=1.5,
                        help='Weight for ITM alignment loss (default: 1.5)')
    parser.add_argument('--aln_weight_itc', type=float, default=2.0,
                        help='Weight for ITC alignment loss (default: 2.0)')
    # ITC temperature hyper-params
    parser.add_argument('--itc_temp_init', type=float, default=0.07,
                        help='Initial temperature τ for ITC (default 0.07; effective logit scale = ln(1/τ))')
    parser.add_argument('--itc_temp_reg', type=float, default=1e-4,
                        help='L2 regularization towards ln(1/τ0) for logit_scale')
    
    # DifferenceDetector parameters
    # Tuning priority: 1) raise percentile (98.5→99.0) if still "full-screen", 2) increase min_samples (20→25) if too noisy
    # Only loosen eps (3.5→4.0) if missing valid clusters
    parser.add_argument('--diff_threshold_percentile', type=float, default=98.5, help='Percentile for difference threshold (default: 98.5)')
    parser.add_argument('--diff_threshold_multiplier', type=float, default=1.0, help='Multiplier for difference threshold (default: 1.0)')
    parser.add_argument('--dbscan_eps', type=float, default=3.5, help='DBSCAN eps parameter (default: 3.5)')
    parser.add_argument('--dbscan_min_samples', type=int, default=20, help='DBSCAN min_samples parameter (default: 20)')
    
    args = parser.parse_args()
    
    # Seed everything for reproducibility
    set_seed(args.seed)
    # Create BLIP2 model
    print("Loading BLIP2 model...")
    
    # ★ Select weight loading dtype based on --precision
    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': None,            # None = use weight default (mostly fp32)
    }
    load_dtype = dtype_map[args.precision]
    
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-opt-2.7b',
        device_map="auto",
        torch_dtype=load_dtype,
        low_cpu_mem_usage=True
    )
    
    # Create enhanced model
    print("Creating enhanced difference detection model...")
    print(f"  - Max clusters: {args.max_clusters}")
    print(f"  - Difference threshold: {args.diff_threshold_percentile}th percentile × {args.diff_threshold_multiplier}")
    print(f"  - DBSCAN: eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}")
    model = EnhancedDiffBLIP2(
        blip2_model, 
        max_clusters=args.max_clusters,
        diff_threshold_percentile=args.diff_threshold_percentile,
        diff_threshold_multiplier=args.diff_threshold_multiplier,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples
    )
    # Store in model for shared use in forward/generate
    model.prefix_scale = float(getattr(args, 'prefix_scale', 0.25))
    model.prefix_warmup_steps = int(getattr(args, 'prefix_warmup_steps', 200))
    # Apply alignment loss weights (override class defaults 0.5)
    model.aln_weight_itm = float(args.aln_weight_itm)
    model.aln_weight_itc = float(args.aln_weight_itc)
    print(f"  - Alignment weights: ITM={model.aln_weight_itm}, ITC={model.aln_weight_itc}")
    
    # Inject ITC temperature hyper-params into model for init/regularisation
    model.itc_temp_init = float(getattr(args, 'itc_temp_init', 0.07))
    model.itc_temp_reg  = float(getattr(args, 'itc_temp_reg', 1e-4))
    # Re-init logit_scale/target based on CLI in case the model was constructed earlier
    with torch.no_grad():
        val = math.log(1.0 / max(1e-6, model.itc_temp_init))
        model.logit_scale.data.fill_(val)  # 0-dim 参数用 fill_ 即可
        model.itc_temp_target = torch.tensor(
            val, dtype=model.logit_scale.dtype, device=model.logit_scale.device
        )
    
    # Test model structure
    print("Testing model structure...")
    try:
        dummy_image1 = torch.randn(1, 3, 224, 224)
        dummy_image2 = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            fused_feats, cluster_info, cluster_masks = model.forward(dummy_image1, dummy_image2, return_loss=False)
            print(f"✓ Forward pass successful")
            print(f"  - fused_feats shape: {fused_feats.shape}")
            print(f"  - cluster_info length: {len(cluster_info)}")
            print(f"  - cluster_masks shape: {cluster_masks.shape}")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    if args.mode == 'train':
        # Load training data
        print("Loading training dataset...")
        train_dataset = EnhancedDiffDataset(args.data_dir, split='train',
                                            max_samples=args.max_samples,
                                            max_length=args.text_max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        # Load validation data
        print("Loading validation dataset...")
        val_dataset = EnhancedDiffDataset(args.data_dir, split='val',
                                          max_samples=args.max_samples,
                                          max_length=args.text_max_length)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Print evaluation settings and recommendations
        print(f"\nEvaluation settings:")
        print(f"  - Eval every: {args.eval_every} epoch(s)")
        print(f"  - Eval samples: {args.eval_samples if args.eval_samples else 'all'}")
        print(f"  - CIDEr-D: {'enabled' if args.eval_cider else 'disabled'}")
        print(f"  - SPICE: {'enabled' if args.eval_spice else 'disabled'}")
        if args.eval_every == 1 and args.eval_samples is None:
            print(f"  ⚠️  Note: Full validation set evaluation every epoch may be slow.")
            print(f"     Consider: --eval_samples 200 or --eval_every 2 for faster training")
        if args.eval_spice:
            print(f"  ⚠️  Note: SPICE evaluation is slow. Consider disabling if training speed is priority")
        
        # Train model
        train_model(model, train_loader, val_loader, args)
        
    elif args.mode == 'test':
        # Load test data
        print("Loading test dataset...")
        test_dataset = EnhancedDiffDataset(args.data_dir, split='test',
                                           max_samples=args.max_test_samples,
                                           max_length=args.text_max_length)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Test model
        test_model(model, test_loader, args)

if __name__ == '__main__':
    main()
    
# =========================
# 运行示例与说明
# =========================
"""
Quick Start Examples:

1. Basic Training (Recommended Configuration):
   python enhanced_diff_tblip2.py --mode train --data_dir data/diff/spot-the-diff-harsh19 \
     --batch_size 2 --num_epochs 10 --save_plots --eval_every 1 --eval_samples 200

2. Fast Training (Reduced Evaluation Overhead):
   python enhanced_diff_tblip2.py --mode train --data_dir data/diff/spot-the-diff-harsh19 \
     --batch_size 2 --num_epochs 10 --eval_every 2 --eval_samples 100

3. Full Evaluation (Including SPICE, Slower):
   python enhanced_diff_tblip2.py --mode train --data_dir data/diff/spot-the-diff-harsh19 \
     --batch_size 2 --num_epochs 10 --save_plots --eval_every 1 --eval_samples None --eval_spice

4. Test Mode:
   python enhanced_diff_tblip2.py --mode test --data_dir data/diff/spot-the-diff-harsh19

5. Quick Fix Parameters (Immediate Relief):
   python enhanced_diff_tblip2.py --mode train \
     --data_dir data/diff/spot-the-diff-harsh19 \
     --num_epochs 8 --batch_size 32 --learning_rate 5e-5 \
     --weight_decay 0.001 --precision bf16 \
     --prefix_scale 1.0 --prefix_warmup_steps 300 \
     --aln_weight_itm 1.5 --aln_weight_itc 2.0 \
     --diff_threshold_percentile 99.5 --diff_threshold_multiplier 1.0 \
     --dbscan_eps 4.0 --dbscan_min_samples 20

Key Features:
✅ Multi-cluster visual prefix (consistent training/inference)
✅ Prompt masking during training, normal generation during inference
✅ Geometric position encoding + difference detection
✅ ITM/ITC alignment losses
✅ Single forward pass to get hidden states
✅ BLEU-1/4, METEOR, CIDEr-D, SPICE evaluation
✅ Training history logging and visualisation
✅ Device/precision automatic alignment
✅ Final batch gradient accumulation flush

Training Monitoring:
- Display loss components (LM/ITM/ITC) every 10 steps
- Automatically save training history to checkpoints/training_history.json
- Generate loss curve plots to plots/loss_curve.png
- Generate loss component plots to plots/loss_components.png
- Generate evaluation metric plots to plots/metrics_curve.png
- Generate ITC temperature evolution plot to plots/itc_temperature_curve.png
- Warning if batch_size=1 (ITM/ITC will be skipped)

Important Notes:
- Default --eval_samples 200 avoids slow full validation set evaluation
- Default --eval_cider True, --eval_spice True (both enabled by default)
- Use --no-eval_cider or --no-eval_spice to disable specific metrics
- Using --eval_every 2 can further reduce evaluation frequency
""" 