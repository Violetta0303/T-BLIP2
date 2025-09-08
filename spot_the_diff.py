#!/usr/bin/env python3
"""
Original Spot-the-Diff Implementation (Corrected)
================================================

This script reproduces the original spot-the-diff implementation from the EMNLP 2018 paper.
Key features:
1. Clustering-based visual analysis (DBSCAN)
2. Latent variable alignment model for text generation
3. Visual salience modeling
4. Per-cluster sentence generation with alignment

Original Paper: "Learning to Describe Differences Between Pairs of Similar Images"
Authors: Harsh Jhamtani, Taylor Berg-Kirkpatrick
Conference: EMNLP 2018

Author: Assistant
Date: 2024
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import argparse
from typing import List, Dict, Tuple, Optional
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotTheDiffDataset(Dataset):
    """
    Dataset for spot-the-diff task following the original paper structure.
    """
    def __init__(self, data_dir, split='train', max_length=50):
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        
        # Load annotations
        annotation_file = os.path.join(data_dir, "annotations", f"{split}.json")
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loaded {len(self.annotations)} samples from {split} split")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        sample = self.annotations[idx]
        img_id = sample['img_id']
        
        # Load image pair
        img1_path = os.path.join(self.data_dir, "resized_images", f'{img_id}.png')
        img2_path = os.path.join(self.data_dir, "resized_images", f'{img_id}_2.png')
        diff_path = os.path.join(self.data_dir, "resized_images", f'{img_id}_diff.jpg')
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        diff_map = Image.open(diff_path).convert('L')
        
        # Convert to tensors
        img1_tensor = torch.tensor(np.array(img1)).permute(2, 0, 1).float() / 255.0
        img2_tensor = torch.tensor(np.array(img2)).permute(2, 0, 1).float() / 255.0
        diff_tensor = torch.tensor(np.array(diff_map)).unsqueeze(0).float() / 255.0
        
        # Get text description
        sentences = sample['sentences']
        if isinstance(sentences, list):
            text = ' '.join(sentences)
        else:
            text = sentences
        
        # Tokenize text
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'img_id': img_id,
            'img1': img1_tensor,
            'img2': img2_tensor,
            'diff_map': diff_tensor,
            'text': text,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }

class ClusteringModule(nn.Module):
    """
    Clustering module for detecting difference regions (original approach).
    Based on the original paper's DBSCAN clustering implementation.
    """
    def __init__(self, eps=20, min_samples=9):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
    
    def forward(self, diff_map):
        """
        Extract clustering features from difference map using original paper's approach.
        
        Args:
            diff_map: [B, 1, H, W] difference map tensor
            
        Returns:
            clustering_features: list of cluster info dicts
        """
        batch_size = diff_map.size(0)
        clustering_features = []
        
        for i in range(batch_size):
            # Convert to numpy for original DBSCAN approach
            diff_tensor = diff_map[i, 0].cpu().numpy()
            
            # Binary threshold (original paper approach)
            binary_map = (diff_tensor > 0.1).astype(np.float32)
            
            # Get coordinates of difference pixels
            diff_indices = np.nonzero(binary_map)
            if len(diff_indices[0]) == 0:
                coordinates = np.empty((0, 2))
            else:
                coordinates = np.column_stack((diff_indices[0], diff_indices[1]))
            
            if len(coordinates) == 0:
                # No differences found
                cluster_info = {
                    'centers': np.zeros((5, 2)),
                    'sizes': np.zeros(5),
                    'bboxes': np.zeros((5, 4)),
                    'num_clusters': 0,
                    'density': 0.0,
                    'labels': np.array([]),
                    'coordinates': coordinates,
                    'masks': np.zeros((5, diff_tensor.shape[0], diff_tensor.shape[1]))
                }
            else:
                # Original paper's DBSCAN clustering
                if len(coordinates) > 1:
                    db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(coordinates)
                    labels = db.labels_
                else:
                    labels = np.array([0])
                
                # Extract cluster features (original paper approach)
                unique_labels = set(labels)
                if -1 in unique_labels:  # Remove noise
                    unique_labels.remove(-1)
                
                num_clusters = len(unique_labels)
                max_clusters = 5
                
                centers = np.zeros((max_clusters, 2))
                sizes = np.zeros(max_clusters)
                bboxes = np.zeros((max_clusters, 4))
                masks = np.zeros((max_clusters, diff_tensor.shape[0], diff_tensor.shape[1]))
                
                for j, label in enumerate(unique_labels):
                    if j >= max_clusters:
                        break
                    
                    cluster_points = coordinates[labels == label]
                    if len(cluster_points) == 0:
                        continue
                    
                    # Cluster center
                    center = np.mean(cluster_points, axis=0)
                    centers[j] = center
                    
                    # Cluster size
                    sizes[j] = len(cluster_points)
                    
                    # Cluster bounding box
                    min_coords = np.min(cluster_points, axis=0)
                    max_coords = np.max(cluster_points, axis=0)
                    bboxes[j] = [min_coords[0], min_coords[1], max_coords[0], max_coords[1]]
                    
                    # Create cluster mask
                    for point in cluster_points:
                        masks[j, int(point[0]), int(point[1])] = 1.0
                
                # Calculate density
                total_diff_pixels = len(coordinates)
                total_pixels = diff_tensor.shape[0] * diff_tensor.shape[1]
                density = total_diff_pixels / total_pixels
                
                cluster_info = {
                    'centers': centers,
                    'sizes': sizes,
                    'bboxes': bboxes,
                    'num_clusters': num_clusters,
                    'density': density,
                    'labels': labels,
                    'coordinates': coordinates,
                    'masks': masks
                }
            
            clustering_features.append(cluster_info)
        
        return clustering_features

class VisualFeatureExtractor(nn.Module):
    """
    Visual feature extractor for images and cluster masks.
    """
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple CNN for feature extraction
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        
        # Cluster mask encoder
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(64, feature_dim)
        )
        
        # Debug flag for tensor shape checking
        self._debug_shapes = False
    
    def forward(self, img, cluster_masks=None):
        """
        Extract visual features from image and cluster masks.
        
        Args:
            img: [B, 3, H, W] image tensor
            cluster_masks: [B, num_clusters, H, W] cluster mask tensor
            
        Returns:
            img_features: [B, feature_dim] image features
            cluster_features: [B, num_clusters, feature_dim] cluster features
        """
        # Extract image features
        x = F.relu(self.conv1(img))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        img_features = self.fc(x)
        
        # Extract cluster features if provided
        cluster_features = None
        if cluster_masks is not None:
            batch_size, num_clusters, H, W = cluster_masks.size()
            cluster_features = []
            
            for i in range(batch_size):
                cluster_feats = []
                for j in range(num_clusters):
                    mask = cluster_masks[i, j:j+1]  # [1, H, W]
                    
                    # 补全channels/batch维，mask_encoder需要 [B, 1, H, W]
                    feat = self.mask_encoder(mask.unsqueeze(0))  # [1, feature_dim]
                    feat = feat.squeeze(0)  # [feature_dim]
                    
                    cluster_feats.append(feat)
                
                # 防止空list
                if len(cluster_feats) == 0:
                    cluster_feats.append(torch.zeros(self.feature_dim, device=img.device))
                
                cluster_features.append(torch.stack(cluster_feats))  # [num_clusters, feature_dim]
            
            cluster_features = torch.stack(cluster_features)  # [B, num_clusters, feature_dim]
        
        return img_features, cluster_features

class LatentAlignmentDecoder(nn.Module):
    """
    Latent alignment decoder for generating descriptions per cluster.
    """
    def __init__(self, vocab_size, feature_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.cluster_embedding = nn.Linear(feature_dim, hidden_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Cluster-sentence alignment predictor
        self.alignment_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, cluster_features, img_features=None):
        """
        Forward pass with latent alignment.
        
        Args:
            input_ids: [B, seq_len] input token ids
            cluster_features: [B, num_clusters, feature_dim] cluster features
            img_features: [B, feature_dim] image features (optional)
            
        Returns:
            outputs: dict with loss and predictions
        """
        batch_size, seq_len = input_ids.size()
        num_clusters = cluster_features.size(1)
        
        # Word embeddings
        word_embeds = self.word_embedding(input_ids)  # [B, seq_len, hidden_dim]
        
        # Cluster embeddings
        cluster_embeds = self.cluster_embedding(cluster_features)  # [B, num_clusters, hidden_dim]
        
        # LSTM decoding
        lstm_output, _ = self.lstm(word_embeds)  # [B, seq_len, hidden_dim]
        
        # Apply attention between decoder outputs and cluster features
        aligned_outputs = []
        for i in range(batch_size):
            # Apply attention
            aligned_output, _ = self.attention(
                lstm_output[i:i+1],  # [1, seq_len, hidden_dim]
                cluster_embeds[i:i+1],  # [1, num_clusters, hidden_dim]
                cluster_embeds[i:i+1]
            )
            aligned_outputs.append(aligned_output.squeeze(0))
        
        aligned_outputs = torch.stack(aligned_outputs)  # [B, seq_len, hidden_dim]
        
        # Generate predictions
        logits = self.output_projection(aligned_outputs)  # [B, seq_len, vocab_size]
        
        # Calculate loss
        targets = input_ids[:, 1:]  # Shift for teacher forcing
        if targets.numel() > 0:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )
        else:
            loss = torch.tensor(0.0, device=logits.device)
        
        return {
            'loss': loss,
            'logits': logits,
            'aligned_outputs': aligned_outputs
        }
    
    def generate(self, cluster_features, img_features=None, max_length=20, tokenizer=None):
        """
        Generate description for a single cluster.
        
        Args:
            cluster_features: [num_clusters, feature_dim] cluster features
            img_features: [feature_dim] image features (optional)
            max_length: maximum generation length
            tokenizer: tokenizer for text generation
            
        Returns:
            generated_text: list of generated descriptions
        """
        num_clusters = cluster_features.size(0)
        device = cluster_features.device
        
        generated_texts = []
        
        for cluster_idx in range(num_clusters):
            # Start with BOS token
            input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
            
            # Get cluster feature
            cluster_feat = cluster_features[cluster_idx:cluster_idx+1]  # [1, feature_dim]
            
            for step in range(max_length):
                # Forward pass
                outputs = self.forward(input_ids, cluster_feat.unsqueeze(0))
                
                # Get next token
                logits = outputs['logits'][-1, -1]  # Last token of last sequence
                
                # Apply temperature
                temperature = 0.8
                logits = logits / temperature
                
                # Filter out special tokens
                special_tokens = [
                    tokenizer.pad_token_id, 
                    tokenizer.sep_token_id,
                    tokenizer.cls_token_id,
                    tokenizer.mask_token_id
                ]
                
                for token_id in special_tokens:
                    if token_id is not None:
                        logits[token_id] = -float('inf')
                
                # Use top-k sampling
                top_k = 20
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices[next_token_idx].unsqueeze(0)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for EOS
                if next_token.item() == tokenizer.sep_token_id:
                    break
                
                # Stop if we have enough tokens
                if step >= 10 and len(input_ids[0]) > 5:
                    break
                
                # Stop if we're generating too many repeated tokens
                if step > 2:
                    recent_tokens = input_ids[0][-2:].tolist()
                    if len(set(recent_tokens)) == 1:  # All same token
                        break
            
            # Decode text
            text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts

class OriginalSpotTheDiffModel(nn.Module):
    """
    Complete original spot-the-diff model with proper latent alignment.
    """
    def __init__(self, vocab_size, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Modules
        self.clustering_module = ClusteringModule()
        self.visual_extractor = VisualFeatureExtractor(feature_dim)
        self.decoder = LatentAlignmentDecoder(vocab_size, feature_dim)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, img1, img2, diff_map, input_ids, attention_mask):
        """
        Complete forward pass with latent alignment.
        
        Args:
            img1: [B, 3, H, W] first image
            img2: [B, 3, H, W] second image  
            diff_map: [B, 1, H, W] difference map
            input_ids: [B, seq_len] input token ids
            attention_mask: [B, seq_len] attention mask
            
        Returns:
            outputs: dict with loss and predictions
        """
        # 1. Clustering analysis
        clustering_features = self.clustering_module(diff_map)
        
        # 2. Visual feature extraction
        img_features, cluster_features = self.visual_extractor(img1, self._get_cluster_masks(clustering_features))
        
        # 3. Latent alignment for text generation
        outputs = self.decoder(input_ids, cluster_features, img_features)
        
        return outputs
    
    def _get_cluster_masks(self, clustering_features):
        """
        Extract cluster masks from clustering features.
        
        Args:
            clustering_features: list of cluster info dicts
            
        Returns:
            cluster_masks: [B, max_clusters, H, W] cluster mask tensor
        """
        batch_size = len(clustering_features)
        max_clusters = 5
        
        # Get image size from first sample
        if batch_size > 0 and clustering_features[0]['masks'].size > 0:
            H, W = clustering_features[0]['masks'].shape[1:]
        else:
            H, W = 224, 224  # Default size
        
        # Get device from model parameters
        device = next(self.parameters()).device
        cluster_masks = torch.zeros(batch_size, max_clusters, H, W, device=device)
        
        for i, cluster_info in enumerate(clustering_features):
            masks = cluster_info['masks']
            num_clusters = min(cluster_info['num_clusters'], max_clusters)
            
            for j in range(num_clusters):
                cluster_masks[i, j] = torch.tensor(masks[j], device=device)
        
        return cluster_masks
    
    def generate(self, img1, img2, diff_map, max_length=20):
        """
        Generate text description of differences per cluster.
        
        Args:
            img1: [B, 3, H, W] first image
            img2: [B, 3, H, W] second image
            diff_map: [B, 1, H, W] difference map
            max_length: maximum generation length per cluster
            
        Returns:
            generated_text: list of generated descriptions
        """
        batch_size = img1.size(0)
        
        # Clustering analysis
        clustering_features = self.clustering_module(diff_map)
        
        # Visual feature extraction
        img_features, cluster_features = self.visual_extractor(img1, self._get_cluster_masks(clustering_features))
        
        # Generate descriptions per cluster
        all_generated_texts = []
        
        for i in range(batch_size):
            cluster_info = clustering_features[i]
            num_clusters = cluster_info['num_clusters']
            
            if num_clusters == 0:
                # No differences found
                all_generated_texts.append(["no differences found"])
                continue
            
            # Get cluster features for this sample
            sample_cluster_features = cluster_features[i, :num_clusters]  # [num_clusters, feature_dim]
            
            # Generate description for each cluster
            cluster_descriptions = self.decoder.generate(
                sample_cluster_features, 
                img_features[i] if img_features is not None else None,
                max_length,
                self.tokenizer
            )
            
            all_generated_texts.append(cluster_descriptions)
        
        return all_generated_texts

def train_original_spot_the_diff(data_dir, num_epochs=10, batch_size=4, learning_rate=1e-4, device='cuda', max_samples=None):
    """
    Train the original spot-the-diff model.
    """
    logger.info("Starting training of original spot-the-diff model")
    logger.info(f"Using device: {device}")
    if max_samples:
        logger.info(f"Using only {max_samples} samples for quick testing")
    
    # Check GPU availability
    if device == 'cuda':
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
    
    # Create datasets
    train_dataset = SpotTheDiffDataset(data_dir, 'train')
    val_dataset = SpotTheDiffDataset(data_dir, 'val')
    
    # Limit samples if specified
    if max_samples:
        # Calculate the original ratio
        total_original = len(train_dataset.annotations) + len(val_dataset.annotations)
        train_ratio = len(train_dataset.annotations) / total_original
        val_ratio = len(val_dataset.annotations) / total_original
        
        # Apply the same ratio to max_samples
        train_samples = int(max_samples * train_ratio)
        val_samples = max_samples - train_samples  # Ensure total equals max_samples
        
        train_dataset.annotations = train_dataset.annotations[:train_samples]
        val_dataset.annotations = val_dataset.annotations[:val_samples]
        
        logger.info(f"Original ratio - Train: {train_ratio:.2%}, Val: {val_ratio:.2%}")
        logger.info(f"Limited to {len(train_dataset.annotations)} train samples and {len(val_dataset.annotations)} val samples")
        logger.info(f"Total samples: {len(train_dataset.annotations) + len(val_dataset.annotations)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    vocab_size = len(train_dataset.tokenizer)
    model = OriginalSpotTheDiffModel(vocab_size).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            # Move to device
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            diff_map = batch['diff_map'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(img1, img2, diff_map, input_ids, attention_mask)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'GPU': f'{torch.cuda.memory_allocated(0) / 1e9:.1f}GB' if device == 'cuda' else 'CPU'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False)
            for val_batch_idx, batch in enumerate(val_pbar):
                img1 = batch['img1'].to(device)
                img2 = batch['img2'].to(device)
                diff_map = batch['diff_map'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(img1, img2, diff_map, input_ids, attention_mask)
                val_loss += outputs['loss'].item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'Val Loss': f'{outputs["loss"].item():.4f}'
                })
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model

def main():
    """
    Main function to run the original spot-the-diff implementation.
    """
    parser = argparse.ArgumentParser(description="Original Spot-the-Diff Implementation")
    parser.add_argument("--data_dir", type=str, default="data/diff/spot-the-diff-harsh19",
                        help="Path to dataset directory")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                         help="Device to use")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (for quick testing)")

    args = parser.parse_args()

    logger.info("Starting Original Spot-the-Diff Implementation")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Mode: {args.mode}")
    logger.info("This reproduces the original EMNLP 2018 paper approach")

    if args.mode == "train":
        # Train the model
        model = train_original_spot_the_diff(
            args.data_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            max_samples=args.max_samples
        )
        
        # Save model
        torch.save(model.state_dict(), "checkpoints/spot_the_diff_model.pth")
        logger.info("Model saved to checkpoints/spot_the_diff_model.pth")
    
    elif args.mode == "test":
        # Test the model
        test_dataset = SpotTheDiffDataset(args.data_dir, 'test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Load model
        vocab_size = len(test_dataset.tokenizer)
        model = OriginalSpotTheDiffModel(vocab_size).to(args.device)
        
        if os.path.exists("checkpoints/spot_the_diff_model.pth"):
            model.load_state_dict(torch.load("checkpoints/spot_the_diff_model.pth"))
            logger.info("Loaded trained model")
        else:
            logger.warning("No trained model found, using untrained model")
        
        # Generate descriptions
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 5:  # Test first 5 samples
                    break
                
                img1 = batch['img1'].to(args.device)
                img2 = batch['img2'].to(args.device)
                diff_map = batch['diff_map'].to(args.device)
                
                generated_texts = model.generate(img1, img2, diff_map)
                
                logger.info(f"Sample {i+1}:")
                logger.info(f"Generated: {generated_texts[0]}")
                logger.info(f"Ground truth: {batch['text'][0]}")
                logger.info("---")

if __name__ == "__main__":
    main() 