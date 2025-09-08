# T-BLIP2: Enhanced Difference Detection with Multi-Modal AI

## Overview

T-BLIP2 is an advanced difference detection system that combines state-of-the-art computer vision clustering with BLIP2 language model capabilities to generate high-quality, context-aware descriptions of visual differences between image pairs. The system features a comprehensive training pipeline, interactive GUI, and robust evaluation metrics.

## 🚀 Key Features

- **Multi-Cluster Difference Detection**: Advanced DBSCAN clustering with adaptive thresholding
- **Enhanced BLIP2 Integration**: Optimized visual prefix integration with learnable temperature
- **Comprehensive Training Pipeline**: Validation support, early stopping, and automatic checkpointing
- **Interactive GUI**: Modern tkinter-based interface for easy image comparison
- **Multi-Metric Evaluation**: BLEU, METEOR, CIDEr-D, and SPICE scoring
- **Memory Optimized**: Gradient checkpointing, mixed precision, and efficient data loading
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📁 Project Structure

```
T-BLIP2/
├── tblip2.py      # Main training and inference script
├── tblip2_demo.py        # Demo script for quick testing
├── spot_the_diff.py    # DDLA training and inference script
├── spot_the_diff_demo.py            # Basic difference detection demo
├── datasets.py                  # Dataset utilities and loading
├── data/                        # Dataset directory
│   ├── diff/                    # Spot-the-diff dataset
│   ├── relation/                # NLVR dataset
│   ├── single/                  # COCO dataset
│   └── story/                   # Story dataset
├── checkpoints/                 # Model checkpoints and training history
├── plots/                       # Training curves and visualizations
└── requirements.txt             # Python dependencies
```

## 🏗️ Architecture

### Core Components

#### 1. DifferenceDetector
- **Adaptive Thresholding**: Otsu method with percentile fallback
- **Morphological Operations**: Noise reduction and hole filling
- **DBSCAN Clustering**: Robust region detection with configurable parameters
- **Feature Encoding**: CNN-based cluster feature extraction

#### 2. EnhancedDiffBLIP2
- **Visual Feature Fusion**: Global + difference + cluster features
- **Geometric Encoding**: Position, size, and area information
- **Multi-Modal Integration**: Seamless BLIP2 language model integration
- **Alignment Losses**: ITM (Image-Text Matching) and ITC (Image-Text Contrastive)

#### 3. Training Pipeline
- **Multi-Loss Optimization**: LM loss + ITM loss + ITC loss
- **Dynamic Weight Scheduling**: Adaptive loss weight adjustment
- **Early Stopping**: Automatic training termination with patience
- **Checkpoint Management**: Best model saving and recovery

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd T-BLIP2

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Training

```bash
# Basic training with recommended settings
python enhanced_diff_tblip2.py \
    --mode train \
    --data_dir data/diff/spot-the-diff-harsh19 \
    --batch_size 2 \
    --num_epochs 20 \
    --learning_rate 3e-5 \
    --save_plots \
    --eval_every 1 \
    --eval_samples 200
```

#### 2. Testing

```bash
# Test trained model
python enhanced_diff_tblip2.py \
    --mode test \
    --data_dir data/diff/spot-the-diff-harsh19 \
    --max_test_samples 10
```

#### 3. Interactive GUI

```bash
# Launch interactive interface
python interactive_diff_generator.py --gui

# Use specific model
python interactive_diff_generator.py --gui \
    --model_path checkpoints/best_enhanced_diff_model.pth
```

#### 4. Demo

```bash
# Run enhanced demo
python enhanced_diff_demo.py

# Run basic demo
python spot_diff_demo.py
```

## ⚙️ Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `train` | Operation mode: train/test |
| `--data_dir` | `data/diff/spot-the-diff-harsh19` | Dataset directory |
| `--batch_size` | `2` | Training batch size |
| `--num_epochs` | `20` | Number of training epochs |
| `--learning_rate` | `3e-5` | Learning rate |
| `--max_clusters` | `5` | Maximum number of clusters |
| `--early_stopping_patience` | `5` | Early stopping patience |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--diff_threshold_percentile` | `98.5` | Difference threshold percentile |
| `--dbscan_eps` | `3.5` | DBSCAN epsilon parameter |
| `--dbscan_min_samples` | `20` | DBSCAN minimum samples |
| `--precision` | `bf16` | Computation precision (fp16/bf16/fp32) |

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval_every` | `1` | Evaluate every N epochs |
| `--eval_samples` | `200` | Max samples for evaluation |
| `--eval_cider` | `True` | Enable CIDEr-D evaluation |
| `--eval_spice` | `True` | Enable SPICE evaluation |

## 📊 Training Features

### 1. Multi-Loss Training
- **Language Model Loss**: Cross-entropy for text generation
- **ITM Loss**: Image-text matching with hard negatives
- **ITC Loss**: Contrastive learning with learnable temperature
- **Dynamic Weighting**: Adaptive loss weight scheduling

### 2. Advanced Optimization
- **Gradient Accumulation**: Effective large batch training
- **Mixed Precision**: FP16/BF16 training for memory efficiency
- **Gradient Checkpointing**: Memory optimization for large models
- **Separate Learning Rates**: Higher LR for alignment heads

### 3. Training Monitoring
- **Real-time Metrics**: Loss components displayed every 10 steps
- **Automatic Plotting**: Loss curves, metrics, and temperature evolution
- **History Logging**: Complete training history in JSON format
- **Early Stopping**: Automatic training termination

## 🔍 Evaluation Metrics

### Automatic Metrics
- **BLEU-1/4**: N-gram overlap scoring
- **METEOR**: Semantic similarity with WordNet
- **CIDEr-D**: Consensus-based evaluation
- **SPICE**: Semantic propositional content

### Custom Metrics
- **Cluster Quality**: Size, density, and coverage analysis
- **Generation Diversity**: Output variety and consistency
- **Alignment Score**: Visual-text alignment quality

## 💾 Model Management

### Checkpoint System
- **Best Model**: Automatically saved when validation improves
- **Regular Checkpoints**: Saved every 5 epochs
- **Training State**: Complete optimizer and scheduler states
- **Automatic Recovery**: Resume training from interruptions

### Model Loading
```python
# Automatic best model detection
model = EnhancedDiffBLIP2(blip2_model)
load_best_model(model)  # Loads from checkpoints/best_enhanced_diff_model.pth

# Manual checkpoint loading
checkpoint = torch.load('checkpoints/model_epoch_10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🎨 Interactive Interface

### GUI Features
- **Modern Design**: Clean, intuitive interface
- **Drag & Drop**: Easy image pair selection
- **Real-time Processing**: Background generation with progress
- **Visual Results**: Side-by-side comparison with descriptions
- **Batch Processing**: Multiple image pair support

### Command Line Interface
```bash
# Process single image pair
python interactive_diff_generator.py \
    --model_path checkpoints/best_model.pth \
    --image1 path/to/image1.jpg \
    --image2 path/to/image2.jpg

# Batch processing
python interactive_diff_generator.py \
    --model_path checkpoints/best_model.pth \
    --input_dir path/to/image/pairs \
    --output_file results.json
```

## 📈 Performance Optimization

### Memory Management
```python
# Enable gradient checkpointing
model.blip2_model.language_model.gradient_checkpointing_enable()

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss = model(image1, image2, captions)
```

### Training Efficiency
```python
# Separate learning rates for different components
optimizer = torch.optim.AdamW([
    {'params': head_params, 'lr': args.learning_rate * 10.0},
    {'params': lm_params, 'lr': args.learning_rate},
])

# Dynamic weight scheduling
if epoch < 4:
    model.aln_weight_itc = 1.5
elif epoch < 8:
    model.aln_weight_itc = 1.0
else:
    model.aln_weight_itc = 0.7
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Memory Errors
```bash
# Reduce batch size
--batch_size 1

# Enable gradient accumulation
--gradient_accumulation_steps 4

# Use mixed precision
--precision bf16
```

#### 2. Training Instability
```bash
# Reduce learning rate
--learning_rate 1e-5

# Increase patience
--early_stopping_patience 10

# Adjust loss weights
--aln_weight_itm 1.0 --aln_weight_itc 1.5
```

#### 3. Poor Clustering Results
```bash
# Adjust threshold
--diff_threshold_percentile 99.0

# Modify DBSCAN parameters
--dbscan_eps 4.0 --dbscan_min_samples 25

# Increase cluster limit
--max_clusters 8
```

## 🔧 Advanced Usage

### Custom Dataset Integration
```python
class CustomDiffDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.samples = self.load_data()
    
    def load_data(self):
        # Implement custom data loading logic
        pass
```

### Model Customization
```python
# Custom difference detector
class CustomDifferenceDetector(nn.Module):
    def __init__(self, custom_params):
        super().__init__()
        # Implement custom detection logic
        pass

# Custom fusion layer
class CustomFusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
```

### Evaluation Pipeline
```python
def custom_evaluation(model, test_loader):
    """Custom evaluation function"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Custom evaluation logic
            pass
    
    return results
```

## 📚 API Reference

### EnhancedDiffBLIP2 Class

#### Methods
- `forward(img1, img2, captions=None, return_loss=False)`: Main forward pass
- `generate(img1, img2, prompt="Diff: ", max_new_tokens=50)`: Generate descriptions
- `encode_images(img1, img2)`: Encode image pairs
- `compute_difference_map(img1, img2)`: Compute difference maps

#### Attributes
- `max_clusters`: Maximum number of clusters
- `diff_detector`: Difference detection module
- `blip2_model`: BLIP2 language model
- `fusion_layer`: Feature fusion layer

### DifferenceDetector Class

#### Methods
- `forward(diff_map)`: Process difference maps
- `compute_clusters(binary_map)`: Extract clusters

#### Parameters
- `cluster_embed_dim`: Cluster feature dimension
- `max_clusters`: Maximum cluster count
- `eps`: DBSCAN epsilon
- `min_samples`: DBSCAN minimum samples

## 🎯 Use Cases

### 1. Quality Control
- **Manufacturing**: Detect defects in product images
- **Medical Imaging**: Identify changes in medical scans
- **Document Analysis**: Find differences in forms or contracts

### 2. Content Creation
- **Before/After**: Generate descriptions for transformation images
- **Product Updates**: Document changes in product versions
- **Progress Tracking**: Monitor development over time

### 3. Research Applications
- **Scientific Imaging**: Analyze experimental results
- **Environmental Monitoring**: Track landscape changes
- **Biological Studies**: Monitor organism development

## 🤝 Contributing

### Development Setup
```bash
# Clone and setup
git clone <repository-url>
cd T-BLIP2
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black *.py
isort *.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add comprehensive docstrings
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **BLIP2**: Salesforce's BLIP2 model for vision-language understanding
- **DBSCAN**: Density-based clustering algorithm
- **Hugging Face**: Transformers library and model hosting
- **PyTorch**: Deep learning framework

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation
- Contact the maintainers

---

**T-BLIP2**: Advanced difference detection powered by multi-modal AI. Transform image comparison into meaningful insights with state-of-the-art computer vision and natural language generation. 