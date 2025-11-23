<<<<<<< HEAD
It on processing
=======
# Research Paper Recommendation System

A hybrid recommendation system that combines fine-tuned SciBERT embeddings with Graph Neural Networks (GNN) to provide accurate research paper recommendations based on natural language queries.

## Overview

This project implements a two-stage recommendation system:
1. **BERT Fine-tuning**: Fine-tunes SciBERT (allenai/scibert_scivocab_uncased) using LoRA for efficient parameter-efficient fine-tuning
2. **GNN Refinement**: Uses GraphSAGE to refine embeddings by leveraging graph structure built from paper categories and author relationships

The system processes research papers from arXiv and builds a knowledge graph connecting papers through:
- Shared categories
- Shared authors
- KNN similarity edges based on BERT embeddings

## Features

- **Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation) for parameter-efficient BERT fine-tuning
- **Graph-based Learning**: Leverages Graph Neural Networks to capture structural relationships between papers
- **Hybrid Approach**: Combines semantic embeddings (BERT) with graph structure (GNN) for better recommendations
- **Natural Language Queries**: Accepts free-form text queries to find relevant research papers
- **Automatic Data Handling**: Falls back to synthetic data if real arXiv data is unavailable

## Architecture

### Model Components

1. **BertProjector** (`model/bert_projector.py`):
   - Wraps SciBERT model
   - Projects embeddings to GNN dimension (128)
   - Includes classification head for category prediction

2. **GNN** (`model/gnn.py`):
   - 2-layer GraphSAGE architecture
   - Batch normalization and dropout for regularization
   - Outputs refined embeddings for recommendation

### Training Pipeline

1. **Data Loading**: Loads arXiv metadata (or generates synthetic data)
2. **Graph Construction**: Builds base graph from categories and authors
3. **BERT Fine-tuning**: Fine-tunes SciBERT with LoRA on paper classification
4. **Embedding Projection**: Projects BERT embeddings to GNN dimension
5. **KNN Augmentation**: Adds KNN edges to graph based on embedding similarity
6. **GNN Training**: Trains GraphSAGE on augmented graph
7. **Recommendation**: Uses refined embeddings for cosine similarity search

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Recommend_System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Dependencies

- `torch>=2.0.0` - PyTorch framework
- `torch-geometric` - Graph neural network library
- `transformers` - Hugging Face transformers
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `bitsandbytes` - Quantization support
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `networkx` - Graph construction
- `tqdm` - Progress bars
- `huggingface-hub` - Model downloading

## Data Preparation

### Option 1: Using Real arXiv Data

1. Download arXiv metadata from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) or directly from arXiv
2. Place the file at `data/arxiv-metadata-oai-snapshot.json`
3. The system will automatically load and process the data

### Option 2: Synthetic Data (Default)

If real data is not available, the system automatically generates synthetic data for testing purposes.

## Usage

### Training the Models

Run the fine-tuning script to train both BERT and GNN models:

```bash
python fintuner.py
```

This will:
- Load/prepare the data
- Build the knowledge graph
- Fine-tune SciBERT with LoRA
- Train the GNN model
- Save checkpoints in the `models/` directory

### Making Recommendations

After training, use the recommendation script:

```bash
python recommend.py
```

Enter your research query when prompted, and the system will return top-k relevant papers.

### Example Query

```
Enter your research query: advancements in graph neural networks for computer vision
```

The system will return:
- Paper titles
- Abstracts (truncated)
- Similarity scores

## Configuration

Edit `config.py` to customize training parameters:

```python
SEED = 8                          # Random seed
MODEL_NAME = "allenai/scibert_scivocab_uncased"  # BERT model
MAX_SAMPLES = 12000              # Maximum number of papers
BERT_BATCH_SIZE = 64             # BERT training batch size
EPOCHS_BERT = 43                 # BERT training epochs
EPOCHS_GNN = 250                 # GNN training epochs
BERT_LR = 5e-5                   # BERT learning rate
GNN_LR = 5e-4                    # GNN learning rate
MAX_LENGTH = 256                 # Maximum sequence length
GNN_DIM = 128                    # GNN embedding dimension
KNN_K = 6                        # KNN edges to add
CONT_WEIGHT = 0.0                # Contrastive loss weight
```

## Project Structure

```
Recommend_System/
├── config.py                    # Configuration parameters
├── data_loader.py              # Data loading and preprocessing
├── fintuner.py                 # Main training script
├── recommend.py                # Recommendation interface
├── requirements.txt            # Python dependencies
├── model/
│   ├── bert_projector.py      # BERT model with projection
│   └── gnn.py                  # Graph Neural Network
├── utils/
│   ├── graph_builder.py        # Graph construction utilities
│   └── knn_augment.py          # KNN edge augmentation
├── models/                     # Saved model checkpoints
│   ├── bert_final.pt
│   ├── gnn_final.pt
│   └── ...
└── notebooks version (google colab)/
    └── recommendSystem_new.ipynb  # Colab notebook version
```

## Model Details

### BERT Fine-tuning

- **Base Model**: SciBERT (allenai/scibert_scivocab_uncased)
- **Fine-tuning Method**: LoRA (r=8, alpha=16)
- **Loss**: Cross-entropy + optional contrastive loss
- **Optimizer**: AdamW with learning rate scheduling
- **Regularization**: Label smoothing, gradient clipping

### GNN Architecture

- **Type**: GraphSAGE (2 layers)
- **Hidden Dimension**: 128
- **Activation**: ReLU
- **Regularization**: Batch normalization, dropout (0.3)
- **Loss**: Label smoothing cross-entropy

### Graph Construction

1. **Base Graph**: Built from:
   - Papers sharing the same category → edges
   - Papers sharing the same author → edges

2. **KNN Augmentation**: Adds top-K most similar papers based on BERT embeddings

## Performance

The system tracks:
- BERT validation accuracy and F1 score
- GNN test accuracy
- Training losses for both models

Checkpoints are saved periodically during training.

## Notes

- The system uses mixed precision training (AMP) for efficiency
- GPU is recommended but not required (CPU fallback available)
- Model checkpoints are saved in the `models/` directory
- The system automatically handles missing data by generating synthetic examples

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

>>>>>>> 6a274d6 (Readme)
