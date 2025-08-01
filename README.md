# contrastive-gnn-nids
Network Intrusion Detection using Graph Neural Networks with Contrastive Learning
This project was a hands-on experience on IBM Cloud, built during my AI/ML internship at Edunet Foundation in collaboration with IBM SkillsBuild. The system detects malicious network traffic by leveraging graph neural networks and contrastive pretraining to enhance representation learning.
# 🛡️ Network Intrusion Detection System Using Graph Neural Networks

A sophisticated Network Intrusion Detection System (NIDS) that leverages Graph Neural Networks (GNNs) to identify and classify cyber-attacks in network traffic. This project transforms network connection data into graph structures and uses contrastive learning for robust anomaly detection.

## 🌟 Features

- **Graph-based Network Modeling**: Converts network connections into meaningful graph structures
- **Contrastive Pre-training**: Self-supervised learning approach for better feature representation
- **High Performance**: Achieves 99.12% AUC and 96.07% F1-score
- **Data Augmentation**: Advanced graph augmentation techniques for improved generalization
- **Chronological Evaluation**: Realistic temporal split to prevent data leakage

## 📊 Results

Our model demonstrates exceptional performance on the network intrusion detection task:

- **AUC Score**: 99.12%
- **F1 Score**: 96.07%  
- **Accuracy**: 96.08%
- **Precision**: 96.15%
- **Recall**: 96.08%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| Normal | 0.95 | 0.98 | 0.96 | 5,397 |
| Anomaly | 0.98 | 0.94 | 0.96 | 4,680 |

## 🏗️ Architecture

The system consists of three main components:

1. **Graph Construction**: Transforms network connections into graph structures with nodes representing sources, destinations, protocols, and services
2. **Graph Encoder**: GIN-based (Graph Isomorphism Network) encoder with attention pooling
3. **Classification Head**: Multi-layer classifier for binary anomaly detection

### Graph Structure
- **Nodes**: Source, Destination, Protocol, Service (4 nodes per connection)
- **Edges**: Bidirectional connections with temporal and statistical features
- **Features**: 10-dimensional node features capturing network statistics

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ankitsencode123/contrastive-gnn-nids.git
cd contrastive-gnn-nids
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - The model expects data in CSV format only

### Usage

Run the complete training pipeline:

```bash
jupyter notebook ml-gnn.ipynb
```

Or execute the Python script directly if you extract the code from the notebook.

## 📁 Project Structure

```
├── images/                     # Project screenshots and visualizations
│   ├── IBM_CLOUD_6.png
│   ├── IBM_CLOUD_7.png
│   ├── P11.png - P14.png
│   ├── P2.png - P3.png
│   └── SHELL_1.png
├── presentation/               # Project documentation
│   ├── PROJECT_IBM-EDUNET.pdf
│   └── PROJECT_IBM-EDUNET.pptx
├── ml-gnn.ipynb               # Main Jupyter notebook
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## 🔧 Technical Details

### Dataset
- **Source**: Network intrusion detection dataset simulating US Air Force LAN
- **Features**: 41 quantitative and qualitative features per connection
- **Size**: ~125,000 network connections
- **Classes**: Normal vs. Anomalous traffic

### Model Architecture
- **Encoder**: 3-layer GIN with batch normalization and residual connections
- **Hidden Dimensions**: 128 → 64 (projection head)
- **Parameters**: 120,467 total (117,825 encoder + 2,642 classifier)

### Training Strategy
1. **Contrastive Pre-training** (50 epochs): Self-supervised learning on unlabeled data
2. **Fine-tuning** (30 epochs): Supervised training with frozen encoder layers
3. **Evaluation**: Chronological split to simulate real-world deployment

### Data Augmentation
- **Structural**: Node dropout, edge dropout
- **Attribute**: Feature masking, temporal noise injection

## 🔬 Key Innovations

1. **Graph Representation**: Novel approach to represent network connections as graphs
2. **Contrastive Learning**: Self-supervised pre-training for better feature extraction
3. **Attention Pooling**: Weighted aggregation of node features for graph-level prediction
4. **Temporal Awareness**: Incorporates timing information in graph structures

## 📈 Performance Analysis

The model excels at detecting various types of network intrusions:
- **High Precision**: 98% for anomaly detection, minimizing false positives
- **Balanced Performance**: Strong performance on both normal and anomalous traffic
- **Scalability**: Efficient architecture suitable for real-time deployment

## 🛠️ Future Enhancements

- [ ] Multi-class classification for specific attack types
- [ ] Real-time inference optimization
- [ ] Integration with network monitoring tools
- [ ] Federated learning for distributed networks
- [ ] Explainable AI for attack attribution

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **NetworkX**: Graph manipulation
- **Scikit-learn**: Machine learning utilities
- **Pandas/NumPy**: Data processing
- **tqdm**: Progress bars

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by the network security research community
- Inspired by recent advances in graph neural networks for cybersecurity
- Built upon PyTorch Geometric framework

## 📞 Contact

For questions or collaboration opportunities, please reach out through GitHub issues or contact the maintainers.

---

**Note**: This project is part of an educational initiative and demonstrates the application of modern machine learning techniques to cybersecurity challenges.
