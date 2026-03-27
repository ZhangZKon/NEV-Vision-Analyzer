
# 🚗 NEV-Vision-Analyzer

A multimodal monitoring system for new-energy vehicle (NEV) companies, combining face recognition and BERT-based text analysis to detect and track leader-related public opinion in news and social media.

## 🎯 Key Features
- Multimodal pipeline: image + text
- Leader-level face detection/recognition
- BERT-based text understanding and classification
- Config-driven training and inference workflow
- Easy extension to new companies, leaders, and datasets

## 🧠 Use Cases
- Brand and executive reputation monitoring
- News and social-media risk screening
- Leader-centric event tracking
- PR and crisis early warning support

## 🛠️ Models & APIs Used
### Large Language Models (LLMs)
- **DeepSeek API**: 
  - Model: `deepseek-chat`
  - Connection: OpenAI-compatible API interface.
- **Qwen API (Alibaba)**: 
  - Model: `qwen-plus`
  - Connection: Alibaba DashScope HTTP API.

### Vision & Multimodal Models
- **Image Captioning**: `Salesforce/blip2-opt-2.7b` (Optimized for CPU inference).
- **Text Extraction (OCR)**: `EasyOCR` (Supports English & Simplified Chinese).
- **Sentiment Analysis**: Custom-trained BERT model (`AutoModelForSequenceClassification`).
- **Face Recognition**: `face_recognition` (dlib) for matching known EV founders/executives.

## 📂 Project Structure
```text
NEV-Vision-Analyzer/
├── main.py                  # Agent core logic & CLI entry point
├── config.py                # System settings & API Key configurations
├── requirements.txt         # Project dependencies
├── utils/
│   └── known_faces/         # Directory for EV leaders' face images (e.g., Elon Musk.jpg)
├── models/
│   └── sentiment_model_bert/# Custom trained BERT model directory
└── README.md                # Project documentation
```
## Environment
- Python 3.9+ (recommended)
- PyTorch / TensorFlow (depending on implementation)
- Transformers (BERT)
- OpenCV (face/image pipeline)

## Installation
```bash
git clone https://github.com/ZhangZKon/NEV-Vision-Analyzer.git
cd NEV-Vision-Analyzer
pip install -r requirements.txt
```

## Configuration
Edit `config.py` for:
- dataset paths
- model checkpoints
- training hyperparameters
- output/log directories

## Data Preparation
- Prepare text data (news/posts/comments) with labels.
- Prepare image data for target leaders.
- Build train/validation/test splits.
- Update paths in `config.py`.

## Evaluation
Report typical metrics such as:
- Precision
- Recall
- F1-score
- Accuracy
- AUC (if applicable)


## Limitations
- Performance depends on data quality and annotation consistency.
- Face recognition may degrade in low-quality or crowded images.
- Model bias and domain shift should be evaluated before production use.
