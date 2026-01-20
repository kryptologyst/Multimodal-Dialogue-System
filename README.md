# Multimodal Dialogue System

State-of-the-art multimodal dialogue system that combines text and image inputs to generate contextual responses using transformer-based models.

## Features

- **Multimodal Processing**: Combines text and image inputs for richer conversations
- **State-of-the-art Models**: Uses Hugging Face transformers (DialoGPT, BLIP) for text generation and image captioning
- **Multiple Interfaces**: Web UI (Streamlit), CLI, and Python API
- **Configurable**: YAML/JSON configuration management
- **Synthetic Data**: Built-in data generation for testing and demonstration
- **Modern Architecture**: Type hints, logging, error handling, and clean code structure
- **Batch Processing**: Process multiple inputs efficiently
- **Analytics**: Confidence scoring and conversation analytics

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Streamlit (for web interface)
- PIL/Pillow
- PyYAML
- NumPy

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kryptologyst/Multimodal-Dialogue-System.git
   cd Multimodal-Dialogue-System
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Web Interface (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501`

### Command Line Interface

**Interactive Chat Mode:**
```bash
python src/cli.py chat
```

**Demo Mode (with sample data):**
```bash
python src/cli.py demo
```

**Batch Processing:**
```bash
python src/cli.py batch data/sample_dialogues.json -o results.json
```

### Python API

```python
from src.multimodal_dialogue import MultimodalDialogueSystem

# Initialize the system
system = MultimodalDialogueSystem()

# Generate response with text only
response = system.generate_response("Hello, how are you?")
print(response.text_response)

# Generate response with text and image
response = system.generate_response(
    "What do you see in this image?", 
    image_path="path/to/image.jpg"
)
print(f"Response: {response.text_response}")
print(f"Image Caption: {response.image_caption}")
print(f"Confidence: {response.confidence_score}")
```

## 📁 Project Structure

```
multimodal-dialogue-system/
├── src/                    # Core source code
│   ├── __init__.py
│   ├── multimodal_dialogue.py  # Main dialogue system
│   ├── config.py              # Configuration management
│   ├── data_utils.py          # Data utilities and synthetic data
│   └── cli.py                 # Command-line interface
├── web_app/               # Streamlit web interface
│   └── app.py
├── data/                  # Data directory (auto-created)
├── models/                # Model cache directory
├── config/                # Configuration files
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Configuration

The system uses YAML configuration files. A default configuration is created automatically at `config/config.yaml`:

```yaml
model:
  text_model: "microsoft/DialoGPT-medium"
  vision_model: "Salesforce/blip-image-captioning-base"
  device: null  # Auto-detect
  max_length: 100
  temperature: 0.7
  do_sample: true

system:
  log_level: "INFO"
  cache_dir: "./cache"
  data_dir: "./data"
  models_dir: "./models"
  max_image_size: 512
  supported_image_formats: [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

app:
  title: "Multimodal Dialogue System"
  description: "A modern multimodal dialogue system combining text and images"
  version: "1.0.0"
  debug: false
  host: "0.0.0.0"
  port: 8501
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Generate and test with sample data:

```bash
python src/cli.py demo
```

## Usage Examples

### Web Interface Features

1. **Chat Tab**: Interactive conversation with optional image uploads
2. **Analytics Tab**: View conversation statistics and confidence scores
3. **Data Tab**: Manage sample datasets and view data validation
4. **Settings Tab**: Configure system parameters

### CLI Examples

**Basic chat:**
```bash
python src/cli.py chat
# Enter: "What do you see in this image?"
# Upload an image when prompted
```

**Batch processing:**
```bash
# Create sample data first
python -c "from src.data_utils import SyntheticDataGenerator; SyntheticDataGenerator().generate_complete_dataset()"

# Process the data
python src/cli.py batch data/sample_dialogues.json
```

### Python API Examples

**Text-only conversation:**
```python
from src.multimodal_dialogue import MultimodalDialogueSystem

system = MultimodalDialogueSystem()
response = system.generate_response("Tell me about artificial intelligence")
print(response.text_response)
```

**Multimodal conversation:**
```python
from PIL import Image

# Load an image
image = Image.open("sample_image.jpg")

# Generate multimodal response
response = system.generate_response(
    "Describe what you see in this image",
    image_path=image
)

print(f"Response: {response.text_response}")
print(f"Caption: {response.image_caption}")
print(f"Confidence: {response.confidence_score:.2f}")
```

**Batch processing:**
```python
inputs = [
    {"user_input": "Hello!", "image_path": None},
    {"user_input": "What's in this image?", "image_path": "image1.jpg"},
    {"user_input": "Describe the scene", "image_path": "image2.jpg"}
]

responses = system.batch_process(inputs)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response.text_response}")
```

## 🔧 Advanced Configuration

### Model Selection

Choose different models for different use cases:

```python
# For faster inference (smaller models)
system = MultimodalDialogueSystem(
    text_model="distilgpt2",
    vision_model="Salesforce/blip-image-captioning-base"
)

# For better quality (larger models)
system = MultimodalDialogueSystem(
    text_model="microsoft/DialoGPT-large",
    vision_model="Salesforce/blip-image-captioning-large"
)
```

### Device Configuration

```python
# Force CPU usage
system = MultimodalDialogueSystem(device="cpu")

# Force GPU usage
system = MultimodalDialogueSystem(device="cuda")

# Auto-detect (default)
system = MultimodalDialogueSystem(device=None)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use smaller models or CPU device
2. **Model download fails**: Check internet connection and Hugging Face access
3. **Image processing errors**: Ensure image files are valid and supported formats

### Performance Tips

1. **Use smaller models** for faster inference
2. **Enable model caching** by setting `cache_dir` in config
3. **Batch process** multiple inputs together
4. **Use GPU** when available for better performance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformer models
- [Microsoft](https://www.microsoft.com/) for DialoGPT
- [Salesforce](https://www.salesforce.com/) for BLIP models
- [Streamlit](https://streamlit.io/) for the web interface framework

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review the example code
# Multimodal-Dialogue-System
