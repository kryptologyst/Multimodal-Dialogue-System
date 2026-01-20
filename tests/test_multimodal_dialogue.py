"""
Test suite for the multimodal dialogue system.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from src.multimodal_dialogue import MultimodalDialogueSystem, DialogueResponse
from src.config import ConfigManager, ModelConfig, SystemConfig, AppConfig
from src.data_utils import SyntheticDataGenerator, DataLoader


class TestMultimodalDialogueSystem:
    """Test cases for MultimodalDialogueSystem."""
    
    @pytest.fixture
    def mock_models(self):
        """Mock the transformer models for testing."""
        with patch('src.multimodal_dialogue.AutoTokenizer') as mock_tokenizer, \
             patch('src.multimodal_dialogue.AutoModelForCausalLM') as mock_text_model, \
             patch('src.multimodal_dialogue.BlipProcessor') as mock_processor, \
             patch('src.multimodal_dialogue.BlipForConditionalGeneration') as mock_vision_model:
            
            # Setup mock tokenizer
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = "<|endoftext|>"
            mock_tokenizer.return_value.eos_token_id = 50256
            mock_tokenizer.return_value.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.return_value.decode.return_value = "Mock response"
            
            # Setup mock text model
            mock_text_model.return_value.generate.return_value = [[1, 2, 3, 4, 5]]
            
            # Setup mock vision processor and model
            mock_processor.return_value.return_tensors = {"input_ids": [[1, 2, 3]], "pixel_values": [[[[1, 2], [3, 4]]]]}
            mock_vision_model.return_value.generate.return_value = [[1, 2, 3, 4, 5]]
            mock_processor.return_value.decode.return_value = "Mock image caption"
            
            yield {
                'tokenizer': mock_tokenizer,
                'text_model': mock_text_model,
                'processor': mock_processor,
                'vision_model': mock_vision_model
            }
    
    def test_system_initialization(self, mock_models):
        """Test system initialization."""
        system = MultimodalDialogueSystem()
        
        assert system.text_model_name == "microsoft/DialoGPT-medium"
        assert system.vision_model_name == "Salesforce/blip-image-captioning-base"
        assert system.max_length == 100
        assert system.temperature == 0.7
        assert system.do_sample is True
    
    def test_text_response_generation(self, mock_models):
        """Test text response generation."""
        system = MultimodalDialogueSystem()
        
        response = system.generate_text_response("Hello, how are you?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        mock_models['tokenizer'].return_value.encode.assert_called()
        mock_models['text_model'].return_value.generate.assert_called()
    
    def test_image_caption_generation(self, mock_models):
        """Test image caption generation."""
        system = MultimodalDialogueSystem()
        
        # Create a temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp_file.name)
            
            caption = system.generate_image_caption(tmp_file.name)
            
            assert isinstance(caption, str)
            assert len(caption) > 0
            
            # Clean up
            Path(tmp_file.name).unlink()
    
    def test_multimodal_response(self, mock_models):
        """Test multimodal response generation."""
        system = MultimodalDialogueSystem()
        
        # Test with image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            img = Image.new('RGB', (100, 100), color='blue')
            img.save(tmp_file.name)
            
            response = system.generate_response("What do you see?", tmp_file.name)
            
            assert isinstance(response, DialogueResponse)
            assert response.text_response is not None
            assert response.image_caption is not None
            assert response.confidence_score is not None
            assert response.metadata is not None
            
            # Clean up
            Path(tmp_file.name).unlink()
    
    def test_text_only_response(self, mock_models):
        """Test text-only response generation."""
        system = MultimodalDialogueSystem()
        
        response = system.generate_response("Hello, world!")
        
        assert isinstance(response, DialogueResponse)
        assert response.text_response is not None
        assert response.image_caption is None
        assert response.confidence_score is not None
    
    def test_batch_processing(self, mock_models):
        """Test batch processing functionality."""
        system = MultimodalDialogueSystem()
        
        inputs = [
            {"user_input": "Hello!"},
            {"user_input": "How are you?"},
            {"user_input": "What's the weather like?"}
        ]
        
        responses = system.batch_process(inputs)
        
        assert len(responses) == 3
        assert all(isinstance(r, DialogueResponse) for r in responses)
    
    def test_confidence_calculation(self, mock_models):
        """Test confidence score calculation."""
        system = MultimodalDialogueSystem()
        
        # Test with good response
        confidence = system._calculate_confidence(
            "This is a detailed response with multiple words and proper grammar.",
            "A detailed image caption with multiple descriptive words"
        )
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be high for good inputs
        
        # Test with poor response
        confidence = system._calculate_confidence("Hi", None)
        assert 0 <= confidence <= 1
        assert confidence < 0.8  # Should be lower for poor inputs


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config_manager = ConfigManager()
        
        assert isinstance(config_manager.model_config, ModelConfig)
        assert isinstance(config_manager.system_config, SystemConfig)
        assert isinstance(config_manager.app_config, AppConfig)
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test_config.yaml"
            config_manager = ConfigManager(config_path)
            
            # Modify config
            config_manager.model_config.temperature = 0.9
            config_manager.system_config.log_level = "DEBUG"
            
            # Save config
            config_manager.save_config()
            assert config_path.exists()
            
            # Load config
            new_config_manager = ConfigManager(config_path)
            new_config_manager.load_config()
            
            assert new_config_manager.model_config.temperature == 0.9
            assert new_config_manager.system_config.log_level == "DEBUG"
    
    def test_config_dict(self):
        """Test configuration dictionary conversion."""
        config_manager = ConfigManager()
        config_dict = config_manager.get_config_dict()
        
        assert 'model' in config_dict
        assert 'system' in config_dict
        assert 'app' in config_dict
        assert isinstance(config_dict['model'], dict)
        assert isinstance(config_dict['system'], dict)
        assert isinstance(config_dict['app'], dict)


class TestSyntheticDataGenerator:
    """Test cases for SyntheticDataGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SyntheticDataGenerator(Path(tmp_dir))
            assert generator.output_dir == Path(tmp_dir)
    
    def test_sample_image_generation(self):
        """Test sample image generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SyntheticDataGenerator(Path(tmp_dir))
            image_paths = generator.generate_sample_images(3)
            
            assert len(image_paths) == 3
            assert all(path.exists() for path in image_paths)
            assert all(path.suffix == '.png' for path in image_paths)
    
    def test_sample_dialogue_generation(self):
        """Test sample dialogue generation."""
        generator = SyntheticDataGenerator()
        dialogues = generator.generate_sample_dialogues(5)
        
        assert len(dialogues) == 5
        assert all('id' in d for d in dialogues)
        assert all('user_input' in d for d in dialogues)
        assert all('expected_response' in d for d in dialogues)
        assert all('category' in d for d in dialogues)
        assert all('difficulty' in d for d in dialogues)
    
    def test_complete_dataset_generation(self):
        """Test complete dataset generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SyntheticDataGenerator(Path(tmp_dir))
            dataset_info = generator.generate_complete_dataset(2, 3)
            
            assert dataset_info['num_images'] == 2
            assert dataset_info['num_dialogues'] == 3
            assert 'data_file' in dataset_info
            assert Path(dataset_info['data_file']).exists()


class TestDataLoader:
    """Test cases for DataLoader."""
    
    def test_data_validation(self):
        """Test data validation functionality."""
        loader = DataLoader()
        
        # Valid data
        valid_dialogues = [
            {
                'id': 'test_1',
                'user_input': 'Hello',
                'expected_response': 'Hi there!',
                'image_path': 'test.jpg',
                'category': 'greeting',
                'difficulty': 'easy'
            }
        ]
        
        validation_results = loader.validate_data(valid_dialogues)
        assert validation_results['valid'] is True
        assert len(validation_results['errors']) == 0
    
    def test_data_validation_errors(self):
        """Test data validation with errors."""
        loader = DataLoader()
        
        # Invalid data (missing required fields)
        invalid_dialogues = [
            {
                'id': 'test_1',
                # Missing 'user_input' and 'expected_response'
                'image_path': 'test.jpg'
            }
        ]
        
        validation_results = loader.validate_data(invalid_dialogues)
        assert validation_results['valid'] is False
        assert len(validation_results['errors']) > 0


class TestDialogueResponse:
    """Test cases for DialogueResponse dataclass."""
    
    def test_response_creation(self):
        """Test DialogueResponse creation."""
        response = DialogueResponse(
            text_response="Hello world",
            image_caption="A test image",
            confidence_score=0.85,
            metadata={"test": True}
        )
        
        assert response.text_response == "Hello world"
        assert response.image_caption == "A test image"
        assert response.confidence_score == 0.85
        assert response.metadata["test"] is True
    
    def test_response_defaults(self):
        """Test DialogueResponse with default values."""
        response = DialogueResponse(text_response="Hello")
        
        assert response.text_response == "Hello"
        assert response.image_caption is None
        assert response.confidence_score is None
        assert response.metadata is None


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate sample data
            generator = SyntheticDataGenerator(Path(tmp_dir))
            dataset_info = generator.generate_complete_dataset(1, 2)
            
            # Load data
            loader = DataLoader(Path(tmp_dir))
            dialogues = loader.load_dialogues()
            
            # Validate data
            validation_results = loader.validate_data(dialogues)
            assert validation_results['valid'] is True
            
            # Test with mock models (to avoid downloading real models in tests)
            with patch('src.multimodal_dialogue.AutoTokenizer'), \
                 patch('src.multimodal_dialogue.AutoModelForCausalLM'), \
                 patch('src.multimodal_dialogue.BlipProcessor'), \
                 patch('src.multimodal_dialogue.BlipForConditionalGeneration'):
                
                system = MultimodalDialogueSystem()
                
                # Process dialogues
                for dialogue in dialogues[:1]:  # Test with first dialogue only
                    image_path = loader.get_image_path(dialogue['image_path'])
                    response = system.generate_response(
                        dialogue['user_input'], 
                        image_path
                    )
                    
                    assert isinstance(response, DialogueResponse)
                    assert response.text_response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
