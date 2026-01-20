"""
Test configuration and utilities.
"""

import pytest
from pathlib import Path
import tempfile
import json

from src.config import ConfigManager, ModelConfig, SystemConfig, AppConfig


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig()
    
    assert config.text_model == "microsoft/DialoGPT-medium"
    assert config.vision_model == "Salesforce/blip-image-captioning-base"
    assert config.device is None
    assert config.max_length == 100
    assert config.temperature == 0.7
    assert config.do_sample is True


def test_system_config_defaults():
    """Test SystemConfig default values."""
    config = SystemConfig()
    
    assert config.log_level == "INFO"
    assert config.cache_dir == "./cache"
    assert config.data_dir == "./data"
    assert config.models_dir == "./models"
    assert config.max_image_size == 512
    assert config.supported_image_formats == (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def test_app_config_defaults():
    """Test AppConfig default values."""
    config = AppConfig()
    
    assert config.title == "Multimodal Dialogue System"
    assert config.description == "A modern multimodal dialogue system combining text and images"
    assert config.version == "1.0.0"
    assert config.debug is False
    assert config.host == "0.0.0.0"
    assert config.port == 8501


def test_config_manager_initialization(temp_config_dir):
    """Test ConfigManager initialization."""
    config_path = temp_config_dir / "config.yaml"
    config_manager = ConfigManager(config_path)
    
    assert config_manager.config_path == config_path
    assert isinstance(config_manager.model_config, ModelConfig)
    assert isinstance(config_manager.system_config, SystemConfig)
    assert isinstance(config_manager.app_config, AppConfig)


def test_config_save_yaml(temp_config_dir):
    """Test saving configuration as YAML."""
    config_path = temp_config_dir / "config.yaml"
    config_manager = ConfigManager(config_path)
    
    # Modify some values
    config_manager.model_config.temperature = 0.9
    config_manager.system_config.log_level = "DEBUG"
    
    # Save config
    config_manager.save_config()
    
    # Check file exists
    assert config_path.exists()
    
    # Check file content
    with open(config_path, 'r') as f:
        content = f.read()
        assert "temperature: 0.9" in content
        assert "log_level: DEBUG" in content


def test_config_save_json(temp_config_dir):
    """Test saving configuration as JSON."""
    config_path = temp_config_dir / "config.json"
    config_manager = ConfigManager(config_path)
    
    # Modify some values
    config_manager.app_config.debug = True
    config_manager.app_config.port = 8080
    
    # Save config
    config_manager.save_config()
    
    # Check file exists
    assert config_path.exists()
    
    # Load and verify
    with open(config_path, 'r') as f:
        data = json.load(f)
        assert data['app']['debug'] is True
        assert data['app']['port'] == 8080


def test_config_load_yaml(temp_config_dir):
    """Test loading configuration from YAML."""
    config_path = temp_config_dir / "config.yaml"
    
    # Create test config file
    test_config = {
        'model': {
            'temperature': 0.8,
            'max_length': 150
        },
        'system': {
            'log_level': 'WARNING'
        },
        'app': {
            'debug': True,
            'port': 9000
        }
    }
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    # Load config
    config_manager = ConfigManager(config_path)
    config_manager.load_config()
    
    # Verify loaded values
    assert config_manager.model_config.temperature == 0.8
    assert config_manager.model_config.max_length == 150
    assert config_manager.system_config.log_level == "WARNING"
    assert config_manager.app_config.debug is True
    assert config_manager.app_config.port == 9000


def test_config_load_json(temp_config_dir):
    """Test loading configuration from JSON."""
    config_path = temp_config_dir / "config.json"
    
    # Create test config file
    test_config = {
        'model': {
            'text_model': 'gpt2',
            'temperature': 0.6
        },
        'system': {
            'cache_dir': '/tmp/cache'
        },
        'app': {
            'title': 'Test App'
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(test_config, f)
    
    # Load config
    config_manager = ConfigManager(config_path)
    config_manager.load_config()
    
    # Verify loaded values
    assert config_manager.model_config.text_model == 'gpt2'
    assert config_manager.model_config.temperature == 0.6
    assert config_manager.system_config.cache_dir == '/tmp/cache'
    assert config_manager.app_config.title == 'Test App'


def test_config_load_nonexistent_file(temp_config_dir):
    """Test loading configuration from non-existent file."""
    config_path = temp_config_dir / "nonexistent.yaml"
    config_manager = ConfigManager(config_path)
    
    # This should create default config
    config_manager.load_config()
    
    # Verify default values are used
    assert config_manager.model_config.temperature == 0.7
    assert config_manager.system_config.log_level == "INFO"
    assert config_manager.app_config.debug is False


def test_config_get_dict(temp_config_dir):
    """Test getting configuration as dictionary."""
    config_path = temp_config_dir / "config.yaml"
    config_manager = ConfigManager(config_path)
    
    # Modify some values
    config_manager.model_config.temperature = 0.9
    config_manager.system_config.log_level = "ERROR"
    
    # Get config dict
    config_dict = config_manager.get_config_dict()
    
    # Verify structure
    assert 'model' in config_dict
    assert 'system' in config_dict
    assert 'app' in config_dict
    
    # Verify values
    assert config_dict['model']['temperature'] == 0.9
    assert config_dict['system']['log_level'] == "ERROR"
    
    # Verify types
    assert isinstance(config_dict['model'], dict)
    assert isinstance(config_dict['system'], dict)
    assert isinstance(config_dict['app'], dict)


def test_config_save_default(temp_config_dir):
    """Test saving default configuration."""
    config_path = temp_config_dir / "default_config.yaml"
    config_manager = ConfigManager(config_path)
    
    # Save default config
    config_manager.save_default_config()
    
    # Check file exists
    assert config_path.exists()
    
    # Load and verify it's the default
    new_config_manager = ConfigManager(config_path)
    new_config_manager.load_config()
    
    assert new_config_manager.model_config.temperature == 0.7
    assert new_config_manager.system_config.log_level == "INFO"
    assert new_config_manager.app_config.debug is False


def test_config_invalid_file_format(temp_config_dir):
    """Test handling of invalid file format."""
    config_path = temp_config_dir / "config.txt"
    config_manager = ConfigManager(config_path)
    
    # Create invalid config file
    with open(config_path, 'w') as f:
        f.write("This is not valid YAML or JSON")
    
    # This should not crash and should use defaults
    config_manager.load_config()
    
    # Verify default values are used
    assert config_manager.model_config.temperature == 0.7
    assert config_manager.system_config.log_level == "INFO"


def test_config_partial_loading(temp_config_dir):
    """Test loading configuration with partial data."""
    config_path = temp_config_dir / "partial_config.yaml"
    
    # Create partial config file
    partial_config = {
        'model': {
            'temperature': 0.8
            # Missing other model fields
        }
        # Missing system and app sections
    }
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(partial_config, f)
    
    # Load config
    config_manager = ConfigManager(config_path)
    config_manager.load_config()
    
    # Verify loaded values
    assert config_manager.model_config.temperature == 0.8
    
    # Verify defaults are used for missing fields
    assert config_manager.model_config.max_length == 100  # Default
    assert config_manager.system_config.log_level == "INFO"  # Default
    assert config_manager.app_config.debug is False  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
