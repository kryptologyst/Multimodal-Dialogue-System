"""
Configuration management for the multimodal dialogue system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    text_model: str = "microsoft/DialoGPT-medium"
    vision_model: str = "Salesforce/blip-image-captioning-base"
    device: Optional[str] = None
    max_length: int = 100
    temperature: float = 0.7
    do_sample: bool = True


@dataclass
class SystemConfig:
    """Configuration for system parameters."""
    log_level: str = "INFO"
    cache_dir: str = "./cache"
    data_dir: str = "./data"
    models_dir: str = "./models"
    max_image_size: int = 512
    supported_image_formats: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


@dataclass
class AppConfig:
    """Configuration for the application."""
    title: str = "Multimodal Dialogue System"
    description: str = "A modern multimodal dialogue system combining text and images"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8501


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("config/config.yaml")
        self.model_config = ModelConfig()
        self.system_config = SystemConfig()
        self.app_config = AppConfig()
    
    def load_config(self, config_path: Optional[Path] = None) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            self.config_path = config_path
        
        if not self.config_path.exists():
            self.save_default_config()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configurations
            if 'model' in config_data:
                for key, value in config_data['model'].items():
                    if hasattr(self.model_config, key):
                        setattr(self.model_config, key, value)
            
            if 'system' in config_data:
                for key, value in config_data['system'].items():
                    if hasattr(self.system_config, key):
                        setattr(self.system_config, key, value)
            
            if 'app' in config_data:
                for key, value in config_data['app'].items():
                    if hasattr(self.app_config, key):
                        setattr(self.app_config, key, value)
                        
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
    
    def save_config(self, config_path: Optional[Path] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        if config_path:
            self.config_path = config_path
        
        config_data = {
            'model': asdict(self.model_config),
            'system': asdict(self.system_config),
            'app': asdict(self.app_config)
        }
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def save_default_config(self) -> None:
        """Save default configuration to file."""
        self.save_config()
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            'model': asdict(self.model_config),
            'system': asdict(self.system_config),
            'app': asdict(self.app_config)
        }
