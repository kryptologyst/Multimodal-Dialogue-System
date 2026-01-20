"""
Data utilities and synthetic data generation for the multimodal dialogue system.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class SyntheticDataGenerator:
    """Generates synthetic data for testing and demonstration purposes."""
    
    def __init__(self, output_dir: Path = Path("data")):
        """
        Initialize the synthetic data generator.
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_sample_images(self, num_images: int = 10) -> List[Path]:
        """
        Generate sample images with simple shapes and text.
        
        Args:
            num_images: Number of images to generate
            
        Returns:
            List of paths to generated images
        """
        image_paths = []
        
        # Sample image descriptions for captions
        image_descriptions = [
            "A red circle on a white background",
            "A blue square with yellow text",
            "A green triangle pointing upward",
            "A purple rectangle with diagonal stripes",
            "A yellow star on a black background",
            "A red heart shape with white outline",
            "A blue diamond with gradient fill",
            "A green oval with polka dots",
            "A purple hexagon with geometric pattern",
            "A yellow crescent moon on dark blue background"
        ]
        
        for i in range(num_images):
            # Create a new image
            img = Image.new('RGB', (256, 256), color='white')
            draw = ImageDraw.Draw(img)
            
            # Generate random shapes and colors
            shape_type = random.choice(['circle', 'rectangle', 'triangle', 'star'])
            color = random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
            
            if shape_type == 'circle':
                draw.ellipse([50, 50, 200, 200], fill=color, outline='black', width=2)
            elif shape_type == 'rectangle':
                draw.rectangle([50, 50, 200, 200], fill=color, outline='black', width=2)
            elif shape_type == 'triangle':
                points = [(128, 50), (50, 200), (200, 200)]
                draw.polygon(points, fill=color, outline='black', width=2)
            elif shape_type == 'star':
                # Simple star shape
                points = []
                for angle in range(0, 360, 36):
                    x = 128 + 60 * np.cos(np.radians(angle))
                    y = 128 + 60 * np.sin(np.radians(angle))
                    points.extend([x, y])
                draw.polygon(points, fill=color, outline='black', width=2)
            
            # Add some text
            try:
                font = ImageFont.load_default()
                text = f"Sample {i+1}"
                draw.text((10, 10), text, fill='black', font=font)
            except:
                draw.text((10, 10), f"Sample {i+1}", fill='black')
            
            # Save image
            image_path = self.output_dir / f"sample_image_{i+1:03d}.png"
            img.save(image_path)
            image_paths.append(image_path)
        
        return image_paths
    
    def generate_sample_dialogues(self, num_dialogues: int = 20) -> List[Dict[str, Any]]:
        """
        Generate sample dialogue data.
        
        Args:
            num_dialogues: Number of dialogues to generate
            
        Returns:
            List of dialogue dictionaries
        """
        dialogues = []
        
        # Sample user inputs
        user_inputs = [
            "What do you see in this image?",
            "Describe the picture for me",
            "Can you tell me about this image?",
            "What's happening in this photo?",
            "Explain what you see",
            "What colors are in this image?",
            "Describe the shapes you see",
            "What objects are visible?",
            "Tell me about the composition",
            "What's the main subject?",
            "How would you describe this scene?",
            "What details do you notice?",
            "What's interesting about this image?",
            "Can you identify the elements?",
            "What story does this image tell?",
            "Describe the mood of this picture",
            "What emotions does this evoke?",
            "What's the artistic style?",
            "How would you categorize this?",
            "What makes this image unique?"
        ]
        
        # Sample responses
        sample_responses = [
            "I can see a beautiful composition with vibrant colors and interesting shapes.",
            "This image shows a creative arrangement of geometric forms and patterns.",
            "The image contains several distinct elements that work together harmoniously.",
            "I notice the use of contrasting colors and bold shapes in this composition.",
            "This appears to be an abstract or artistic representation with symbolic elements.",
            "The image features a balanced layout with both positive and negative space.",
            "I can identify various visual elements that create a dynamic composition.",
            "This picture demonstrates interesting use of color theory and design principles.",
            "The image shows a thoughtful arrangement of visual elements and textures.",
            "I see a creative interpretation that combines different artistic techniques."
        ]
        
        for i in range(num_dialogues):
            dialogue = {
                "id": f"dialogue_{i+1:03d}",
                "user_input": random.choice(user_inputs),
                "image_path": f"sample_image_{random.randint(1, 10):03d}.png",
                "expected_response": random.choice(sample_responses),
                "category": random.choice(["description", "analysis", "interpretation", "general"]),
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "metadata": {
                    "generated": True,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "version": "1.0.0"
                }
            }
            dialogues.append(dialogue)
        
        return dialogues
    
    def save_sample_data(self, dialogues: List[Dict[str, Any]]) -> Path:
        """
        Save sample dialogue data to JSON file.
        
        Args:
            dialogues: List of dialogue dictionaries
            
        Returns:
            Path to saved data file
        """
        data_file = self.output_dir / "sample_dialogues.json"
        
        with open(data_file, 'w') as f:
            json.dump(dialogues, f, indent=2)
        
        return data_file
    
    def generate_complete_dataset(self, num_images: int = 10, num_dialogues: int = 20) -> Dict[str, Any]:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_images: Number of images to generate
            num_dialogues: Number of dialogues to generate
            
        Returns:
            Dictionary containing dataset information
        """
        # Generate images
        image_paths = self.generate_sample_images(num_images)
        
        # Generate dialogues
        dialogues = self.generate_sample_dialogues(num_dialogues)
        
        # Save dialogues
        data_file = self.save_sample_data(dialogues)
        
        # Create dataset info
        dataset_info = {
            "name": "Synthetic Multimodal Dialogue Dataset",
            "version": "1.0.0",
            "description": "Synthetic dataset for testing multimodal dialogue system",
            "num_images": len(image_paths),
            "num_dialogues": len(dialogues),
            "image_directory": str(self.output_dir),
            "data_file": str(data_file),
            "generated_at": "2024-01-01T00:00:00Z",
            "categories": list(set(d["category"] for d in dialogues)),
            "difficulties": list(set(d["difficulty"] for d in dialogues))
        }
        
        # Save dataset info
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        return dataset_info


class DataLoader:
    """Utility class for loading and managing data."""
    
    def __init__(self, data_dir: Path = Path("data")):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
    
    def load_dialogues(self, data_file: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Load dialogue data from JSON file.
        
        Args:
            data_file: Path to data file
            
        Returns:
            List of dialogue dictionaries
        """
        if data_file is None:
            data_file = self.data_dir / "sample_dialogues.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, 'r') as f:
            dialogues = json.load(f)
        
        return dialogues
    
    def get_image_path(self, image_filename: str) -> Path:
        """
        Get full path to image file.
        
        Args:
            image_filename: Name of image file
            
        Returns:
            Full path to image file
        """
        return self.data_dir / image_filename
    
    def validate_data(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate dialogue data structure.
        
        Args:
            dialogues: List of dialogue dictionaries
            
        Returns:
            Validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_dialogues": len(dialogues),
                "with_images": 0,
                "without_images": 0,
                "categories": {},
                "difficulties": {}
            }
        }
        
        required_fields = ["id", "user_input", "expected_response"]
        
        for i, dialogue in enumerate(dialogues):
            # Check required fields
            for field in required_fields:
                if field not in dialogue:
                    validation_results["errors"].append(f"Dialogue {i}: Missing required field '{field}'")
                    validation_results["valid"] = False
            
            # Check image path
            if "image_path" in dialogue and dialogue["image_path"]:
                validation_results["stats"]["with_images"] += 1
                image_path = self.get_image_path(dialogue["image_path"])
                if not image_path.exists():
                    validation_results["warnings"].append(f"Dialogue {i}: Image file not found: {dialogue['image_path']}")
            else:
                validation_results["stats"]["without_images"] += 1
            
            # Count categories and difficulties
            if "category" in dialogue:
                cat = dialogue["category"]
                validation_results["stats"]["categories"][cat] = validation_results["stats"]["categories"].get(cat, 0) + 1
            
            if "difficulty" in dialogue:
                diff = dialogue["difficulty"]
                validation_results["stats"]["difficulties"][diff] = validation_results["stats"]["difficulties"].get(diff, 0) + 1
        
        return validation_results
