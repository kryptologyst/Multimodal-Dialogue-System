#!/usr/bin/env python3
"""
Example script demonstrating the multimodal dialogue system.

This script shows how to use the multimodal dialogue system
with both text-only and multimodal inputs.
"""

import logging
from pathlib import Path
from PIL import Image
import tempfile

from src.multimodal_dialogue import MultimodalDialogueSystem
from src.data_utils import SyntheticDataGenerator, DataLoader


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_text_only():
    """Example of text-only conversation."""
    print("=" * 60)
    print("📝 TEXT-ONLY CONVERSATION EXAMPLE")
    print("=" * 60)
    
    # Initialize the system
    print("🤖 Initializing multimodal dialogue system...")
    system = MultimodalDialogueSystem()
    
    # Example conversations
    conversations = [
        "Hello, how are you today?",
        "Tell me about artificial intelligence",
        "What's the weather like?",
        "Can you explain machine learning?",
        "What are your capabilities?"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"\n💬 Conversation {i}:")
        print(f"👤 User: {user_input}")
        
        # Generate response
        response = system.generate_response(user_input)
        
        print(f"🤖 Assistant: {response.text_response}")
        print(f"📊 Confidence: {response.confidence_score:.2f}")
        print("-" * 40)


def example_multimodal():
    """Example of multimodal conversation with images."""
    print("\n" + "=" * 60)
    print("🖼️ MULTIMODAL CONVERSATION EXAMPLE")
    print("=" * 60)
    
    # Initialize the system
    print("🤖 Initializing multimodal dialogue system...")
    system = MultimodalDialogueSystem()
    
    # Generate sample images
    print("🎨 Generating sample images...")
    generator = SyntheticDataGenerator()
    image_paths = generator.generate_sample_images(3)
    
    # Example multimodal conversations
    conversations = [
        ("What do you see in this image?", image_paths[0]),
        ("Describe the colors and shapes", image_paths[1]),
        ("What story does this image tell?", image_paths[2])
    ]
    
    for i, (user_input, image_path) in enumerate(conversations, 1):
        print(f"\n💬 Multimodal Conversation {i}:")
        print(f"👤 User: {user_input}")
        print(f"📷 Image: {image_path.name}")
        
        # Generate response
        response = system.generate_response(user_input, image_path)
        
        print(f"🤖 Assistant: {response.text_response}")
        print(f"📷 Image Caption: {response.image_caption}")
        print(f"📊 Confidence: {response.confidence_score:.2f}")
        print("-" * 40)


def example_batch_processing():
    """Example of batch processing multiple inputs."""
    print("\n" + "=" * 60)
    print("📦 BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Initialize the system
    print("🤖 Initializing multimodal dialogue system...")
    system = MultimodalDialogueSystem()
    
    # Generate sample data
    print("🎨 Generating sample data...")
    generator = SyntheticDataGenerator()
    dataset_info = generator.generate_complete_dataset(num_images=2, num_dialogues=3)
    
    # Load dialogues
    loader = DataLoader()
    dialogues = loader.load_dialogues()
    
    # Prepare batch inputs
    batch_inputs = []
    for dialogue in dialogues[:3]:  # Use first 3 dialogues
        input_data = {
            "user_input": dialogue["user_input"],
            "image_path": loader.get_image_path(dialogue["image_path"]) if dialogue.get("image_path") else None
        }
        batch_inputs.append(input_data)
    
    print(f"📊 Processing {len(batch_inputs)} inputs in batch...")
    
    # Process batch
    responses = system.batch_process(batch_inputs)
    
    # Display results
    for i, (input_data, response) in enumerate(zip(batch_inputs, responses), 1):
        print(f"\n📝 Batch Item {i}:")
        print(f"👤 User: {input_data['user_input']}")
        if input_data['image_path']:
            print(f"📷 Image: {input_data['image_path'].name}")
        print(f"🤖 Assistant: {response.text_response}")
        if response.image_caption:
            print(f"📷 Caption: {response.image_caption}")
        print(f"📊 Confidence: {response.confidence_score:.2f}")
        print("-" * 40)


def example_custom_image():
    """Example with a custom uploaded image."""
    print("\n" + "=" * 60)
    print("🎨 CUSTOM IMAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize the system
    print("🤖 Initializing multimodal dialogue system...")
    system = MultimodalDialogueSystem()
    
    # Create a custom image
    print("🎨 Creating a custom test image...")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        # Create a simple test image
        img = Image.new('RGB', (200, 200), color='lightblue')
        
        # Add some simple shapes
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a circle
        draw.ellipse([50, 50, 150, 150], fill='red', outline='black', width=3)
        
        # Add text
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
            draw.text((10, 10), "Test Image", fill='black', font=font)
        except:
            draw.text((10, 10), "Test Image", fill='black')
        
        # Save image
        img.save(tmp_file.name)
        image_path = Path(tmp_file.name)
    
    # Example conversations with custom image
    conversations = [
        "What do you see in this image?",
        "Describe the colors and shapes in detail",
        "What emotions does this image evoke?",
        "How would you categorize this image?"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"\n💬 Custom Image Conversation {i}:")
        print(f"👤 User: {user_input}")
        
        # Generate response
        response = system.generate_response(user_input, image_path)
        
        print(f"🤖 Assistant: {response.text_response}")
        print(f"📷 Image Caption: {response.image_caption}")
        print(f"📊 Confidence: {response.confidence_score:.2f}")
        print("-" * 40)
    
    # Clean up
    image_path.unlink()


def main():
    """Run all examples."""
    setup_logging()
    
    print("🚀 MULTIMODAL DIALOGUE SYSTEM - EXAMPLES")
    print("This script demonstrates various features of the system.")
    print("Note: First run may take time to download models.")
    
    try:
        # Run examples
        example_text_only()
        example_multimodal()
        example_batch_processing()
        example_custom_image()
        
        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("  - Try the web interface: streamlit run web_app/app.py")
        print("  - Use CLI: python src/cli.py chat")
        print("  - Run tests: python -m pytest tests/ -v")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
