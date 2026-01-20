#!/usr/bin/env python3
"""
Command-line interface for the multimodal dialogue system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.multimodal_dialogue import MultimodalDialogueSystem
from src.config import ConfigManager
from src.data_utils import SyntheticDataGenerator, DataLoader


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('multimodal_dialogue.log')
        ]
    )


def chat_mode(system: MultimodalDialogueSystem) -> None:
    """Interactive chat mode."""
    print("🤖 Multimodal Dialogue System - Chat Mode")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'help' for available commands")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'clear':
                print("\033[2J\033[H")  # Clear screen
                continue
            elif not user_input:
                continue
            
            # Process input
            print("🤖 Processing...")
            response = system.generate_response(user_input)
            
            print(f"🤖 Assistant: {response.text_response}")
            if response.image_caption:
                print(f"📷 Image Caption: {response.image_caption}")
            if response.confidence_score:
                print(f"📊 Confidence: {response.confidence_score:.2f}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def print_help() -> None:
    """Print help information."""
    print("\n📖 Available Commands:")
    print("  help     - Show this help message")
    print("  clear    - Clear the screen")
    print("  quit/exit - End the conversation")
    print("\n💡 Tips:")
    print("  - Ask questions about images by providing image paths")
    print("  - Use descriptive language for better responses")
    print("  - The system works with both text-only and multimodal inputs")


def batch_mode(system: MultimodalDialogueSystem, input_file: Path, output_file: Optional[Path] = None) -> None:
    """Batch processing mode."""
    print(f"📁 Processing batch input from: {input_file}")
    
    try:
        data_loader = DataLoader()
        dialogues = data_loader.load_dialogues(input_file)
        
        print(f"📊 Found {len(dialogues)} dialogues to process")
        
        results = []
        for i, dialogue in enumerate(dialogues):
            print(f"🔄 Processing dialogue {i+1}/{len(dialogues)}")
            
            # Get image path if available
            image_path = None
            if dialogue.get('image_path'):
                image_path = data_loader.get_image_path(dialogue['image_path'])
            
            # Generate response
            response = system.generate_response(
                dialogue['user_input'], 
                image_path
            )
            
            # Store result
            result = {
                'id': dialogue.get('id', f'dialogue_{i+1}'),
                'user_input': dialogue['user_input'],
                'generated_response': response.text_response,
                'expected_response': dialogue.get('expected_response'),
                'image_caption': response.image_caption,
                'confidence_score': response.confidence_score,
                'metadata': response.metadata
            }
            results.append(result)
        
        # Save results
        if output_file is None:
            output_file = Path("batch_results.json")
        
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        sys.exit(1)


def demo_mode() -> None:
    """Demo mode with sample data."""
    print("🎯 Demo Mode - Generating sample data and running examples")
    
    try:
        # Generate sample data
        generator = SyntheticDataGenerator()
        dataset_info = generator.generate_complete_dataset(num_images=3, num_dialogues=5)
        
        print(f"📊 Generated dataset:")
        print(f"  - Images: {dataset_info['num_images']}")
        print(f"  - Dialogues: {dataset_info['num_dialogues']}")
        
        # Initialize system
        print("\n🤖 Initializing multimodal dialogue system...")
        system = MultimodalDialogueSystem()
        
        # Load sample data
        data_loader = DataLoader()
        dialogues = data_loader.load_dialogues()
        
        # Run sample dialogues
        print("\n💬 Running sample dialogues:")
        print("-" * 50)
        
        for i, dialogue in enumerate(dialogues[:3]):  # Show first 3
            print(f"\n📝 Sample {i+1}:")
            print(f"👤 User: {dialogue['user_input']}")
            
            # Get image path if available
            image_path = None
            if dialogue.get('image_path'):
                image_path = data_loader.get_image_path(dialogue['image_path'])
                print(f"📷 Image: {dialogue['image_path']}")
            
            # Generate response
            response = system.generate_response(dialogue['user_input'], image_path)
            
            print(f"🤖 Assistant: {response.text_response}")
            if response.image_caption:
                print(f"📷 Caption: {response.image_caption}")
            if response.confidence_score:
                print(f"📊 Confidence: {response.confidence_score:.2f}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in demo mode: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Multimodal Dialogue System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s chat                    # Interactive chat mode
  %(prog)s demo                    # Run demo with sample data
  %(prog)s batch input.json        # Process batch input file
  %(prog)s batch input.json -o results.json  # Save results to file
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['chat', 'demo', 'batch'],
        help='Operation mode'
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        type=Path,
        help='Input file for batch mode'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file for batch mode'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config_manager.load_config()
    
    try:
        if args.mode == 'chat':
            print("🚀 Starting interactive chat mode...")
            system = MultimodalDialogueSystem(
                text_model=config_manager.model_config.text_model,
                vision_model=config_manager.model_config.vision_model,
                device=config_manager.model_config.device,
                max_length=config_manager.model_config.max_length,
                temperature=config_manager.model_config.temperature,
                do_sample=config_manager.model_config.do_sample
            )
            chat_mode(system)
            
        elif args.mode == 'demo':
            demo_mode()
            
        elif args.mode == 'batch':
            if not args.input_file:
                print("❌ Error: Input file required for batch mode")
                sys.exit(1)
            
            if not args.input_file.exists():
                print(f"❌ Error: Input file not found: {args.input_file}")
                sys.exit(1)
            
            print("🚀 Starting batch processing mode...")
            system = MultimodalDialogueSystem(
                text_model=config_manager.model_config.text_model,
                vision_model=config_manager.model_config.vision_model,
                device=config_manager.model_config.device,
                max_length=config_manager.model_config.max_length,
                temperature=config_manager.model_config.temperature,
                do_sample=config_manager.model_config.do_sample
            )
            batch_mode(system, args.input_file, args.output)
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
