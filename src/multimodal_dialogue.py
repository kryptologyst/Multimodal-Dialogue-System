"""
Core multimodal dialogue system implementation.

This module provides the main MultimodalDialogueSystem class that handles
text and image inputs to generate contextual responses.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import warnings

import torch
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BlipProcessor, 
    BlipForConditionalGeneration,
    pipeline
)
import numpy as np
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


@dataclass
class DialogueResponse:
    """Response from the multimodal dialogue system."""
    text_response: str
    image_caption: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class MultimodalDialogueSystem:
    """
    A modern multimodal dialogue system that processes both text and image inputs.
    
    This system uses state-of-the-art transformer models for:
    - Text generation using causal language models
    - Image captioning using vision-language models
    - Contextual response generation combining both modalities
    """
    
    def __init__(
        self,
        text_model: str = "microsoft/DialoGPT-medium",
        vision_model: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ):
        """
        Initialize the multimodal dialogue system.
        
        Args:
            text_model: Hugging Face model name for text generation
            vision_model: Hugging Face model name for image captioning
            device: Device to run models on ('cpu', 'cuda', 'mps', or None for auto)
            max_length: Maximum length for generated text
            temperature: Sampling temperature for text generation
            do_sample: Whether to use sampling for text generation
        """
        self.text_model_name = text_model
        self.vision_model_name = vision_model
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._load_models()
    
    def _load_models(self) -> None:
        """Load and initialize the text and vision models."""
        try:
            logger.info("Loading text generation model...")
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            self.text_model = AutoModelForCausalLM.from_pretrained(self.text_model_name)
            self.text_model.to(self.device)
            
            # Add padding token if not present
            if self.text_tokenizer.pad_token is None:
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            
            logger.info("Loading vision-language model...")
            self.vision_processor = BlipProcessor.from_pretrained(self.vision_model_name)
            self.vision_model = BlipForConditionalGeneration.from_pretrained(self.vision_model_name)
            self.vision_model.to(self.device)
            
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def generate_text_response(self, user_input: str, context: Optional[str] = None) -> str:
        """
        Generate a text response based on user input and optional context.
        
        Args:
            user_input: The user's text input
            context: Optional context from image captioning
            
        Returns:
            Generated text response
        """
        try:
            # Prepare input text
            if context:
                input_text = f"User: {user_input}\nContext: {context}\nAssistant:"
            else:
                input_text = f"User: {user_input}\nAssistant:"
            
            # Tokenize input
            inputs = self.text_tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_length - 50
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.text_model.generate(
                    inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    pad_token_id=self.text_tokenizer.eos_token_id,
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text response: {e}")
            return "I apologize, but I encountered an error generating a response."
    
    def generate_image_caption(self, image_path: Union[str, Path, Image.Image]) -> str:
        """
        Generate a caption for the provided image.
        
        Args:
            image_path: Path to image file or PIL Image object
            
        Returns:
            Generated image caption
        """
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path.convert("RGB")
            
            # Process image
            inputs = self.vision_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.vision_model.generate(**inputs, max_length=50)
            
            caption = self.vision_processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Error generating image caption: {e}")
            return "Unable to process the image."
    
    def generate_response(
        self, 
        user_input: str, 
        image_path: Optional[Union[str, Path, Image.Image]] = None
    ) -> DialogueResponse:
        """
        Generate a multimodal response combining text and image inputs.
        
        Args:
            user_input: The user's text input
            image_path: Optional path to image file or PIL Image object
            
        Returns:
            DialogueResponse object containing the generated response
        """
        try:
            image_caption = None
            metadata = {"has_image": image_path is not None}
            
            # Process image if provided
            if image_path:
                image_caption = self.generate_image_caption(image_path)
                metadata["image_caption"] = image_caption
            
            # Generate text response with image context
            text_response = self.generate_text_response(user_input, image_caption)
            
            # Calculate confidence score (simplified)
            confidence_score = self._calculate_confidence(text_response, image_caption)
            metadata["confidence_score"] = confidence_score
            
            return DialogueResponse(
                text_response=text_response,
                image_caption=image_caption,
                confidence_score=confidence_score,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error generating multimodal response: {e}")
            return DialogueResponse(
                text_response="I apologize, but I encountered an error processing your request.",
                metadata={"error": str(e)}
            )
    
    def _calculate_confidence(self, text_response: str, image_caption: Optional[str]) -> float:
        """
        Calculate a simple confidence score for the response.
        
        Args:
            text_response: Generated text response
            image_caption: Optional image caption
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic-based confidence scoring
        score = 0.5  # Base score
        
        # Increase score for longer, more detailed responses
        if len(text_response.split()) > 5:
            score += 0.2
        
        # Increase score if image was processed successfully
        if image_caption and len(image_caption.split()) > 3:
            score += 0.2
        
        # Increase score for responses that seem more natural
        if any(word in text_response.lower() for word in ["the", "a", "an", "and", "or"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def batch_process(
        self, 
        inputs: List[Dict[str, Any]]
    ) -> List[DialogueResponse]:
        """
        Process multiple inputs in batch for efficiency.
        
        Args:
            inputs: List of dictionaries containing 'user_input' and optional 'image_path'
            
        Returns:
            List of DialogueResponse objects
        """
        responses = []
        for input_data in inputs:
            user_input = input_data.get("user_input", "")
            image_path = input_data.get("image_path")
            response = self.generate_response(user_input, image_path)
            responses.append(response)
        
        return responses
