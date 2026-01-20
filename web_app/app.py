"""
Streamlit web interface for the multimodal dialogue system.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional
from PIL import Image
import json
import time

from src.multimodal_dialogue import MultimodalDialogueSystem, DialogueResponse
from src.config import ConfigManager
from src.data_utils import SyntheticDataGenerator, DataLoader


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDialogueApp:
    """Streamlit application for the multimodal dialogue system."""
    
    def __init__(self):
        """Initialize the application."""
        self.config_manager = ConfigManager()
        self.config_manager.load_config()
        self.system = None
        self.data_generator = SyntheticDataGenerator()
        self.data_loader = DataLoader()
        
        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
    
    def initialize_system(self) -> bool:
        """Initialize the multimodal dialogue system."""
        try:
            with st.spinner("Loading AI models... This may take a few minutes on first run."):
                self.system = MultimodalDialogueSystem(
                    text_model=self.config_manager.model_config.text_model,
                    vision_model=self.config_manager.model_config.vision_model,
                    device=self.config_manager.model_config.device,
                    max_length=self.config_manager.model_config.max_length,
                    temperature=self.config_manager.model_config.temperature,
                    do_sample=self.config_manager.model_config.do_sample
                )
                st.session_state.system_initialized = True
                return True
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            logger.error(f"System initialization failed: {e}")
            return False
    
    def generate_sample_data(self):
        """Generate sample data for demonstration."""
        with st.spinner("Generating sample data..."):
            dataset_info = self.data_generator.generate_complete_dataset(
                num_images=5, 
                num_dialogues=10
            )
            st.success(f"Generated {dataset_info['num_images']} images and {dataset_info['num_dialogues']} dialogues!")
            return dataset_info
    
    def display_conversation_history(self):
        """Display conversation history."""
        if st.session_state.conversation_history:
            st.subheader("Conversation History")
            for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Exchange {len(st.session_state.conversation_history) - i}"):
                    st.write(f"**User:** {entry['user_input']}")
                    if entry.get('image_path'):
                        st.write(f"**Image:** {entry['image_path']}")
                    st.write(f"**Response:** {entry['response']}")
                    if entry.get('confidence_score'):
                        st.write(f"**Confidence:** {entry['confidence_score']:.2f}")
    
    def process_user_input(self, user_input: str, uploaded_file: Optional[bytes] = None) -> DialogueResponse:
        """Process user input and generate response."""
        if not self.system:
            return DialogueResponse(
                text_response="System not initialized. Please wait for models to load.",
                metadata={"error": "System not initialized"}
            )
        
        try:
            # Handle uploaded image
            image_path = None
            if uploaded_file:
                # Save uploaded image temporarily
                image = Image.open(uploaded_file)
                temp_path = Path("temp_uploaded_image.jpg")
                image.save(temp_path)
                image_path = temp_path
            
            # Generate response
            response = self.system.generate_response(user_input, image_path)
            
            # Clean up temporary file
            if image_path and image_path.exists():
                image_path.unlink()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return DialogueResponse(
                text_response=f"I encountered an error: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def run(self):
        """Run the Streamlit application."""
        # Page configuration
        st.set_page_config(
            page_title=self.config_manager.app_config.title,
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🤖 Multimodal Dialogue System")
        st.markdown(self.config_manager.app_config.description)
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            # Model settings
            st.subheader("Model Settings")
            text_model = st.selectbox(
                "Text Model",
                ["microsoft/DialoGPT-medium", "gpt2", "distilgpt2"],
                index=0
            )
            
            vision_model = st.selectbox(
                "Vision Model", 
                ["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"],
                index=0
            )
            
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
            max_length = st.slider("Max Length", 50, 200, 100, 10)
            
            # Data generation
            st.subheader("Sample Data")
            if st.button("Generate Sample Data"):
                self.generate_sample_data()
            
            # System status
            st.subheader("System Status")
            if st.session_state.system_initialized:
                st.success("✅ System Ready")
            else:
                st.warning("⚠️ System Not Initialized")
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "📁 Data", "⚙️ Settings"])
        
        with tab1:
            st.header("Multimodal Chat Interface")
            
            # Initialize system if not done
            if not st.session_state.system_initialized:
                if st.button("Initialize System"):
                    self.initialize_system()
            else:
                # Chat interface
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    user_input = st.text_area(
                        "Enter your message:",
                        placeholder="Ask me about an image or have a conversation...",
                        height=100
                    )
                
                with col2:
                    uploaded_file = st.file_uploader(
                        "Upload an image (optional):",
                        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
                        help="Upload an image to get multimodal responses"
                    )
                
                if st.button("Send Message", type="primary"):
                    if user_input.strip():
                        # Process input
                        response = self.process_user_input(user_input, uploaded_file)
                        
                        # Display response
                        st.subheader("Response")
                        st.write(response.text_response)
                        
                        if response.image_caption:
                            st.write(f"**Image Caption:** {response.image_caption}")
                        
                        if response.confidence_score:
                            st.write(f"**Confidence Score:** {response.confidence_score:.2f}")
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            'user_input': user_input,
                            'image_path': uploaded_file.name if uploaded_file else None,
                            'response': response.text_response,
                            'confidence_score': response.confidence_score,
                            'timestamp': time.time()
                        })
                        
                        # Display uploaded image if present
                        if uploaded_file:
                            st.subheader("Uploaded Image")
                            image = Image.open(uploaded_file)
                            st.image(image, caption="Uploaded Image", use_column_width=True)
                    else:
                        st.warning("Please enter a message.")
                
                # Display conversation history
                self.display_conversation_history()
        
        with tab2:
            st.header("Analytics Dashboard")
            
            if st.session_state.conversation_history:
                # Basic statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Exchanges", len(st.session_state.conversation_history))
                
                with col2:
                    avg_confidence = sum(
                        entry.get('confidence_score', 0) 
                        for entry in st.session_state.conversation_history
                    ) / len(st.session_state.conversation_history)
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                with col3:
                    exchanges_with_images = sum(
                        1 for entry in st.session_state.conversation_history 
                        if entry.get('image_path')
                    )
                    st.metric("Multimodal Exchanges", exchanges_with_images)
                
                # Confidence distribution
                st.subheader("Confidence Score Distribution")
                confidence_scores = [
                    entry.get('confidence_score', 0) 
                    for entry in st.session_state.conversation_history
                ]
                if confidence_scores:
                    st.bar_chart(confidence_scores)
            else:
                st.info("No conversation data available yet. Start chatting to see analytics!")
        
        with tab3:
            st.header("Data Management")
            
            # Sample data section
            st.subheader("Sample Data")
            if st.button("Generate New Sample Dataset"):
                dataset_info = self.generate_sample_data()
                
                # Display dataset info
                st.json(dataset_info)
            
            # Load and display existing data
            try:
                dialogues = self.data_loader.load_dialogues()
                st.subheader(f"Loaded Dialogues ({len(dialogues)} total)")
                
                # Data validation
                validation_results = self.data_loader.validate_data(dialogues)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Valid Dialogues", len(dialogues))
                    st.metric("With Images", validation_results["stats"]["with_images"])
                
                with col2:
                    st.metric("Categories", len(validation_results["stats"]["categories"]))
                    st.metric("Difficulties", len(validation_results["stats"]["difficulties"]))
                
                # Display sample dialogues
                if dialogues:
                    st.subheader("Sample Dialogues")
                    for i, dialogue in enumerate(dialogues[:3]):
                        with st.expander(f"Dialogue {i+1}: {dialogue.get('category', 'Unknown')}"):
                            st.write(f"**Input:** {dialogue['user_input']}")
                            st.write(f"**Expected Response:** {dialogue['expected_response']}")
                            if dialogue.get('image_path'):
                                st.write(f"**Image:** {dialogue['image_path']}")
                
            except FileNotFoundError:
                st.info("No sample data found. Click 'Generate New Sample Dataset' to create some!")
        
        with tab4:
            st.header("System Settings")
            
            # Configuration display
            st.subheader("Current Configuration")
            config_dict = self.config_manager.get_config_dict()
            st.json(config_dict)
            
            # Save configuration
            if st.button("Save Current Configuration"):
                self.config_manager.save_config()
                st.success("Configuration saved!")
            
            # System information
            st.subheader("System Information")
            if self.system:
                st.write(f"**Text Model:** {self.system.text_model_name}")
                st.write(f"**Vision Model:** {self.system.vision_model_name}")
                st.write(f"**Device:** {self.system.device}")
                st.write(f"**Max Length:** {self.system.max_length}")
                st.write(f"**Temperature:** {self.system.temperature}")
            else:
                st.warning("System not initialized")


def main():
    """Main function to run the application."""
    app = MultimodalDialogueApp()
    app.run()


if __name__ == "__main__":
    main()
