import os
import time
from typing import Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import asyncio

# Load environment variables
load_dotenv()

# Configure API
API_KEY = os.getenv('GOOGLE_API_KEY')
REQUEST_DELAY = float(os.getenv('GEMINI_REQUEST_DELAY', '3'))
last_request_time = 0

def configure_gemini():
    """Configure the Gemini API with the API key."""
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=API_KEY)

def enforce_rate_limit():
    """Enforce rate limiting between API calls."""
    global last_request_time
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    
    if time_since_last_request < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - time_since_last_request)
    
    last_request_time = time.time()

async def analyze_image_with_gemini(image_path: str) -> Tuple[str, str, bool]:
    """
    Analyze an image using Gemini Vision API.
    Returns: (description, object_type, success)
    """
    try:
        # Configure if not already done
        if not API_KEY:
            configure_gemini()

        # Enforce rate limiting
        enforce_rate_limit()

        # Load and prepare the image
        img = Image.open(image_path)
        
        # Get Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Prepare the prompt
        prompt = """
        Analyze this image and provide:
        1. A detailed description of what you see
        2. The main object or subject type (if any)
        3. Any relevant categories or tags
        Format your response as: description|object_type|category
        Keep the description under 100 words.
        """

        # Generate response
        response = model.generate_content([prompt, img])
        
        # Parse response
        parts = response.text.split('|')
        if len(parts) >= 2:
            description = parts[0].strip()
            object_type = parts[1].strip()
            return description, object_type, True
        else:
            return "Could not analyze image", "unknown", False

    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        return "Error analyzing image", "unknown", False

def is_api_available() -> bool:
    """Check if the Gemini API is properly configured and available."""
    try:
        configure_gemini()
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        return True
    except:
        return False