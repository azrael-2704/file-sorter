"""Local inference implementation using pure Python libraries."""
import os
from typing import Generator, Dict, Any
import numpy as np
from PIL import Image
import cv2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class LocalTextInference:
    """Local replacement for text inference using scikit-learn and gensim."""
    
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def create_completion(self, prompt: str) -> Dict[str, Any]:
        """Generate text completion using heuristics and ML techniques."""
        # Extract content from various prompt types
        if "Text:" in prompt:
            text = prompt.split("Text:")[1].split("Summary:")[0].strip()
        elif "Summary:" in prompt:
            text = prompt.split("Summary:")[1].split("\n")[0].strip()
        else:
            text = prompt

        # Handle different prompt types
        if "filename" in prompt.lower():
            words = self._extract_keywords(text, max_words=3)
            result = "_".join(words)
        elif "category" in prompt.lower():
            words = self._extract_keywords(text, max_words=2)
            result = "_".join(words)
        else:
                # For general summarization use extractive summarization
                result = self._extractive_summarize(text)

        return {
            'choices': [{
                'text': result.strip()
            }]
        }
    
    def _extract_keywords(self, text: str, max_words: int = 3) -> list:
        """Extract key phrases using TF-IDF."""
        # Defensive: handle empty or stopword-only text
        if not text or not text.strip():
            return ["untitled"]

        # Use TF-IDF to extract top terms as keywords, but fall back to
        # simple token-frequency extraction if TF-IDF produces an empty
        # vocabulary (e.g. when the document is only stopwords).
        try:
            X = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = zip(feature_names, X.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            words = [word for word, score in sorted_scores[:max_words]]
            if words:
                return words
        except ValueError:
            # fall through to token-frequency fallback
            pass

        # Token-frequency fallback
        tokens = [w for w in word_tokenize(text.lower()) if w.isalnum() and w not in self.stop_words]
        if not tokens:
            # As a last resort, use a simple split and take first alnum substrings
            parts = [p for p in re.split(r"\W+", text) if p]
            return [parts[0].lower()[:max(3, len(parts[0]))]] if parts else ["untitled"]

        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_tokens[:max_words]]
    
    def _extractive_summarize(self, text: str, num_sentences: int = 3) -> str:
        """Simple extractive summarization using sentence scoring."""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
            
        # Calculate sentence scores based on word frequency
        word_freq = {}
        for word in word_tokenize(text.lower()):
            if word.isalnum() and word not in self.stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
                    
        # Get top sentences
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        summary_sentences = [s[0] for s in sorted_sentences[:num_sentences]]
        
        # Restore original order
        ordered_summary = [s for s in sentences if s in summary_sentences]
        return " ".join(ordered_summary)


class LocalVLMInference:
    """Local replacement for image inference using OpenCV."""
    
    def __init__(self, **kwargs):
        pass
        
    def _chat(self, prompt: str, image_path: str) -> Generator[Dict[str, Any], None, None]:
        """Generate image description using OpenCV-based analysis."""
        try:
            # Read image with PIL first (for format support)
            pil_image = Image.open(image_path)
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Basic image analysis
            height, width = cv_image.shape[:2]
            aspect_ratio = width / height
            orientation = "portrait" if height > width else "landscape"
            
            # Color analysis
            average_color = cv2.mean(cv_image)[:3]
            colors = self._analyze_colors(cv_image)

            # Try simple object/fruit detection using color + shape heuristics
            detected_object = self._detect_object_type(cv_image)
            
            # Edge detection for complexity estimation
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / (width * height)
            
            # Generate description
            desc_parts = []
            
            # Content-first description: prefer detected object labels over generic orientation
            if detected_object:
                desc_parts.append(f"This image appears to show a {detected_object}")
            else:
                # Basic properties
                desc_parts.append(f"This appears to be a {orientation} image")

            # Color description (secondary)
            if colors:
                desc_parts.append(f"predominantly showing {', '.join(colors[:2])} tones")
            
            # Content complexity
            if edge_density > 0.1:
                desc_parts.append("with complex detail or multiple elements")
            elif edge_density > 0.05:
                desc_parts.append("with moderate detail")
            else:
                desc_parts.append("with simple composition")
            
            description = " ".join(desc_parts)
            
            # Yield the description in chunks to match original interface
            chunk_size = 50
            for i in range(0, len(description), chunk_size):
                chunk = description[i:i + chunk_size]
                yield {
                    'choices': [{
                        'delta': {'content': chunk},
                        'index': 0
                    }]
                }
                
        except Exception as e:
            # Fallback to basic file analysis
            filename = os.path.basename(image_path)
            fallback_desc = f"An image file named {filename}"
            yield {
                'choices': [{
                    'delta': {'content': fallback_desc},
                    'index': 0
                }]
            }
    
    def _analyze_colors(self, image, n_colors=3):
        """Analyze dominant colors in the image."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Simple color binning
        colors = []
        h, s, v = cv2.split(hsv)
        
        # Average HSV values
        h_avg = np.mean(h)
        s_avg = np.mean(s)
        v_avg = np.mean(v)
        
        # Basic color classification
        if s_avg < 50:  # Low saturation indicates grayscale
            if v_avg < 85:
                colors.append("dark")
            elif v_avg > 170:
                colors.append("bright")
            return ["grayscale"] + colors
            
        # Color wheel segments
        if h_avg < 30:
            colors.append("red")
        elif h_avg < 90:
            colors.append("green")
        elif h_avg < 150:
            colors.append("blue")
        elif h_avg < 180:
            colors.append("purple")
            
        # Brightness description
        if v_avg < 85:
            colors.append("dark")
        elif v_avg > 170:
            colors.append("bright")
            
        return colors

    def _detect_object_type(self, image):
        """Simple heuristic detector for common fruits using color and shape.

        Returns a short label like 'apple', 'banana', 'orange' or None.
        This is a lightweight heuristic and will not match a full ML classifier,
        but helps avoid mis-labeling obvious fruit photos as e.g. 'landscape'.
        """
        try:
            # work on a copy
            img = image.copy()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Prepare color masks for red, yellow, orange
            # red (two ranges)
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

            # yellow (bananas)
            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # orange (oranges)
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([20, 255, 255])
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

            # basic morphology to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
            mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

            h, w = img.shape[:2]
            area = h * w

            def mask_area_ratio(m):
                return int(cv2.countNonZero(m)) / float(area)

            red_ratio = mask_area_ratio(mask_red)
            yellow_ratio = mask_area_ratio(mask_yellow)
            orange_ratio = mask_area_ratio(mask_orange)

            # Hough circle detection to identify round fruits (apple/orange)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                       param1=50, param2=30, minRadius=10, maxRadius=int(min(h, w)/2))

            is_round = circles is not None

            # Heuristics: banana => strong yellow mask and elongated contours
            if yellow_ratio > 0.02:
                # check elongation via contours
                contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw == 0 or ch == 0:
                        continue
                    aspect = max(cw / ch, ch / cw)
                    if aspect > 2.0 and cv2.contourArea(cnt) > 0.001 * area:
                        return 'banana'
                # if mostly yellow and not elongated, maybe apple/orange (yellow apple varieties exist)
                if yellow_ratio > 0.05 and not is_round:
                    return 'fruit (yellow)'

            # Orange detection
            if orange_ratio > 0.02:
                if is_round or orange_ratio > 0.04:
                    return 'orange'

            # Red -> apple (or other red fruit)
            if red_ratio > 0.02:
                if is_round or red_ratio > 0.04:
                    return 'apple'

            # If round and significant mask area in any of the fruit colors, prefer round fruit
            if is_round:
                if red_ratio > 0.01:
                    return 'apple'
                if orange_ratio > 0.01:
                    return 'orange'
                if yellow_ratio > 0.01:
                    return 'fruit (yellow)'

            return None
        except Exception:
            return None


def download_models():
    """No-op download helper for this lightweight local setup.

    This project uses pure-Python tools (scikit-learn, OpenCV) and doesn't
    require large transformer model downloads. Keep this function so
    `download_models.py` can run without error. It will ensure NLTK data is
    present.
    """
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')