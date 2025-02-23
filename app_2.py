import streamlit as st
from PIL import Image
import torch
from transformers import (
    CLIPProcessor, 
    CLIPModel,
    AutoTokenizer,
    AutoModelForCausalLM
)
import re

# Configure Streamlit page
st.set_page_config(
    page_title="AI Caption Generator",
    page_icon="📸",
    layout="wide"
)

# Custom CSS with enhanced UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        color: #666;
        font-size: 1.2rem;
    }
    .output-text, .hashtag-text {
        color: #000;
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 1.2rem;
    }
    .icon-btn {
        font-size: 1.5rem;
        color: #1E88E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #000;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load models
@st.cache_resource
def load_models():
    """Load and cache models to improve performance"""
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return clip_processor, clip_model, tokenizer, model

def get_image_features(image, clip_processor, clip_model):
    """Extract visual features and concepts from image"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    inputs = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs)
    
    candidate_concepts = [
        "landscape", "portrait", "food", "architecture", "nature",
        "urban", "abstract", "animals", "people", "technology",
        "art", "fashion", "sports", "vehicles", "interior"
    ]
    
    text_inputs = clip_processor(
        text=candidate_concepts,
        return_tensors="pt",
        padding=True
    )
    
    text_features = clip_model.get_text_features(**text_inputs)
    similarity = torch.nn.functional.cosine_similarity(
        image_features, text_features
    )
    
    return [candidate_concepts[i] for i in similarity.argsort(descending=True)[:3]]

def generate_description(concepts, tokenizer, model):
    """Generate image caption based on detected concepts"""
    prompt = f"""
    Write a creative and engaging Instagram caption for an image featuring: {', '.join(concepts)}.
    Make it natural, engaging, and memorable. Keep it under 100 words.

    Caption:
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Caption:")[-1].strip()

def generate_hashtags(caption, concepts):
    """Generate relevant hashtags based on caption and concepts"""
    words = re.findall(r'\w+', caption.lower())
    common_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    hashtags = [f"#{word}" for word in words if word not in common_words and len(word) > 2]
    
    concept_hashtags = {
        'landscape': ['#landscapephotography', '#naturephotography', '#scenic'],
        'portrait': ['#portraitphotography', '#portraiture', '#model'],
        'food': ['#foodphotography', '#foodie', '#culinary'],
        'architecture': ['#architecturephotography', '#building', '#design'],
        'nature': ['#naturephotography', '#outdoors', '#wildlife'],
        'urban': ['#urbanphotography', '#citylife', '#street'],
        'abstract': ['#abstractart', '#artistic', '#creative'],
        'animals': ['#wildlifephotography', '#animal', '#pets'],
        'people': ['#peoplephotography', '#lifestyle', '#candid'],
        'technology': ['#tech', '#gadgets', '#innovation']
    }
    
    for concept in concepts:
        hashtags.append(f"#{concept}")
        if concept in concept_hashtags:
            hashtags.extend(concept_hashtags[concept])
    
    general_hashtags = [
        '#photography', '#photooftheday', '#picoftheday', 
        '#instagood', '#beautiful', '#capture', '#moment'
    ]
    hashtags.extend(general_hashtags)
    
    return list(set(hashtags))
def main():
    st.markdown('<h1 class="main-header">✨ AI Caption & Hashtag Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Transform your images into engaging captions and trending hashtags!</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Upload Your Image</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Your Image', use_column_width=True)
    
    with col2:
        if uploaded_file:
            with st.spinner("✨ AI is working its magic..."):
                # Load models (cached)
                clip_processor, clip_model, tokenizer, model = load_models()
                
                # Generate content
                concepts = get_image_features(image, clip_processor, clip_model)
                caption = generate_description(concepts, tokenizer, model)
                hashtags = generate_hashtags(caption, concepts)
                hashtags_text = " ".join(hashtags)
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["📝 Caption", "#️⃣ Hashtags", "🎯 Concepts"])
                
                with tab1:
                    st.markdown('<h3>Generated Caption</h3>', unsafe_allow_html=True)
                    st.markdown(f'<div class="output-text">{caption}</div>', unsafe_allow_html=True)
                    if st.button("📋 Copy Caption", key="copy_caption"):
                        st.toast("Caption copied to clipboard!")
                
                with tab2:
                    st.markdown('<h3>Generated Hashtags</h3>', unsafe_allow_html=True)
                    st.markdown(f'<div class="hashtag-text">{hashtags_text}</div>', unsafe_allow_html=True)
                    if st.button("📋 Copy Hashtags", key="copy_hashtags"):
                        st.toast("Hashtags copied to clipboard!")
                
                with tab3:
                    st.markdown('<h3>Detected Concepts</h3>', unsafe_allow_html=True)
                    for concept in concepts:
                        st.markdown(f'<div class="output-text">🏷️ {concept.title()}</div>', unsafe_allow_html=True)
                
                # Export options
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "⬇️ Download Caption",
                        caption,
                        file_name="caption.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        "⬇️ Download Hashtags",
                        hashtags_text,
                        file_name="hashtags.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        else:
            st.markdown('<p class="info-text">👈 Upload an image to get started!</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
