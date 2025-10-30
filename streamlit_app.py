"""
INVINCIX Chatbot - Simple Streamlit Interface
No RAG - Just fine-tuned model
"""

import streamlit as st
from llama_cpp import Llama
import time

# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="INVINCIX Chatbot v1",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS FOR INVINCIX BRANDING
# ================================================================

st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .stTitle {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Input box */
    .stChatInputContainer {
        border-radius: 25px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    /* Custom banner */
    .banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .banner h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    
    .banner p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        font-style: italic;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ================================================================
# LOAD MODEL (CACHED)
# ================================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path="Llama-3.2-1B.Q4_K_M.gguf"):
    """Load the fine-tuned INVINCIX model"""
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,          # Context window
            n_threads=4,          # CPU threads
            n_gpu_layers=0,       # Set to 0 for CPU-only
            verbose=False
        )
        return llm, None
    except Exception as e:
        return None, str(e)

# ================================================================
# GENERATE RESPONSE
# ================================================================

def generate_response(llm, user_message, temperature=0.7, max_tokens=300):
    """Generate chatbot response"""
    
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request as an INVINCIX chatbot assistant.

### Instruction:
You are a helpful and empathetic assistant for INVINCIX, a software engineering and product development company. Respond to the customer's inquiry with warmth, professionalism, and accurate information about INVINCIX's services, products, and values. Keep your response concise and engaging.

### Input:
{user_message}

### Response:
"""
    
    try:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["###", "\n\n\n\n", "### Input:", "### Instruction:"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        
        # Clean up any remaining formatting
        if "### Response:" in answer:
            answer = answer.split("### Response:")[-1].strip()
        
        return answer
    
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."

# ================================================================
# MAIN APP
# ================================================================


    # Banner
st.markdown("""
        <div class="banner">
            <h1>💬 INVINCIX Chatbot</h1>
            <p>"Simplicity is our culture and simplification is what we do"</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ================================================================
    # SIDEBAR
    # ================================================================
    
with st.sidebar:
        st.image("https://via.placeholder.com/300x80/667eea/ffffff?text=INVINCIX", use_column_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 About INVINCIX")
        st.markdown("""
        **Mission**: Digitize 500+ startups
        
        **Core Values**:
        - 🤝 Engage
        - 💡 Innovate  
        - 🚀 Invent
        - ⭐ Excel
        """)
        
        st.markdown("---")
        
        st.markdown("### 🛠️ Our Services")
        st.markdown("""
        - Software Development
        - Product Engineering
        - Mobile Apps (iOS/Android)
        - Data Analytics
        - Cloud Solutions
        - IoT & Drone Solutions
        """)
        
        st.markdown("---")
        
        st.markdown("### 📦 Our Products")
        st.markdown("""
        - **Xprodedge**: Product innovation
        - **DigiStack**: Service automation
        - **Zikshaa**: Mentorship platform
        - **AD4P**: Application development
        """)
        
        st.markdown("---")
        
        st.markdown("### 📞 Contact Us")
        st.markdown("""
        📧 info@invincix.com  
        📱 +91 674 297 2316  
        🌐 [invincix.com](https://invincix.com)
        """)
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        temperature = st.slider(
            "Response Creativity",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more consistent, Higher = more creative"
        )
        
        max_tokens = st.slider(
            "Response Length",
            min_value=100,
            max_value=500,
            value=300,
            step=50,
            help="Maximum words in response"
        )
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Model info
        with st.expander("ℹ️ Model Information"):
            st.markdown("""
            **Model**: Llama 3.2 3B  
            **Fine-tuned**: INVINCIX specific  
            **Running on**: CPU  
            **Status**: ✅ Active
            """)
    
    # ================================================================
    # LOAD MODEL
    # ================================================================
    
with st.spinner("🔄 Loading INVINCIX chatbot model..."):
        llm, error = load_model()
    
if error:
        st.error(f"❌ **Error loading model**: {error}")
        st.markdown("""
        ### Troubleshooting:
        1. Make sure `invincix_model-Q4_K_M.gguf` is in the same directory as this script
        2. Install required package: `pip install llama-cpp-python`
        3. Check if the model file is corrupted
        
        **Model Path Expected**: `./invincix_model-Q4_K_M.gguf`
        """)
        st.stop()
    
st.success("✅ Chatbot loaded successfully!")
    
    # ================================================================
    # CHAT INTERFACE
    # ================================================================
    
    # Initialize chat history
if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm the INVINCIX chatbot. 👋\n\nI'm here to help you learn about our services, products, and how we can support your business transformation journey. Whether you're a startup founder or an enterprise leader, I'd love to assist you!\n\nWhat would you like to know?"
            }
        ]
    
    # Display chat messages
for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
if prompt := st.chat_input("Ask me anything about INVINCIX..."):
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Generate response
                response = generate_response(
                    llm, 
                    prompt, 
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Simulate typing effect (optional - remove if you want instant responses)
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
                
                message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # ================================================================
    # FOOTER
    # ================================================================
    
st.markdown("---")
st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
            <p>💡 <strong>Tip</strong>: Try asking about our services, products, company culture, or how we can help your business!</p>
            <p style='font-size: 0.9em; margin-top: 10px;'>
                Powered by fine-tuned AI | Built for INVINCIX | 
                <em>"We keep our feet grounded to ensure your head is in the cloud"</em>
            </p>
        </div>
    """, unsafe_allow_html=True)

# ================================================================
# RUN APP
# ================================================================

