import os
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import heapq
from groq import Groq
import pandas as pd
from requests.exceptions import HTTPError, RequestException
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import asyncio
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Load environment variables at the top of the file
load_dotenv()

# Configure Streamlit to avoid some torch-related issues
if hasattr(torch, 'set_num_threads'):
    torch.set_num_threads(1)

def retrieve(query, data, topic_embeddings, content_embeddings, model, top_n=5):
    """
    FIXED: Retrieves top-N relevant results with similarity scores.
    
    Returns:
    - results (list): List of (similarity_score, data_item) tuples
    - query_embedding (torch.Tensor): The embedding of the query (on CPU)
    """
    try:
        # Ensure all embeddings are on CPU and handle device compatibility
        if isinstance(topic_embeddings, torch.Tensor):
            topic_embeddings = topic_embeddings.cpu()
        else:
            topic_embeddings = torch.tensor(topic_embeddings, dtype=torch.float32).cpu()
            
        if isinstance(content_embeddings, torch.Tensor):
            content_embeddings = content_embeddings.cpu()
        else:
            content_embeddings = torch.tensor(content_embeddings, dtype=torch.float32).cpu()

        # Generate query embedding and move to CPU
        with torch.no_grad():  # Save memory and avoid gradient computation
            query_embedding = model.encode(query, convert_to_tensor=True, device='cpu')
            if hasattr(query_embedding, 'cpu'):
                query_embedding = query_embedding.cpu()

        # Compute cosine similarities
        topic_sim = util.cos_sim(query_embedding, topic_embeddings)[0]
        content_sim = util.cos_sim(query_embedding, content_embeddings)[0]

        # Weighted similarity
        combined_similarity = 0.3 * topic_sim + 0.7 * content_sim

        # Get top N indices with scores
        top_k_result = torch.topk(combined_similarity, k=min(top_n, len(data)))
        top_scores = top_k_result.values.tolist()
        top_indices = top_k_result.indices.tolist()

        # Return tuples of (similarity, data_item)
        results = [(top_scores[i], data[top_indices[i]]) for i in range(len(top_indices))]

        return results, query_embedding
        
    except Exception as e:
        st.error(f"Error in retrieve function: {str(e)}")
        return [], None


def is_context_relevant(query, result, query_embedding, topic_embedding, content_embedding, threshold=0.4):
    """
    FIXED: Added error handling for relevance checking with better tensor handling
    """
    try:
        # Ensure CPU placement for tensors and handle different input types
        if hasattr(query_embedding, 'cpu'):
            query_embedding = query_embedding.cpu()
        
        if isinstance(topic_embedding, (list, np.ndarray)):
            topic_tensor = torch.tensor(topic_embedding, dtype=torch.float32).cpu()
        else:
            topic_tensor = topic_embedding.cpu() if hasattr(topic_embedding, 'cpu') else topic_embedding
            
        if isinstance(content_embedding, (list, np.ndarray)):
            content_tensor = torch.tensor(content_embedding, dtype=torch.float32).cpu()
        else:
            content_tensor = content_embedding.cpu() if hasattr(content_embedding, 'cpu') else content_embedding

        # Calculate similarities with no_grad for efficiency
        with torch.no_grad():
            topic_similarity = util.cos_sim(query_embedding, topic_tensor).item()
            content_similarity = util.cos_sim(query_embedding, content_tensor).item()

        # Combined similarity with higher weight on content
        combined_similarity = (0.3 * topic_similarity) + (0.7 * content_similarity)

        return combined_similarity > threshold
    except Exception as e:
        st.warning(f"Error checking context relevance: {str(e)}")
        return False


@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model with caching to prevent reloading"""
    try:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        return model
    except Exception as e:
        st.error(f"Failed to load sentence transformer: {str(e)}")
        return None


@st.cache_data
def load_data_json():
    """Load JSON data with caching"""
    try:
        with open('data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data.json: {str(e)}")
        return None


def load_or_generate_embeddings(data, embed_model, cache_file='embeddings_cache.npz'):
    """
    FIXED: Better error handling and validation with memory efficiency
    """
    # Check if cache file exists
    if os.path.exists(cache_file):
        try:
            cache = np.load(cache_file, allow_pickle=True)
            print("Loaded embeddings from cache.")
            cached_topic_embeddings = cache['topic_embeddings']
            cached_content_embeddings = cache['content_embeddings']
            
            # Check whether the length of the Cache is same as length of the data
            if len(cached_topic_embeddings) == len(data):
                print("Using cached embeddings...")
                return cached_topic_embeddings, cached_content_embeddings
            else:
                print("Cache size mismatch. Regenerating embeddings...")
        except Exception as e:
            print(f"Error loading cache: {e}")
            # Continue to generate new embeddings
    
    # Generate embeddings
    print("Generating new embeddings...")
    try:
        # Extract topic and content with validation
        topics = []
        contents = []
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Data item {i} is not a dictionary: {type(item)}")
            if 'topic' not in item:
                raise KeyError(f"Data item {i} missing 'topic' key")
            if 'content' not in item:
                raise KeyError(f"Data item {i} missing 'content' key")
            
            topics.append(str(item['topic']))  # Ensure string
            contents.append(str(item['content']))  # Ensure string

        # Generate embeddings with batch processing for efficiency
        with torch.no_grad():
            topic_embeddings = embed_model.encode(
                topics, 
                convert_to_tensor=False,
                device='cpu',
                show_progress_bar=True
            )
            content_embeddings = embed_model.encode(
                contents, 
                convert_to_tensor=False,
                device='cpu',
                show_progress_bar=True
            )

        # Save embeddings to cache
        np.savez_compressed(cache_file, 
                          topic_embeddings=topic_embeddings, 
                          content_embeddings=content_embeddings)
        print("Generated new embeddings and saved to cache.")

        return topic_embeddings, content_embeddings
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


def get_groq_client():
    """
    FIXED: Better API key handling
    """
    api_key = (
        os.getenv("GROQ_API_KEY") or 
        os.getenv("groq_key") or 
        os.getenv("GROQ_KEY")
    )
    
    if not api_key or api_key.strip() == "":
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        st.info("Please set your Groq API key in one of these ways:")
        st.code("""
# Method 1: In .env file
GROQ_API_KEY=your_api_key_here

# Method 2: Environment variable
export GROQ_API_KEY=your_api_key_here
        """)
        st.stop()

    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {str(e)}")
        st.stop()


def prepare_conversation_history(messages, max_tokens=4000):
    """
    FIXED: Better token estimation and error handling
    """
    try:
        history = []
        current_tokens = 0

        # Iterate in reverse to keep the most recent messages in context
        for msg in reversed(messages):
            if not isinstance(msg, dict) or 'content' not in msg or 'role' not in msg:
                continue
                
            # Estimate token count (rough approximation: ~4 chars per token)
            msg_tokens = len(str(msg['content'])) // 4

            if current_tokens + msg_tokens > max_tokens:
                break

            # Prepend to maintain chronological order
            history.insert(0, {"role": msg['role'], "content": str(msg['content'])})
            current_tokens += msg_tokens

        return history
    except Exception as e:
        st.warning(f"Error preparing conversation history: {str(e)}")
        return []


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "groq_model" not in st.session_state:
        st.session_state.groq_model = 'llama3-70b-8192'


def main():
    # Configure page
    st.set_page_config(
        page_title="Hydraulic Engineering Assistant",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the session variables
    initialize_session_state()
    
    st.title("🌊 Hydraulic Engineering Assistant")
    # st.markdown("*AI Teaching Assistant for Prof. Saud Afzal's course at IIT Kharagpur*")
    
    # Load the Embedding Model with caching
    model = load_sentence_transformer()
    if model is None:
        st.stop()
    
    # Load the JSON file with caching
    data = load_data_json()
    if data is None or not data:
        st.error("❌ Failed to load or empty data.json!")
        st.stop()
    
    # Validate data structure
    if not isinstance(data, list):
        st.error("❌ data.json should contain a list of items!")
        st.stop()
        
    st.success(f"✅ Loaded {len(data)} items from data.json")
    
    # Generate/Load Embeddings
    try:
        with st.spinner("Loading/generating embeddings..."):
            topic_embeddings, content_embeddings = load_or_generate_embeddings(data, model)
        st.success("✅ Embeddings ready!")
    except Exception as e:
        st.error(f"❌ Error with embeddings: {str(e)}")
        st.stop()
        
    # Initialize Groq client with proper error handling
    client = get_groq_client()
    
    # Sidebar for options
    st.sidebar.title("⚙️ Options")

    # Dropdown for model selection
    model_options = [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192", 
        "gemma-7b-it",
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]
    
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    st.session_state.groq_model = selected_model
    st.sidebar.write(f"Current Model: **{st.session_state.groq_model}**")
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 100)
        top_n_results = st.slider("Retrieval Results", 3, 10, 5, 1)
    
    # Clear chat history button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.sidebar.success("Chat history cleared.")
        st.experimental_rerun()
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # User prompt handling
    if prompt := st.chat_input("Ask a Hydraulic Engineering question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            # Retrieve relevant topics and content
            retrieved_results, query_embedding = retrieve(
                prompt, data, topic_embeddings, content_embeddings, model, top_n=top_n_results
            )

            if not retrieved_results or query_embedding is None:
                st.error("❌ Failed to retrieve relevant content")
                st.stop()

            # Filter and prepare relevant contexts
            relevant_contexts = []
            for similarity, result in retrieved_results:
                try:
                    # Find the index of this result in the original data
                    result_index = next(i for i, item in enumerate(data) if item == result)
                    
                    if is_context_relevant(
                            prompt, result,
                            query_embedding,
                            topic_embeddings[result_index],
                            content_embeddings[result_index]
                    ):
                        relevant_contexts.append((similarity, result))

                    # Stop after finding enough relevant contexts
                    if len(relevant_contexts) >= 3:
                        break
                except (StopIteration, IndexError) as e:
                    continue

            # Prepare system prompt
            if relevant_contexts:
                enriched_context = "Relevant sections from the course material:\n\n"
                for i, (similarity, result) in enumerate(relevant_contexts, 1):
                    enriched_context += (
                        f"**Section {i}:**\n"
                        f"**Topic:** {result.get('topic', 'N/A')}\n"
                        f"**Content:** {result.get('content', 'N/A')}\n"
                        f"**Relevance Score:** {similarity:.3f}\n\n"
                    )
                enriched_context += "Based on these relevant sections, please provide a comprehensive answer:"
                use_context = True
            else:
                enriched_context = "No highly relevant sections were found in the course material. Please provide a general response based on hydraulic engineering principles."
                use_context = False

            # Prepare system prompt with conditional context
            system_prompt = (
                f"You are an AI Teaching Assistant for the Hydraulic Engineering course taught by Prof. Saud Afzal at IIT Kharagpur. "
                f"Provide clear, concise, and educational responses that help students understand complex concepts.\n\n"
                f"Guidelines:\n"
                f"- Use proper LaTeX formatting for mathematical equations\n"
                f"- Explain concepts step by step when appropriate\n"
                f"- Provide practical examples when helpful\n"
                f"- Reference relevant formulas and principles\n\n"
                f"{enriched_context}"
            )

            # Generate response from Groq
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                try:
                    # Prepare conversation history
                    conversation_history = prepare_conversation_history(
                        st.session_state.messages[:-1]  # Exclude the current message
                    )

                    # Prepare messages with history, system prompt, and current query
                    messages = [
                        {"role": "system", "content": system_prompt}
                    ] + conversation_history + [
                        {"role": "user", "content": prompt}
                    ]

                    # Get response from Groq with streaming
                    stream = client.chat.completions.create(
                        model=st.session_state.groq_model,
                        messages=messages,
                        stream=True,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")

                    # Finalize response
                    response_placeholder.markdown(full_response)
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    response_placeholder.markdown(error_msg)
                    full_response = error_msg

                # Add assistant response to the chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ Unexpected error in main processing: {str(e)}")
            # Optionally show traceback in development
            # st.exception(e)


# Run the Streamlit app
if __name__ == "__main__":
    main()