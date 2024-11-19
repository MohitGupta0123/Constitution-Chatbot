import streamlit as st
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex, SimpleKeywordTableIndex, get_response_synthesizer
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import os
import json
from dotenv import load_dotenv
import re
import nltk
from nltk.corpus import stopwords
import tempfile
from gtts import gTTS
from datetime import datetime
from streamlit_mic_recorder import speech_to_text

# Download stopwords if not already downloaded
nltk.download('stopwords')

st.set_page_config(page_title="L.A.W.S. Chatbot", page_icon="💬", layout="centered")

st.title("💼 L.A.W.S. - Legal Assistance and Wisdom System Chatbot")
st.write("Ask me questions about your PDF!")
st.markdown("<hr>", unsafe_allow_html=True)

# Initialize conversation history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to clean up temporary audio data
def clean_up_audio_data():
    if 'audio_data' in st.session_state:
        st.session_state['audio_data'] = []

# Restart Chat button
if st.sidebar.button("🔄 New Chat"):
    st.session_state['messages'] = []
    clean_up_audio_data()

# Preprocess text for indexing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text.strip()

# Function to load documents and initialize query engine
def initialize_query_engine():
    if 'documents' not in st.session_state:
        documents = SimpleDirectoryReader('data').load_data()
        for doc in documents:
            doc.text = preprocess_text(doc.text)
        st.session_state['documents'] = documents
    else:
        documents = st.session_state['documents']

    if 'nodes' not in st.session_state:
        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        st.session_state['nodes'] = nodes
    else:
        nodes = st.session_state['nodes']

    # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

    # Initialize embeddings and LLM
    if 'embed_model' not in st.session_state:
        st.session_state['embed_model'] = GeminiEmbedding(model_name="models/embedding-001", api_key=GEMINI_API_KEY)
        Settings.embed_model = st.session_state['embed_model']

    if 'llm' not in st.session_state:
        st.session_state['llm'] = Gemini(api_key=GEMINI_API_KEY)
        Settings.llm = st.session_state['llm']

    # Initialize storage context and create indexes
    if 'storage_context' not in st.session_state:
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        st.session_state['storage_context'] = storage_context
    else:
        storage_context = st.session_state['storage_context']

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index, top_k=5)

    # Custom retriever to combine vector and keyword retrievers
    class CustomRetriever(BaseRetriever):
        def __init__(self, vector_retriever, keyword_retriever, mode="AND"):
            self._vector_retriever = vector_retriever
            self._keyword_retriever = keyword_retriever
            self._mode = mode if mode in ("AND", "OR") else "AND"
            super().__init__()

        def _retrieve(self, query_bundle):
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
            vector_ids = {n.node.node_id for n in vector_nodes}
            keyword_ids = {n.node.node_id for n in keyword_nodes}

            combined_dict = {n.node.node_id: n for n in vector_nodes}
            combined_dict.update({n.node.node_id: n for n in keyword_nodes})

            if self._mode == "AND":
                retrieve_ids = vector_ids.intersection(keyword_ids)
            else:
                retrieve_ids = vector_ids.union(keyword_ids)

            return [combined_dict[r_id] for r_id in retrieve_ids]

    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever, "OR")
    response_synthesizer = get_response_synthesizer()

    return RetrieverQueryEngine(retriever=custom_retriever, response_synthesizer=response_synthesizer)

# Transcribe audio using streamlit_mic_recorder
def transcribe_audio():
    transcription = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
    return transcription

# Text-to-speech output using gTTS
@st.cache_resource
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_file = fp.name
            tts.save(temp_file)
        return temp_file
    except Exception as e:
        st.error(f"Error in TTS: {e}")
        return None

# Load data and initialize query engine
if 'custom_query_engine' not in st.session_state:
    with st.spinner("Loading the data, please wait..."):
        st.session_state['custom_query_engine'] = initialize_query_engine()
    st.success("Ready! You can now ask questions.")

# Select input method
if 'input_method_selected' not in st.session_state:
    st.session_state['input_method_selected'] = None

st.sidebar.markdown("## Input Method")
input_method = st.sidebar.selectbox("Choose input method:", ("Text", "Voice"), key="input_method")

# Save Conversation
if st.sidebar.button("Save Conversation"):
    # Convert messages to JSON string
    messages_json = json.dumps(st.session_state['messages'], indent=2)
    # Provide a download button
    st.sidebar.download_button(
        label="Download Conversation",
        data=messages_json,
        file_name='conversation.json',
        mime='application/json'
    )

# Load Conversation
uploaded_file = st.sidebar.file_uploader("Load Conversation", type="json")
if uploaded_file is not None:
    # Read the uploaded JSON file
    messages_json = uploaded_file.read()
    # Update the session state with loaded messages
    st.session_state['messages'] = json.loads(messages_json)
    st.success("Conversation loaded successfully!")

# Query input and processing
query = None
if input_method == "Text":
    query = st.chat_input("Enter your question:")
    if query:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Append user message to messages
        st.session_state['messages'].append({"role": "user", "content": query, "timestamp": timestamp})
elif input_method == "Voice":
    st.write(" Click the Button to Record your voice")

    with st.spinner("Transcribing audio..."):
        query = transcribe_audio()

    st.write(f"Transcribed Text: {query}")

    if query:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state['messages'].append({"role": "user", "content": query, "timestamp": timestamp})

# Process the query and get the response
if query:
    with st.spinner("Searching for an answer..."):
        custom_query_engine = st.session_state['custom_query_engine']
        result = custom_query_engine.query(query)
        response = result.response
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if response != 'Empty Response':
            # Add assistant's response to the messages
            st.snow()
            st.balloons()
            assistant_message = {
                "role": "assistant",
                "content": "Response Generated",
                "message_type": "success",
                "timestamp": timestamp
            }
            st.session_state['messages'].append(assistant_message)
        else:
            response = """You are asking OUT OF CONTEXT INFORMATION. That is not available to me. 
                        \n Try something else 😊\n"""
            assistant_message = {
                "role": "assistant",
                "content": "No Context Found",
                "message_type": "warning",
                "timestamp": timestamp
            }
            st.session_state['messages'].append(assistant_message)
        # Add assistant's response to the messages
        assistant_message = {
            "role": "assistant",
            "content": response,
            "timestamp": timestamp  # Ensure timestamp is added here
        }
        st.session_state['messages'].append(assistant_message)

        # Automatically generate TTS if input method is Voice
        if input_method == "Voice":
            audio_file = text_to_speech(response)
            if audio_file:
                assistant_message["audio"] = audio_file

# Now, display the conversation history after processing the new messages
for message in st.session_state['messages']:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**{message['content']}**  \n*{message['timestamp']}*")
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            if message.get("message_type") == "success":
                st.success(f"{message['content']}")
            elif message.get("message_type") == "warning":
                st.warning(f"{message['content']}")
            else:
                st.markdown(f"{message['content']}  \n*{message['timestamp']}*")
            # If there is an audio file, display the audio player
            if "audio" in message:
                with open(message["audio"], "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format='audio/mp3', start_time=0)