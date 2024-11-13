# Constitutional Chatbot

This application is a chatbot framework that enables users to interact with a specified document or set of documents through both text and voice inputs. It leverages semantic embeddings and large language models (LLM) to provide contextually accurate answers, Whisper for speech-to-text (STT) conversion, and pyttsx3 for text-to-speech (TTS) output. Built with Streamlit, this chatbot allows users to ask questions and receive informative responses from any uploaded document.

## Features

- **Text and Voice Query**: Allows users to query documents via text or voice input.
- **Voice Response**: Reads out responses for a fully voice-based interaction experience.
- **Semantic Understanding**: Utilizes embeddings for accurate, context-sensitive answers.
- **Speech Recognition**: Converts spoken queries into text using Whisper’s STT model.
- **Response Synthesis**: Combines vector-based and keyword-based retrieval methods for optimized answers.

## Requirements

- Python 3.8+
- Required Python packages:
  - `streamlit`
  - `llama_index`
  - `nltk`
  - `pyaudio`
  - `pyttsx3`
  - `whisper`
  - `gtts`
  - `numpy`
  - `dotenv`

To install dependencies, use:
```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:
    ```bash
    git clone <repo-url>
    cd <repo-directory>
    ```

2. Add your API key (if needed) in a `.streamlit/secrets.toml` file:
    ```plaintext
    GEMINI_API_KEY=your_api_key
    ```

4. Prepare audio hardware for voice input.

## Usage

1. **Run the Application**:
   ```bash
   python -m streamlit run chatbot.py
   ```

2. **Using the Interface**:
   - Upload document(s) to be queried (default directory: `data/`).
   - Select text or voice as the input method.
   - For text input, type your question.
   - For voice input, click "Record Question," speak your question, and wait for the transcription.

3. **Query Results**:
   - Responses are displayed on the screen.
   - For voice input, responses are also read aloud.

## File Structure

- **`chatbot.py`**: Main application file, initializes components, handles both text and voice queries.
- **`data/`**: Directory for storing documents to be queried.
- **`.env`**: Stores sensitive data, such as API keys.

## Code Overview

- **Query Engine**: Uses embeddings for LLM and vector similarity, with `VectorStoreIndex` and `SimpleKeywordTableIndex` for document retrieval.
- **Custom Retriever**: Combines vector-based and keyword-based retrieval methods for comprehensive results.
- **Voice Interface**:
  - Whisper is used for audio transcription.
  - `gtts` provides TTS capabilities to read responses aloud.

## Future Enhancements

- **Multilingual Support**: Integrate support for multilingual transcription and response translation.
- **Advanced Document Preprocessing**: Enable document pre-filtering and enrichment for more specialized document sets.
- **Add Image/Video Explaining Functionality**: Enabling asking questions from a user uploaded image or video.

