

import os
import gc
import uuid
import tempfile
import base64
from dotenv import load_dotenv
from rag_code import Transcribe, EmbedData, QdrantVDB_QB, Retriever, RAG, extract_audio_from_video
import streamlit as st
import time
import subprocess
import shutil


if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
collection_name = f"chat_with_media_{str(session_id)[:8]}"  # Make collection unique per session
batch_size = 32

load_dotenv()

def clear_qdrant_docker_storage():
    try:
        # Step 1: Get the most recent container running Qdrant
        result = subprocess.run(
            ["docker", "ps", "--filter", "ancestor=qdrant/qdrant", "--format", "{{.Names}}"],
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
        container_name = result.stdout.strip().splitlines()[0]  # Take the first match
        print(f"Detected Qdrant container: {container_name}")

        # Step 2: Try removing files using bash, fallback to sh
        try:
            subprocess.run(
                ["docker", "exec", container_name, "bash", "-c", "rm -rf /qdrant/storage/*"],
                check=True
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["docker", "exec", container_name, "sh", "-c", "rm -rf /qdrant/storage/*"],
                check=True
            )

        # Step 3: Restart the container
        subprocess.run(["docker", "restart", container_name], check=True)
        st.success(f"✅ Cleared Qdrant storage and restarted container: `{container_name}`")

    except subprocess.CalledProcessError as e:
        st.error(f"❌ Failed to clear Qdrant storage: {e}")
    except IndexError:
        st.error("❌ No running Qdrant container found.")
    except FileNotFoundError:
        st.error("❌ Docker not found. Is it installed and running?")



def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

with st.sidebar:
    st.header("Add your media file!")

    # --- Storage monitor + cleanup ---
    st.markdown("### 🧹 Qdrant Storage Control")

    # Show available disk space
    total, used, free = shutil.disk_usage("/")
    st.write(f"💾 Free space: {free // (2**30)} GB / {total // (2**30)} GB")

    # Add cleanup button
    with st.expander("⚠️ Dangerous Actions: Qdrant Cleanup", expanded=False):
        st.markdown("This will permanently delete all vector data stored in Qdrant and restart the container.")
    
        confirm_clear = st.checkbox("I understand the risk and want to continue.")
    
        if st.button("🧹 Clear Qdrant Storage"):
            if confirm_clear:
                clear_qdrant_docker_storage()
            else:
                st.warning("Please confirm by checking the box before proceeding.")


    
    # Check for API key
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        st.error("⚠️ AssemblyAI API key not found in .env file. Please add it as ASSEMBLYAI_API_KEY=your_key")
    
    # Tabs for audio and video
    media_type = st.radio("Select media type:", ["Audio", "Video"])
    
    if media_type == "Audio":
        uploaded_file = st.file_uploader("Choose your audio file", type=["mp3", "wav", "m4a"])
        is_video = False
    else:  # Video
        uploaded_file = st.file_uploader("Choose your video file", type=["mp4", "mov", "avi", "mkv"])
        is_video = True

    if uploaded_file:
        # Process file
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                status_text.text("Saving uploaded file...")
                progress_bar.progress(10)
                
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                
                # For video, we need to extract audio first
                if is_video:
                    status_text.text("Extracting audio from video...")
                    progress_bar.progress(20)
                    try:
                        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
                        extract_audio_from_video(file_path, audio_path)
        
                        # Verify the audio file exists and is not empty
                        if not os.path.exists(audio_path):
                            st.error("⚠️ Failed to extract audio from video: Output file not created")
                            progress_bar.empty()
                            status_text.empty()
                            st.stop()
            
                        if os.path.getsize(audio_path) == 0:
                            st.error("⚠️ Extracted audio file is empty. The video might not contain audio.")
                            progress_bar.empty()
                            status_text.empty()
                            st.stop()
            
                        processing_path = audio_path
                    except Exception as e:
                        st.error(f"⚠️ Failed to extract audio from video: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()
                        st.stop()
                else:
                    processing_path = file_path
                
                status_text.text("Transcribing with AssemblyAI...")
                progress_bar.progress(30)

                if file_key not in st.session_state.get('file_cache', {}):
                    try:
                        # Verify API key
                        if not api_key:
                            raise ValueError("AssemblyAI API key not found. Please check your .env file.")
                        
                        # Initialize transcriber
                        transcriber = Transcribe(api_key=api_key)
                        
                        # Get speaker-labeled transcripts
                        transcripts = transcriber.transcribe_audio(processing_path)
                        if not transcripts:
                            st.error("⚠️ Transcription failed or returned empty. Please check the audio quality or API key.")
                            progress_bar.empty()
                            status_text.empty()
                            st.stop()

                        # Ensure transcripts is a valid list before continuing
                        if not isinstance(transcripts, list):
                            st.error("⚠️ Transcription API did not return expected format. Please check the audio quality.")
                            progress_bar.empty()
                            status_text.empty()
                            st.stop()
                        
                        progress_bar.progress(50)
                        status_text.text("Embedding transcript data...")
                        
                        st.session_state.transcripts = transcripts
                        
                        # Each speaker segment becomes a separate document for embedding
                        documents = [f"Speaker {t['speaker']}: {t['text']}" for t in transcripts]

                        if not documents:
                            st.error("⚠️ No transcript segments found to process.")
                            st.stop()

                        # embed data    
                        embeddata = EmbedData(embed_model_name="BAAI/bge-large-en-v1.5", batch_size=batch_size)
                        embeddata.embed(documents)
                        
                        if not embeddata.embeddings or len(embeddata.embeddings) == 0:
                            st.error("⚠️ Failed to generate embeddings from transcript.")
                            st.stop()
                            
                        progress_bar.progress(70)
                        status_text.text("Setting up vector database...")

                        # set up vector database
                        qdrant_vdb = QdrantVDB_QB(collection_name=collection_name,
                                              batch_size=batch_size,
                                              vector_dim=1024)
                        try:
                            qdrant_vdb.define_client()
                            qdrant_vdb.create_collection()
                            
                            progress_bar.progress(80)
                            status_text.text("Storing data in vector database...")
                            
                            qdrant_vdb.ingest_data(embeddata=embeddata)
                            
                            progress_bar.progress(90)
                            status_text.text("Setting up retriever and query engine...")
                            
                            # set up retriever
                            retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)

                            # set up rag
                            try:
                                query_engine = RAG(retriever=retriever, llm_name="DeepSeek-R1-Distill-Llama-70B")
                                st.session_state.file_cache[file_key] = query_engine
                                st.session_state.current_media_path = file_path
                                st.session_state.current_media_type = "video" if is_video else "audio"
                            except Exception as e:
                                st.error(f"⚠️ Error setting up query engine: {e}")
                                st.stop()
                                
                        except ConnectionError:
                            st.error("⚠️ Failed to connect to Qdrant database.")
                            st.info("Make sure Qdrant is running with: docker run -p 6333:6333 -p 6334:6334 -v \"%cd%\\qdrant_storage:/qdrant/storage\" qdrant/qdrant")
                            st.stop()
                        except Exception as e:
                            st.error(f"⚠️ Qdrant database error: {e}")
                            st.stop()
                            
                    except Exception as e:
                        st.error(f"⚠️ Error processing file: {e}")
                        progress_bar.progress(0)
                        status_text.empty()
                        st.stop()
                else:
                    query_engine = st.session_state.file_cache[file_key]
                    st.session_state.current_media_path = file_path
                    st.session_state.current_media_type = "video" if is_video else "audio"

                # Complete the progress bar
                progress_bar.progress(100)
                time.sleep(0.5)  # Small delay for visual feedback
                progress_bar.empty()
                status_text.empty()
                
                # Inform the user that the file is processed
                st.success("✅ Ready to Chat!")
                
                # Display appropriate media player
                if is_video:
                    st.video(uploaded_file)
                else:
                    st.audio(uploaded_file)
                
                # Display speaker-labeled transcript
                st.subheader("Transcript")
                with st.expander("Show full transcript", expanded=True):
                    if hasattr(st.session_state, 'transcripts') and st.session_state.transcripts:
                        for t in st.session_state.transcripts:
                             st.text(f"Speaker {t['speaker']}: {t['text']}")
                    else:
                        st.warning("No transcript was returned. Please check the audio quality or API response.")

                
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
            progress_bar.empty()
            status_text.empty()
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("""
    # RAG over Audio & Video 
    Upload audio or video files and chat with your media content using AI.
    """)

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about the conversation in the media..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        file_processed = False
        if uploaded_file:
            file_key = f"{session_id}-{uploaded_file.name}"
            query_engine = st.session_state.file_cache.get(file_key)
            if query_engine:
                file_processed = True
                try:
                    streaming_response = query_engine.query(prompt)
                    
                    for chunk in streaming_response:
                        try:
                            # Try different ways to extract text from the response
                            if hasattr(chunk, 'raw') and 'choices' in chunk.raw:
                                try:
                                    new_text = chunk.raw["choices"][0]["delta"]["content"]
                                    full_response += new_text
                                    message_placeholder.markdown(full_response + "▌")
                                except (KeyError, IndexError):
                                    pass
                            elif hasattr(chunk, 'text'):
                                new_text = chunk.text
                                full_response += new_text
                                message_placeholder.markdown(full_response + "▌")
                            elif isinstance(chunk, str):
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                        except Exception as e:
                            st.warning(f"Warning: Error processing chunk: {e}")
                            continue
                            
                    if not full_response:
                        full_response = "I couldn't generate a response based on the transcript. Please try a different question."
                        
                except Exception as e:
                    full_response = f"Error processing your query: {str(e)}"
                    message_placeholder.markdown(full_response)
        
        if not file_processed:
            full_response = "Please upload a media file first to ask questions about it."
            message_placeholder.markdown(full_response)
        else:
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})