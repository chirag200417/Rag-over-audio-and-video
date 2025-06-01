from qdrant_client import models
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.sambanovasystems import SambaNovaCloud
from llama_index.llms.ollama import Ollama
import assemblyai as aai
from typing import List, Dict
from sentence_transformers import CrossEncoder
import subprocess
import os

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    if lst is None:
        return []  # Return empty list if None is passed
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to the video file
        output_audio_path: Path to save the extracted audio
        
    Returns:
        Path to the extracted audio file
    """
    # Use ffmpeg to extract audio from video
    try:
        print(f"Extracting audio from {video_path} to {output_audio_path}")
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-q:a', '0',
            '-map', 'a',
            '-vn',
            output_audio_path
        ]
        
        # Run the command and capture output
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Verify that the output file was created
        if not os.path.exists(output_audio_path):
            raise Exception(f"Output audio file was not created at {output_audio_path}")
            
        # Get file size to verify it's not empty
        file_size = os.path.getsize(output_audio_path)
        if file_size == 0:
            raise Exception(f"Extracted audio file is empty (0 bytes): {output_audio_path}")
            
        print(f"Successfully extracted audio to {output_audio_path} ({file_size} bytes)")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8') if e.stderr else "No error output available"
        error_msg = f"FFmpeg error: {stderr}"
        print(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        print(f"Failed to extract audio: {str(e)}")
        raise

class EmbedData:

    def __init__(self, embed_model_name="BAAI/bge-large-en-v1.5", batch_size = 32):
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []
        self.contexts = []  # Initialize contexts list
        
    def _load_embed_model(self):
        print(f"Loading embedding model: {self.embed_model_name}")
        try:
            embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, trust_remote_code=True, cache_folder='./hf_cache')
            print("Embedding model loaded successfully")
            return embed_model
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def generate_embedding(self, context):
        if not context:
            print("Warning: Empty context batch passed to generate_embedding")
            return []
        print(f"Generating embeddings for batch of {len(context)} items")
        try:
            embeddings = self.embed_model.get_text_embedding_batch(context)
            print(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
        
    def embed(self, contexts):
        if not contexts:
            print("Error: No contexts provided to embed")
            raise ValueError("No contexts provided to embed method")
            
        print(f"Embedding {len(contexts)} contexts")
        
        # Store the original contexts
        self.contexts = contexts
        
        # Clear previous embeddings if any
        self.embeddings = []
        
        # Process in batches
        batch_count = 0
        for batch_context in batch_iterate(contexts, self.batch_size):
            batch_count += 1
            print(f"Processing batch {batch_count}, size: {len(batch_context)}")
            batch_embeddings = self.generate_embedding(batch_context)
            if batch_embeddings:
                self.embeddings.extend(batch_embeddings)
                print(f"Total embeddings so far: {len(self.embeddings)}")
            else:
                print("Warning: No embeddings generated for this batch")
                
        print(f"Final embedding counts - contexts: {len(self.contexts)}, embeddings: {len(self.embeddings)}")
        
        if len(self.contexts) != len(self.embeddings):
            print("WARNING: Context and embedding counts don't match!")

class QdrantVDB_QB:

    def __init__(self, collection_name, vector_dim = 1024, batch_size=512):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.client = None
        print(f"Initialized QdrantVDB_QB with collection: {collection_name}, vector_dim: {vector_dim}")
        
    def define_client(self):
        # Connect to your already running Qdrant instance
        print("Attempting to connect to Qdrant...")
        try:
            self.client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)
            # Test connection
            collections = self.client.get_collections()
            print(f"Successfully connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
        except Exception as e:
            error_msg = f"Failed to connect to Qdrant: {e}"
            print(error_msg)
            print("Make sure Qdrant is running with: docker run -p 6333:6333 -p 6334:6334 -v \"%cd%\\qdrant_storage:/qdrant/storage\" qdrant/qdrant")
            raise ConnectionError(error_msg)
        
    def create_collection(self):
        if not self.client:
            raise ValueError("Client not initialized. Call define_client() first.")
            
        print(f"Checking if collection '{self.collection_name}' exists...")
        collection_exists = self.client.collection_exists(collection_name=self.collection_name)
        print(f"Collection exists: {collection_exists}")
        
        if not collection_exists:
            print(f"Creating collection '{self.collection_name}'...")
            try:
                self.client.create_collection(
                    collection_name=f"{self.collection_name}",
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.DOT,
                        on_disk=True
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=5,
                        indexing_threshold=0
                    ),
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(always_ram=True)
                    ),
                )
                print(f"Collection '{self.collection_name}' created successfully")
            except Exception as e:
                print(f"Error creating collection: {e}")
                raise
            
    def ingest_data(self, embeddata):
        if not self.client:
            raise ValueError("Client not initialized. Call define_client() first.")
        
        print("Beginning data ingestion...")
        
        # Validate embeddata
        if not hasattr(embeddata, 'contexts') or not hasattr(embeddata, 'embeddings'):
            error_msg = "embeddata object missing required attributes"
            print(error_msg)
            raise ValueError(error_msg)
            
        if not embeddata.contexts:
            error_msg = "No contexts to ingest"
            print(error_msg)
            raise ValueError(error_msg)
            
        if not embeddata.embeddings:
            error_msg = "No embeddings to ingest"
            print(error_msg)
            raise ValueError(error_msg)
            
        # Verify lengths match
        if len(embeddata.contexts) != len(embeddata.embeddings):
            error_msg = f"Contexts length ({len(embeddata.contexts)}) doesn't match embeddings length ({len(embeddata.embeddings)})"
            print(error_msg)
            raise ValueError(error_msg)
        
        print(f"Ready to ingest {len(embeddata.contexts)} items")
        
        # Get the batch iterators before zipping
        context_batches = list(batch_iterate(embeddata.contexts, self.batch_size))
        embedding_batches = list(batch_iterate(embeddata.embeddings, self.batch_size))
        
        print(f"Created {len(context_batches)} context batches and {len(embedding_batches)} embedding batches")
        
        if len(context_batches) != len(embedding_batches):
            error_msg = f"Batch counts don't match: {len(context_batches)} context batches vs {len(embedding_batches)} embedding batches"
            print(error_msg)
            raise ValueError(error_msg)
    
        for i, (batch_context, batch_embeddings) in enumerate(zip(context_batches, embedding_batches)):
            print(f"Processing batch {i+1}/{len(context_batches)}, context size: {len(batch_context)}, embedding size: {len(batch_embeddings)}")
            try:
                self.client.upload_collection(
                    collection_name=self.collection_name,
                    vectors=batch_embeddings,
                    payload=[{"context": context} for context in batch_context]
                )
                print(f"Batch {i+1} uploaded successfully")
            except Exception as e:
                print(f"Error uploading batch {i+1}: {e}")
                raise

        print("Updating collection with indexing threshold")
        try:
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
            )
            print("Collection updated successfully")
        except Exception as e:
            print(f"Error updating collection: {e}")
            raise
        
class Retriever:

    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata
        print("Retriever initialized")

    def search(self, query):
        if not self.vector_db.client:
            raise ValueError("Vector database client not initialized")
            
        print(f"Searching for query: '{query}'")
        try:
            query_embedding = self.embeddata.embed_model.get_query_embedding(query)
            print("Query embedding generated")
            
            result = self.vector_db.client.search(
                collection_name=self.vector_db.collection_name,
                query_vector=query_embedding,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
                limit=5,
                timeout=1000,
            )
            print(f"Search returned {len(result)} results")
            return result
        except Exception as e:
            print(f"Error during search: {e}")
            raise
class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_n=3):
        if not documents:
            return []

        print(f"Reranking {len(documents)} chunks for query: '{query}'")
        # Prepare pairs: (query, chunk)
        pairs = [(query, doc.payload["context"]) for doc in documents]
 

        scores = self.model.predict(pairs)

        # Sort by score
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:top_n]]

        print(f"Top reranked contexts selected: {len(top_docs)}")
        return top_docs
    
class RAG:

    def __init__(self,
                 retriever,
                 llm_name = "Meta-Llama-3.1-405B-Instruct"
    
                ):
        
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that answers questions about the user's media file based on its transcription.",
        )
        self.messages = [system_msg, ]
        self.llm_name = llm_name
        print(f"Initializing RAG with LLM: {llm_name}")
        self.llm = self._setup_llm()
        self.retriever = retriever
        self.reranker = Reranker()

        self.qa_prompt_tmpl_str = ("Context information is below. This comes from a transcribed audio or video file with speaker labels.\n"
                                   "---------------------\n"
                                   "{context}\n"
                                   "---------------------\n"
                                   "Given the context information above I want you to think step by step to answer the query in a crisp manner. If you don't know the answer based on the provided transcript, say 'I don't know based on the available transcript.'\n"
                                   "Query: {query}\n"
                                   "Answer: "
                                   )

    def _setup_llm(self):
        try:
            print(f"Attempting to initialize SambaNovaCloud with model: {self.llm_name}")
            llm = SambaNovaCloud(
                model=self.llm_name,
                temperature=0.7,
                context_window=100000,
            )
            print("SambaNovaCloud initialized successfully")
            return llm
        except Exception as e:
            print(f"Failed to initialize SambaNovaCloud: {e}")
            print("Falling back to Ollama")
            try:
                llm = Ollama(
                    model=self.llm_name,
                    temperature=0.7,
                    context_window=100000,
                )
                print("Ollama initialized successfully")
                return llm
            except Exception as e2:
                print(f"Failed to initialize Ollama: {e2}")
                raise ValueError(f"Could not initialize any LLM: {e}, then {e2}")

    def generate_context(self, query):
        print(f"Generating context for query: '{query}'")
        try:
            result = self.retriever.search(query)
            if not result:
                print("No search results found")
                return "No relevant context found in the transcript."

            # Rerank the retrieved results
            top_docs = self.reranker.rerank(query, result, top_n=3)

            combined_prompt = []
            for i, entry in enumerate(top_docs):
                context_text = entry.payload["context"]

                combined_prompt.append(context_text)
                print(f"Adding reranked context {i+1}: {context_text[:50]}...")

            final_context = "\n\n---\n\n".join(combined_prompt)
            print(f"Final context length: {len(final_context)} characters")
            return final_context
        except Exception as e:
            print(f"Error generating context: {e}")
            raise


    def query(self, query):
        print(f"Processing query: '{query}'")
        try:
            context = self.generate_context(query=query)

            # Enhanced prompt
            prompt = (
            "You are a highly intelligent and helpful assistant.\n\n"
            "Use the following context from a transcribed media file "
            "and your own general knowledge to answer the user's question in a detailed, clear way.\n\n"
            "Context:\n"
            "---------------------\n"
            f"{context}\n"
            "---------------------\n"
            f"User Question: {query}\n"
            "Answer:"
            )

            user_msg = ChatMessage(role=MessageRole.USER, content=prompt)

            print("Sending to LLM for streaming response")
            streaming_response = self.llm.stream_complete(user_msg.content)

            return streaming_response
        except Exception as e:
            print(f"Error in query processing: {e}")
            raise

class Transcribe:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("AssemblyAI API key is required")
        
        print("Initializing AssemblyAI transcriber")
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, str]]:
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found at path: {audio_path}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check if the file is empty
        if os.path.getsize(audio_path) == 0:
            error_msg = f"Audio file is empty (0 bytes): {audio_path}"
            print(error_msg)
            raise ValueError(error_msg)
        
        print(f"Transcribing audio file: {audio_path}")

        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=2
        )

        try:
            print("Sending file to AssemblyAI...")
            transcript = self.transcriber.transcribe(audio_path, config=config)
            print("Transcription completed")
        
            if not transcript:
                print("Warning: Transcript is None")
                return []
            
            if hasattr(transcript, "utterances") and transcript.utterances:
                speaker_transcripts = []
                for utterance in transcript.utterances:
                    speaker_transcripts.append({
                        "speaker": utterance.speaker,
                        "text": utterance.text
                    })
                print(f"Extracted {len(speaker_transcripts)} utterances with speaker labels")
                return speaker_transcripts

            elif hasattr(transcript, "text") and transcript.text:
                print("Found transcript text without speaker labels")
                return [{"speaker": "Unknown", "text": transcript.text}]

            else:
                print("Warning: No transcript content available")
                return []
            
        except Exception as e:
            print(f"Transcription error: {e}")
            raise
