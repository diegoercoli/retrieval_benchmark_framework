from datetime import datetime
from pathlib import Path
import sys
import yaml
from flask import Flask, request, jsonify
import threading
import json
from FlagEmbedding import FlagReranker
import numpy as np
import torch
import threading
#import logging
from sentence_transformers import SentenceTransformer

from src.core.model_manager import ModelManager

#global vars
app = Flask(__name__)
model_manager = ModelManager()

def start_server( configured_model_manager: ModelManager, host: str = "0.0.0.0", port: int = 5000, use_thread: bool = True):
    """
    Start a Flask server with all routes configured.

    Args:
        configured_model_manager: object encapsulating list of models
        host (str): IP address or hostname to bind to. Default "0.0.0.0".
        port (int): Port number. Default 5000.
        use_thread (bool): If True, run Flask in a background thread.
                           If False, block the main process.
    """
    #log_filename = "vectors.log"
    #logging.basicConfig(
    #    filename=log_filename,
    #    level=logging.INFO,
    #    format="%(asctime)s - %(levelname)s - %(message)s",
    #    datefmt="%Y-%m-%d %H:%M:%S"
    #)
    model_manager = configured_model_manager
    #Load the models
    model_manager.preload_models()

    # ---- SERVER RUNNER ----
    def run():
        app.logger.disabled = True
        #log = logging.getLogger('werkzeug')
        #log.setLevel(logging.ERROR)
        app.run(host=host, port=port, debug=False)

    if use_thread:
        flask_thread = threading.Thread(target=run, daemon=True)
        flask_thread.start()
        return flask_thread
    else:
        run()

@app.route('/.well-known/ready', methods=['GET'])
def readiness_check():
    return "Ready", 200


@app.route('/meta', methods=['GET'])
def readiness_check_2():
    #logging.info(f"Received META request")
    return jsonify({'status': 'Ready'}), 200


@app.route('/rerank', methods=['POST'])
def rerank():
    try:
        data = None
        try:
            # Attempt to parse as JSON first
            data = request.json
            if data is None:
                # If request.json was empty, try decoding raw data as JSON string
                text_str = request.data.decode("utf-8")
                data = json.loads(text_str)
            # The entire request body is the JSON object Weaviate sends
            text = data
        except Exception as e:
            # Fallback for unexpected data formats
            try:
                text_str = request.data.decode("utf-8")
                text = json.loads(text_str)
            except Exception as e_inner:
                print(f"Error parsing request data: {e_inner}")
                return jsonify({'error': f"Could not parse request body: {e_inner}"}), 400

        # Validate expected input format from Weaviate
        if not isinstance(text, dict) or 'query' not in text or 'documents' not in text:
            print(f"Invalid input format. Expected dict with 'query' and 'documents'. Got: {text}")
            return jsonify({'error': "Invalid input format. Expected a dictionary with 'query' and 'documents'."}), 400

        query = text['query']
        documents = text['documents']

        if not documents:
            # Return an empty list of scores if no documents are provided for reranking
            # This handles cases where Weaviate might send an empty list, preventing errors
            return jsonify({'scores': []})

        # Prepare pairs for the reranker model
        compares = [(query, doc) for doc in documents]

        # Compute scores using the FlagReranker model
        scores = model_manager.reranker.compute_score(compares)

        # Convert scores (typically a NumPy array or tensor) to a Python list
        scores_list = scores.tolist() if hasattr(scores, 'tolist') else scores

        # Construct the response in the format Weaviate's reranker-transformers module expects
        # This includes the original document text and its computed score
        reranked_results = []
        for i, doc_text in enumerate(documents):
            score = scores_list[i]
            reranked_results.append({
                "document": doc_text,  # Include the original document text
                "score": float(score)  # Use "score" as the key, ensuring float type
            })

        return jsonify({'scores': reranked_results})  # Top-level key is "scores" (plural)

    except Exception as e:
        print(f"Unhandled error in /rerank: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/vectors', methods=['POST'])
def vectorize():
    try:
        try:
            data = request.json.get('text')
        except Exception as e:
            try:
                data = request.data.decode("utf-8")
            except Exception as e:
                print(e)
        text = json.loads(data)

        if isinstance(text, str):
            text = [text]
        else:
            text = text['text']
        # Write in a log file the fact that you received text
        # Example: include timestamp in the log filename
        #logging.basicConfig(
        #    filename=log_filename,
        #    level=logging.INFO,
        #    format='%(asctime)s - %(levelname)s - %(message)s'
        #)
        #logging.info(f"Received text for vectorization: {text}")

        ####
        embeddings = model_manager.embedding_model.encode(text).tolist()

        return jsonify({'vector': embeddings})


    except Exception as e:
        return jsonify({'error': str(e)}), 500


def load_config_and_start_server(config_path: str = "config/benchmark_config.yaml"):
    """
    Load configuration from YAML file and start the embedding server.

    Args:
        config_path: Path to the YAML configuration file
    """
    try:
        # Load configuration
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Extract model configurations
        embedding_model = config['embedding']['model']
        reranking_model = config['reranking']['model']

        # Create model manager
        server_model_manager = ModelManager(
            embedding_model_name=embedding_model,
            reranker_model_name=reranking_model
        )

        # Extract server configuration
        server_url = config['embedding']['transformers_inference_api']

        # Parse host and port from URL (e.g., 'http://172.20.64.1:5000')
        if '://' in server_url:
            host_port = server_url.split('://')[-1]
        else:
            host_port = server_url

        if ':' in host_port:
            host, port_str = host_port.split(':')
            port = int(port_str)
        else:
            host = host_port
            port = 5000

        print(f"Starting embedding server with:")
        print(f"  Embedding model: {embedding_model}")
        print(f"  Reranking model: {reranking_model}")
        print(f"  Server address: {host}:{port}")

        # Start server (blocking mode when run as main)
        start_server(
            configured_model_manager=server_model_manager,
            host=host,
            port=port,
            use_thread=False  # Block the main thread
        )

    except Exception as e:
        print(f"Failed to start embedding server: {e}")
        raise


if __name__ == "__main__":
    """
    Main entry point when running the embedding service standalone.

    Usage:
        python src/vectordb/old/embedding_service.py

    Or with custom config:
        python src/vectordb/old/embedding_service.py --config path/to/config.yaml
    """

    # Check for custom config path argument
    config_path = "../../config/benchmark_config.yaml"  # default

    if len(sys.argv) > 1:
        if sys.argv[1] == "--config" and len(sys.argv) > 2:
            config_path = sys.argv[2]
        else:
            config_path = sys.argv[1]

    print("Starting Embedding & Reranking Server...")
    print(f"Using configuration: {config_path}")

    try:
        load_config_and_start_server(config_path)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server failed: {e}")
        sys.exit(1)


