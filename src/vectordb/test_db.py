import os
import yaml
from src.utils.proxy_helper import set_no_proxy_localhost
from src.vectordb.weaviate_db_manager import WeaviateDBManager


class Chunk:
    def __init__(self, text, chunk_id=None, source=None, doc_id=None):
        """
        Initialize a Chunk object.

        Args:
            text (str): The main text content of the chunk.
            chunk_id (str, optional): Unique identifier for the chunk.
            source (str, optional): Source information (e.g., filename, url).
            doc_id (str, optional): Document identifier this chunk belongs to.
        """
        self.text = text
        self.chunk_id = chunk_id
        self.source = source
        self.doc_id = doc_id

    def to_dict(self):
        """
        Convert the chunk to a dictionary, useful for embedding.

        Returns:
            dict: Dictionary with metadata and text field.
        """
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "doc_id": self.doc_id
        }


def create_samples():
    """Create sample chunks for testing"""
    examples = [
        Chunk(
            text="Deep learning has revolutionized computer vision tasks.",
            chunk_id="c1",
            source="paper1.txt",
            doc_id="docA"
        ),
        Chunk(
            text="Natural language processing enables machines to understand text.",
            chunk_id="c2",
            source="paper2.txt",
            doc_id="docB"
        ),
        Chunk(
            text="Embeddings are vector representations of data.",
            chunk_id="c3",
            source="notes.md",
            doc_id="docC"
        ),
        Chunk(
            text="Chunking is used to divide long texts for easier processing.",
            chunk_id="c4",
            source="tutorial.pdf",
            doc_id="docD"
        ),
        Chunk(
            text="Metadata helps track the origin and context of each chunk.",
            chunk_id="c5",
            source="handbook.docx",
            doc_id="docE"
        ),
        Chunk(
            text="Machine learning models require large amounts of training data.",
            chunk_id="c6",
            source="ml_basics.txt",
            doc_id="docF"
        ),
        Chunk(
            text="Transfer learning allows models to leverage pre-trained knowledge.",
            chunk_id="c7",
            source="advanced_ml.pdf",
            doc_id="docG"
        ),
        Chunk(
            text="Vector databases enable efficient similarity search at scale.",
            chunk_id="c8",
            source="vectordb_guide.md",
            doc_id="docH"
        )
    ]
    return examples


def test_collection(db_manager, collection_name):
    """Test if a collection exists"""
    exist = db_manager.collection_exists(collection_name)
    print(f"\n=== COLLECTION '{collection_name}' exist? {'yes' if exist else 'no'} ===")
    return exist


def test_insertion(db_manager, collection_name):
    """Test inserting data into a collection"""
    print(f"\n{'=' * 60}")
    print(f"Testing Insertion into '{collection_name}'")
    print('=' * 60)

    samples = create_samples()
    if len(samples) == 0:
        print("\n=== NO SAMPLES TO INSERT ===")
        return False

    print(f"Creating collection with {len(samples)} chunks...")
    prototype = samples[0]

    try:
        success = db_manager.create_collection(
            collection_name,
            samples,
            list(prototype.to_dict().keys()),
            "text",
            overwrite=True
        )

        if success:
            print(f"✓ Collection '{collection_name}' created successfully")
            print(f"✓ Inserted {len(samples)} chunks")
            return True
        else:
            print(f"✗ Failed to create collection '{collection_name}'")
            return False
    except Exception as e:
        print(f"✗ Error during insertion: {e}")
        return False


def count_records(db_manager, collection_name):
    """Count the number of records in a collection"""
    print(f"\n{'=' * 60}")
    print(f"Counting Records in '{collection_name}'")
    print('=' * 60)

    try:
        count = db_manager.get_collection_count(collection_name)
        print(f"✓ Found {count} records in collection '{collection_name}'")
        return count
    except Exception as e:
        print(f"✗ Error counting records: {e}")
        return 0


def test_retrieval(db_manager, collection_name):
    """Test various retrieval methods"""
    print(f"\n{'=' * 60}")
    print(f"Testing Retrieval from '{collection_name}'")
    print('=' * 60)

    set_no_proxy_localhost()

    # Test 1: Semantic Search
    print("\n[Test 1] Semantic Search")
    print("-" * 40)
    query = "What is machine learning?"
    try:
        results = db_manager.semantic_search_retrieve(
            query=query,
            collection_name=collection_name,
            top_k=3
        )
        print(f"Query: '{query}'")
        print(f"✓ Retrieved {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Text: {result.get('text', 'N/A')[:80]}...")
            print(f"    Source: {result.get('source', 'N/A')}")
            print(f"    Doc ID: {result.get('doc_id', 'N/A')}")
    except Exception as e:
        print(f"✗ Semantic search failed: {e}")

    # Test 2: BM25 Search
    print("\n[Test 2] BM25 Search")
    print("-" * 40)
    query = "embeddings vector"
    try:
        results = db_manager.bm25_retrieve(
            query=query,
            collection_name=collection_name,
            top_k=3
        )
        print(f"Query: '{query}'")
        print(f"✓ Retrieved {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Text: {result.get('text', 'N/A')[:80]}...")
            print(f"    Chunk ID: {result.get('chunk_id', 'N/A')}")
    except Exception as e:
        print(f"✗ BM25 search failed: {e}")

    # Test 3: Hybrid Search
    print("\n[Test 3] Hybrid Search")
    print("-" * 40)
    query = "deep learning computer vision"
    try:
        results = db_manager.hybrid_search(
            query=query,
            collection_name=collection_name,
            top_k=3,
            alpha=0.5
        )
        print(f"Query: '{query}'")
        print(f"Alpha: 0.5 (balanced semantic and keyword)")
        print(f"✓ Retrieved {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Text: {result.get('text', 'N/A')[:80]}...")
            print(f"    Source: {result.get('source', 'N/A')}")
    except Exception as e:
        print(f"✗ Hybrid search failed: {e}")

    # Test 4: Metadata Filtering
    print("\n[Test 4] Metadata Filtering")
    print("-" * 40)
    try:
        results = db_manager.filter_by_metadata(
            metadata_property="source",
            values=["paper1.txt", "paper2.txt"],
            collection_name=collection_name,
            limit=5
        )
        print(f"Filter: source contains 'paper1.txt' or 'paper2.txt'")
        print(f"✓ Retrieved {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Text: {result.get('text', 'N/A')[:80]}...")
            print(f"    Source: {result.get('source', 'N/A')}")
    except Exception as e:
        print(f"✗ Metadata filtering failed: {e}")


def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(
        os.path.dirname(__file__),
        '../../config/benchmark_config.yaml'
    )

    # Default values
    port = 8081
    grpc_port = 50051
    inference_url = 'http://localhost:8080'

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            port = config.get('weaviate', {}).get('port', 8081)
            grpc_port = config.get('weaviate', {}).get('grpc_port', 50051)
            inference_url = config.get('embedding', {}).get(
                'transformers_inference_api',
                'http://localhost:8080'
            )
            print(f"✓ Loaded configuration from {config_path}")
        except Exception as e:
            print(f"⚠ Failed to load config, using defaults: {e}")
    else:
        print(f"⚠ Config file not found, using default values")

    return port, grpc_port, inference_url


def main():
    """Main test suite execution"""
    print("=" * 60)
    print("Weaviate DB Manager Test Suite")
    print("Using Context Manager for Connection Efficiency")
    print("=" * 60)

    # Test collection names
    test_collections = [
        "test_collection",
        "hierarchicalchunkingparaphrasemultilingualMiniLML12v2"
    ]

    try:
        # Load configuration
        port, grpc_port, inference_url = load_config()
        # print the loaded config
        print(f"\nWeaviate Config - Port: {port}, gRPC Port: {grpc_port}, Inference URL: {inference_url}")
        # Use context manager - connection stays open for all operations
        with WeaviateDBManager(
                port=port,
                grpc_port=grpc_port,
                inference_url=inference_url
        ) as db_manager:

            print(f"\n✓ Connected to Weaviate (connection will be reused for all operations)")

            for collection_name in test_collections:
                print(f"\n\n{'#' * 60}")
                print(f"# Testing Collection: {collection_name}")
                print('#' * 60)

                try:
                    # Check if collection exists
                    exists_before = test_collection(db_manager, collection_name)

                    # Insert data
                    insertion_success = test_insertion(db_manager, collection_name)

                    if insertion_success:
                        # Verify collection exists after insertion
                        exists_after = test_collection(db_manager, collection_name)

                        # Count records
                        record_count = count_records(db_manager, collection_name)

                        # Test retrieval methods
                        if record_count > 0:
                            test_retrieval(db_manager, collection_name)
                        else:
                            print("\n⚠ No records found, skipping retrieval tests")
                    else:
                        print(f"\n⚠ Skipping tests for '{collection_name}' due to insertion failure")

                except Exception as e:
                    print(f"\n✗ Error processing collection '{collection_name}': {e}")
                    print(f"   Continuing with next collection...")
                    continue

        # Connection automatically closed here
        print("\n" + "=" * 60)
        print("Test Suite Completed!")
        print("✓ Connection closed automatically")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user (Ctrl+C)")
        print("Cleaning up...")

    except Exception as e:
        print(f"\n\n✗ Fatal error in test suite: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n" + "=" * 60)
        print("Exiting test suite")
        print("=" * 60)


if __name__ == "__main__":
    main()