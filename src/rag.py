import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
DEFAULT_SAMPLE_PATH = os.path.join(BASE_DIR, "data", "raw_sample.csv")

class RAGRetriever:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, collection_name: str = "amazon_support_history"):
        print(f"Initializing ChromaDB at '{db_path}' and local embedding model...")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Local open-source embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def populate_index(self, sample_csv_path: str = DEFAULT_SAMPLE_PATH):
        """
        Loads preprocessed dataset into ChromaDB if the index is empty.
        """
        if self.collection.count() > 0:
            print(f"Collection already populated with {self.collection.count()} records.")
            return

        print(f"Indexing historical responses from {sample_csv_path}...")
        df = pd.read_csv(sample_csv_path)

        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            documents.append(str(row["customer_text"]))
            metadatas.append({
                "brand_response": str(row["brand_response"]),
                "customer_tweet_id": str(row["customer_tweet_id"])
            })
            ids.append(f"doc_{idx}")

        # Add records to vector store in safe batches
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )
        print(f"Successfully indexed {len(documents)} resolution pairs into ChromaDB.")

    def retrieve_context(self, query: str, top_k: int = 2) -> list:
        """
        Retrieves top_k similar historical resolution pairs for grounding.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        retrieved_contexts = []
        if results and "documents" in results and results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                retrieved_contexts.append({
                    "historical_customer_query": doc,
                    "historical_brand_response": meta["brand_response"]
                })

        return retrieved_contexts

if __name__ == "__main__":
    retriever = RAGRetriever()
    retriever.populate_index()

    test_query = "My package was supposed to arrive yesterday but the tracking number shows delayed."
    contexts = retriever.retrieve_context(test_query, top_k=2)

    print("\n--- Retrieved RAG Contexts ---")
    for i, ctx in enumerate(contexts, 1):
        print(f"\nResult #{i}:")
        print(f"Past Customer Query: {ctx['historical_customer_query']}")
        print(f"Past Brand Answer:   {ctx['historical_brand_response']}")