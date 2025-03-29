import pymongo
import logging
from pymongo import MongoClient
from bson.objectid import ObjectId

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_mongodb")

# MongoDB connection settings
MONGODB_URI = "mongodb+srv://bn00017:QqqUP3%40duTjSxPu@cluster0.nh2ok3z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB = "employees"
MONGODB_COLLECTION = "patients"

def test_mongodb_connection():
    """Test connection to MongoDB"""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        
        # Test connection by getting server info
        server_info = client.server_info()
        logger.info(f"Connected to MongoDB server version: {server_info.get('version')}")
        
        # Get database and collection
        db = client[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]
        
        # Count documents in collection
        doc_count = collection.count_documents({})
        logger.info(f"Collection '{MONGODB_COLLECTION}' contains {doc_count} documents")
        
        # Get a list of all databases
        databases = client.list_database_names()
        logger.info(f"Available databases: {', '.join(databases)}")
        
        # Get a list of all collections in the database
        collections = db.list_collection_names()
        logger.info(f"Collections in '{MONGODB_DB}' database: {', '.join(collections)}")
        
        # Get a sample of documents
        sample_docs = list(collection.find().limit(3))
        logger.info(f"Sample documents from collection (up to 3):")
        for i, doc in enumerate(sample_docs):
            doc_id = doc.get('_id', 'Unknown')
            doc_name = doc.get('name', 'Unknown')
            logger.info(f"Document {i+1}: ID = {doc_id}, Name = {doc_name}")
        
        # Close connection
        client.close()
        logger.info("MongoDB connection test PASSED")
        return True
    
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        logger.error("MongoDB connection test FAILED")
        return False

if __name__ == "__main__":
    logger.info("Starting MongoDB Connection Test")
    logger.info("------------------------------")
    test_mongodb_connection()