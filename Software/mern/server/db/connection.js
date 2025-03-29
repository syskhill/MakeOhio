import { MongoClient, ServerApiVersion } from 'mongodb';
import dotenv from 'dotenv';
dotenv.config();

// Ensure that the URI is set correctly
// const URI = process.env.ATLAS_URI;
const URI = "mongodb+srv://bn00017:QqqUP3%40duTjSxPu@cluster0.nh2ok3z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";
if (!URI) {
  console.error("MongoDB URI not found in the environment variables.");
  process.exit(1);  // Exit the process if no URI is found
}

const client = new MongoClient(URI, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  },
});

async function connectToDatabase() {
  try {
    // Connect the client to the server
    await client.connect();
    // Send a ping to confirm a successful connection
    await client.db('admin').command({ ping: 1 });
    console.log('Pinged your deployment. You successfully connected to MongoDB!');
  } catch (err) {
    console.error('Error connecting to MongoDB:', err);
  }
}

connectToDatabase();  // Call the async function

let db = client.db('employees');

export default db;
