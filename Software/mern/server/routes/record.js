import express from "express";

// This will help us connect to the database
import db from "../db/connection.js";

// This help convert the id from string to ObjectId for the _id.
import { ObjectId } from "mongodb";

// For time calculations
import { parseISO, differenceInMinutes } from "date-fns";

// router is an instance of the express router.
// We use it to define our routes.
// The router will be added as a middleware and will take control of requests starting with path /record.
const router = express.Router();

// This section will help you get a list of all the records.
router.get("/", async (req, res) => {
  let collection = await db.collection("patients");
  let results = await collection.find({}).toArray();
  res.send(results).status(200);
});

// This section will help you get a single record by id
router.get("/:id", async (req, res) => {
  let collection = await db.collection("patients");
  let query = { _id: new ObjectId(req.params.id) };
  let result = await collection.findOne(query);

  if (!result) res.send("Not found").status(404);
  else res.send(result).status(200);
});

// This section will help you create a new record.
router.post("/", async (req, res) => {
  try {
    let newDocument = {
      name: req.body.name,
      photoUrl: req.body.photoUrl,
      pillTimes: req.body.pillTimes,
      slotNumber: req.body.slotNumber,
      createdAt: new Date()
    };
    let collection = await db.collection("patients");
    let result = await collection.insertOne(newDocument);
    res.send(result).status(204);
  } catch (err) {
    console.error(err);
    res.status(500).send("Error adding patient record");
  }
});

// This section will help you update a record by id.
router.patch("/:id", async (req, res) => {
  try {
    const query = { _id: new ObjectId(req.params.id) };
    const updates = {
      $set: {
        name: req.body.name,
        photoUrl: req.body.photoUrl,
        pillTimes: req.body.pillTimes,
        slotNumber: req.body.slotNumber,
        updatedAt: new Date()
      },
    };

    let collection = await db.collection("patients");
    let result = await collection.updateOne(query, updates);
    res.send(result).status(200);
  } catch (err) {
    console.error(err);
    res.status(500).send("Error updating patient record");
  }
});

// This section will help you delete a record
router.delete("/:id", async (req, res) => {
  try {
    const query = { _id: new ObjectId(req.params.id) };

    const collection = db.collection("patients");
    let result = await collection.deleteOne(query);

    res.send(result).status(200);
  } catch (err) {
    console.error(err);
    res.status(500).send("Error deleting patient record");
  }
});

// Arduino API endpoint for face recognition and time-based access control
router.get("/arduino/patients", async (req, res) => {
  try {
    let collection = await db.collection("patients");
    let patients = await collection.find({}).toArray();
    
    // Format the data in a way that's easier for Arduino to process
    const formattedPatients = patients.map(patient => {
      return {
        id: patient._id.toString(),
        name: patient.name,
        photo: patient.photoUrl,
        pillTimes: patient.pillTimes,
        slotNumber: parseInt(patient.slotNumber) || 0
      };
    });
    
    res.json(formattedPatients).status(200);
  } catch (err) {
    console.error(err);
    res.status(500).send("Error retrieving patient data for Arduino");
  }
});

// Arduino API endpoint to check if a patient can access their pills based on time
router.get("/arduino/access/:id", async (req, res) => {
  try {
    const id = req.params.id;
    const collection = await db.collection("patients");
    const patient = await collection.findOne({ _id: new ObjectId(id) });
    
    if (!patient) {
      return res.status(404).json({ 
        access: false, 
        message: "Patient not found" 
      });
    }
    
    // Get current time
    const now = new Date();
    const currentHour = now.getHours();
    const currentMinute = now.getMinutes();
    
    // Parse pill times (format expected: "8:00,12:00,18:00")
    const pillTimeStrings = patient.pillTimes.split(",").map(time => time.trim());
    let accessGranted = false;
    let nearestTime = "";
    
    // Check if current time is within ±1 hour of any pill time
    for (const timeStr of pillTimeStrings) {
      const [hour, minute] = timeStr.split(":").map(num => parseInt(num));
      
      // Create date objects for comparison
      const pillTime = new Date();
      pillTime.setHours(hour, minute, 0);
      
      // Calculate difference in minutes
      const diffMinutes = Math.abs(
        (currentHour * 60 + currentMinute) - (hour * 60 + minute)
      );
      
      // Allow access if within ±60 minutes
      if (diffMinutes <= 60) {
        accessGranted = true;
        nearestTime = timeStr;
        break;
      }
    }
    
    // Return access decision
    return res.json({
      access: accessGranted,
      patientName: patient.name,
      slotNumber: parseInt(patient.slotNumber) || 0,
      nearestTime: nearestTime,
      message: accessGranted 
        ? `Access granted for ${patient.name} to slot ${patient.slotNumber}`
        : `Access denied: Not within dosage time window`
    });
    
  } catch (err) {
    console.error(err);
    res.status(500).json({ 
      access: false, 
      message: "Error checking access" 
    });
  }
});

export default router;
