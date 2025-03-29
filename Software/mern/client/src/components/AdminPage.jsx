import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";

export default function AdminPage() {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch all patient data when component mounts
  useEffect(() => {
    async function fetchPatients() {
      try {
        setLoading(true);
        const response = await fetch("http://localhost:5050/record");
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        setPatients(data);
      } catch (error) {
        console.error("Error fetching patient data:", error);
        setError("Failed to load patient data. Please try again later.");
      } finally {
        setLoading(false);
      }
    }
    
    fetchPatients();
  }, []);

  // Function to format date
  const formatDate = (dateString) => {
    if (!dateString) return "N/A";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div className="container mx-auto p-4">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Admin Dashboard</h1>
        <div className="space-x-2">
          <Link 
            to="/" 
            className="inline-flex items-center justify-center whitespace-nowrap text-sm font-medium bg-gray-200 p-2 rounded hover:bg-gray-300"
          >
            View Patient List
          </Link>
          <Link 
            to="/form" 
            className="inline-flex items-center justify-center whitespace-nowrap text-sm font-medium bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
          >
            Add New Patient
          </Link>
        </div>
      </div>
      
      {loading ? (
        <div className="text-center py-8">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-500 border-r-transparent"></div>
          <p className="mt-2">Loading patient data...</p>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p>{error}</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border rounded-lg overflow-hidden">
            <thead className="bg-gray-100">
              <tr>
                <th className="py-3 px-4 text-left font-semibold">ID</th>
                <th className="py-3 px-4 text-left font-semibold">Patient Name</th>
                <th className="py-3 px-4 text-left font-semibold">Photo</th>
                <th className="py-3 px-4 text-left font-semibold">Pill Times</th>
                <th className="py-3 px-4 text-left font-semibold">Slot #</th>
                <th className="py-3 px-4 text-left font-semibold">Created</th>
                <th className="py-3 px-4 text-left font-semibold">Updated</th>
                <th className="py-3 px-4 text-left font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {patients.length === 0 ? (
                <tr>
                  <td colSpan="8" className="py-4 px-4 text-center">
                    No patient records found
                  </td>
                </tr>
              ) : (
                patients.map((patient) => (
                  <tr key={patient._id} className="border-t hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm">
                      <span className="font-mono">{patient._id}</span>
                    </td>
                    <td className="py-3 px-4">{patient.name}</td>
                    <td className="py-3 px-4">
                      {patient.photoUrl ? (
                        <div className="relative w-12 h-12 rounded overflow-hidden">
                          <span className="text-xs">{patient.photoUrl}</span>
                        </div>
                      ) : (
                        <span className="text-gray-400">No photo</span>
                      )}
                    </td>
                    <td className="py-3 px-4">{patient.pillTimes}</td>
                    <td className="py-3 px-4">{patient.slotNumber}</td>
                    <td className="py-3 px-4 text-sm">{formatDate(patient.createdAt)}</td>
                    <td className="py-3 px-4 text-sm">{formatDate(patient.updatedAt)}</td>
                    <td className="py-3 px-4">
                      <div className="flex gap-2">
                        <Link
                          to={`/edit/${patient._id}`}
                          className="px-3 py-1 bg-blue-100 text-blue-800 rounded text-sm hover:bg-blue-200"
                        >
                          Edit
                        </Link>
                        <button
                          onClick={async () => {
                            if (window.confirm(`Are you sure you want to delete ${patient.name}?`)) {
                              try {
                                await fetch(`http://localhost:5050/record/${patient._id}`, {
                                  method: "DELETE",
                                });
                                setPatients(patients.filter(p => p._id !== patient._id));
                              } catch (error) {
                                console.error("Error deleting patient:", error);
                                alert("Failed to delete patient");
                              }
                            }
                          }}
                          className="px-3 py-1 bg-red-100 text-red-800 rounded text-sm hover:bg-red-200"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
      
      <div className="mt-8 bg-blue-50 p-4 rounded border border-blue-200">
        <h2 className="text-lg font-semibold mb-2">Database Information</h2>
        <p className="text-sm">
          <strong>Collection:</strong> patients<br />
          <strong>Total Records:</strong> {patients.length}<br />
          <strong>Server:</strong> http://localhost:5050
        </p>
      </div>
    </div>
  );
}