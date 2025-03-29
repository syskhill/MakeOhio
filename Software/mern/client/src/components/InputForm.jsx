import React, { useState } from "react";

export default function InputForm() {
  const [formData, setFormData] = useState({
    name: "",
    photoUrl: "",
    pillTimes: "",
    slotNumber: ""
  });
  
  const [imagePreview, setImagePreview] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };
  
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Create a preview URL for the selected image
    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);
    
    // For a real app, you would upload to a server and get back a URL
    // For this example, we'll just store the file name
    setFormData((prevData) => ({
      ...prevData,
      photoUrl: file.name,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      // In a real app, you would first upload the image to a storage service
      // and then save the record with the image URL
      
      const response = await fetch("http://localhost:5050/record", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // Reset the form
      setFormData({
        name: "",
        photoUrl: "",
        pillTimes: "",
        slotNumber: ""
      });
      setImagePreview(null);
      
      alert("Patient record added successfully!");
      
      // Redirect to the patient list
      window.location.href = "/";
      
    } catch (error) {
      console.error("Error submitting the form:", error);
      alert("Failed to submit the form. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-2xl font-bold mb-6">Patient Pill Management</h1>
      <form
        onSubmit={handleSubmit}
        className="bg-white p-6 rounded shadow-md w-96"
        encType="multipart/form-data"
      >
        {/* Name Input */}
        <div className="mb-4">
          <label className="block text-gray-700 font-medium mb-2" htmlFor="name">
            Patient Name
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            placeholder="Enter patient name"
            required
          />
        </div>

        {/* Photo Upload */}
        <div className="mb-4">
          <label
            className="block text-gray-700 font-medium mb-2"
            htmlFor="photo"
          >
            Patient Photo
          </label>
          <input
            type="file"
            id="photo"
            name="photo"
            accept="image/*"
            onChange={handleFileChange}
            className="w-full p-2 border border-gray-300 rounded"
            required
          />
          {imagePreview && (
            <div className="mt-2">
              <img 
                src={imagePreview} 
                alt="Preview" 
                className="w-full h-40 object-cover rounded" 
              />
            </div>
          )}
        </div>

        {/* Pill Times Input */}
        <div className="mb-4">
          <label
            className="block text-gray-700 font-medium mb-2"
            htmlFor="pillTimes"
          >
            Allotted Pill Times
          </label>
          <input
            type="text"
            id="pillTimes"
            name="pillTimes"
            value={formData.pillTimes}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            placeholder="Example: 8:00,12:00,18:00"
            required
          />
          <p className="text-sm text-gray-500 mt-1">
            Enter times separated by commas (24-hour format)
          </p>
        </div>

        {/* Slot Number Input */}
        <div className="mb-4">
          <label
            className="block text-gray-700 font-medium mb-2"
            htmlFor="slotNumber"
          >
            Slot Number
          </label>
          <input
            type="number"
            id="slotNumber"
            name="slotNumber"
            min="1"
            max="10"
            value={formData.slotNumber}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            placeholder="Enter slot number (1-10)"
            required
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 disabled:bg-blue-300"
          disabled={isSubmitting}
        >
          {isSubmitting ? "Submitting..." : "Save Patient Record"}
        </button>
      </form>
    </div>
  );
}