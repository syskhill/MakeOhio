import React, { useState } from "react";

export default function InputForm() {
  const [formData, setFormData] = useState({
    name: "",
    idNumber: "",
    dosage: "",
    timeslot: "",
    config: "",
    
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form Data Submitted:", formData);
    alert("Form submitted successfully!");
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-2xl font-bold mb-6">Input Form</h1>
      <form
        onSubmit={handleSubmit}
        className="bg-white p-6 rounded shadow-md w-96"
      >
        {/* Name Input */}
        <div className="mb-4">
          <label className="block text-gray-700 font-medium mb-2" htmlFor="name">
            Name
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            placeholder="Enter Name"
          />
        </div>

        {/* ID Number Input */}
        <div className="mb-4">
          <label
            className="block text-gray-700 font-medium mb-2"
            htmlFor="idNumber"
          >
            ID Number
          </label>
          <input
            type="text"
            id="idNumber"
            name="idNumber"
            value={formData.idNumber}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            placeholder="Enter ID Number"
          />
        </div>

        {/* Dosage Input */}
        <div className="mb-4">
          <label
            className="block text-gray-700 font-medium mb-2"
            htmlFor="dosage"
          >
            Dosage
          </label>
          <input
            type="text"
            id="dosage"
            name="dosage"
            value={formData.dosage}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            placeholder="Enter Dosage"
          />
        </div>

        {/* Timeslot Input */}
        <div className="mb-4">
          <label
            className="block text-gray-700 font-medium mb-2"
            htmlFor="timeslot"
          >
            Timeslot
          </label>
          <input
            type="text"
            id="timeslot"
            name="timeslot"
            value={formData.timeslot}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            placeholder="Enter Timeslot"
          />
        </div>

        {/* Config Input */}
        <div className="mb-4">
          <label
            className="block text-gray-700 font-medium mb-2"
            htmlFor="config"
          >
            Config
          </label>
          <input
            type="text"
            id="config"
            name="config"
            value={formData.config}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded"
            placeholder="Enter Config"
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
        >
          Submit
        </button>
      </form>
    </div>
  );
}

// filepath: c:\Users\conno\Downloads\Hackathons\MakeOhio\Software\mern\client\src\main.jsx


const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      {
        path: "/",
        element: <RecordList />,
      },
    ],
  },
  {
    path: "/create",
    element: <App />,
    children: [
      {
        path: "/create",
        element: <Record />,
      },
    ],
  },
  {
    path: "/form",
    element: <InputForm />, // Add this route for the InputForm
  },
]);