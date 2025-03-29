import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";

export default function Record() {
  const [form, setForm] = useState({
    name: "",
    photoUrl: "",
    pillTimes: "",
    slotNumber: "",
  });
  const [isNew, setIsNew] = useState(true);
  const [imagePreview, setImagePreview] = useState(null);
  const params = useParams();
  const navigate = useNavigate();

  useEffect(() => {
    async function fetchData() {
      const id = params.id?.toString() || undefined;
      if(!id) return;
      setIsNew(false);
      const response = await fetch(
        `http://localhost:5050/record/${params.id.toString()}`
      );
      if (!response.ok) {
        const message = `An error has occurred: ${response.statusText}`;
        console.error(message);
        return;
      }
      const record = await response.json();
      if (!record) {
        console.warn(`Record with id ${id} not found`);
        navigate("/");
        return;
      }
      setForm(record);
      // If there's a photo URL, set it as preview
      if (record.photoUrl) {
        setImagePreview(record.photoUrl);
      }
    }
    fetchData();
    return;
  }, [params.id, navigate]);

  // These methods will update the state properties.
  function updateForm(value) {
    return setForm((prev) => {
      return { ...prev, ...value };
    });
  }

  // Handle file uploads for patient photos
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Create a preview URL for the selected image
    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);
    
    // For a real app, you would upload to a server and get back a URL
    // For this example, we'll just store the file name
    updateForm({ photoUrl: file.name });
  };

  // This function will handle the submission.
  async function onSubmit(e) {
    e.preventDefault();
    const patient = { ...form };
    try {
      let response;
      if (isNew) {
        // if we are adding a new record we will POST to /record.
        response = await fetch("http://localhost:5050/record", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(patient),
        });
      } else {
        // if we are updating a record we will PATCH to /record/:id.
        response = await fetch(`http://localhost:5050/record/${params.id}`, {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(patient),
        });
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      alert(isNew ? "Patient added successfully!" : "Patient updated successfully!");
    } catch (error) {
      console.error('A problem occurred adding or updating a record: ', error);
      alert("Error saving patient data");
    } finally {
      setForm({ name: "", photoUrl: "", pillTimes: "", slotNumber: "" });
      navigate("/");
    }
  }

  // This following section will display the form that takes the input from the user.
  return (
    <>
      <h3 className="text-lg font-semibold p-4">{isNew ? 'Create' : 'Update'} Patient Record</h3>
      
      {!isNew && (
        <div className="px-4 mb-4">
          <p className="text-sm font-mono bg-gray-100 p-2 rounded">Patient ID: {params.id}</p>
        </div>
      )}
      
      <form
        onSubmit={onSubmit}
        className="border rounded-lg overflow-hidden p-4"
        encType="multipart/form-data"
      >
        <div className="grid grid-cols-1 gap-x-8 gap-y-10 border-b border-slate-900/10 pb-12 md:grid-cols-2">
          <div>
            <h2 className="text-base font-semibold leading-7 text-slate-900">
              Patient Info
            </h2>
            <p className="mt-1 text-sm leading-6 text-slate-600">
              Update the patient information for medication dispensing.
            </p>
          </div>

          <div className="grid max-w-2xl grid-cols-1 gap-x-6 gap-y-8 ">
            {/* Name Input */}
            <div className="sm:col-span-4">
              <label
                htmlFor="name"
                className="block text-sm font-medium leading-6 text-slate-900"
              >
                Patient Name
              </label>
              <div className="mt-2">
                <div className="flex rounded-md shadow-sm ring-1 ring-inset ring-slate-300 focus-within:ring-2 focus-within:ring-inset focus-within:ring-indigo-600 sm:max-w-md">
                  <input
                    type="text"
                    name="name"
                    id="name"
                    className="block flex-1 border-0 bg-transparent py-1.5 pl-1 text-slate-900 placeholder:text-slate-400 focus:ring-0 sm:text-sm sm:leading-6"
                    placeholder="Patient Full Name"
                    value={form.name}
                    onChange={(e) => updateForm({ name: e.target.value })}
                    required
                  />
                </div>
              </div>
            </div>
            
            {/* Photo Upload */}
            <div className="sm:col-span-4">
              <label
                htmlFor="photoUrl"
                className="block text-sm font-medium leading-6 text-slate-900"
              >
                Photo URL/Path
              </label>
              <div className="mt-2">
                <div className="flex rounded-md shadow-sm ring-1 ring-inset ring-slate-300 focus-within:ring-2 focus-within:ring-inset focus-within:ring-indigo-600 sm:max-w-md">
                  <input
                    type="text"
                    name="photoUrl"
                    id="photoUrl"
                    className="block flex-1 border-0 bg-transparent py-1.5 pl-1 text-slate-900 placeholder:text-slate-400 focus:ring-0 sm:text-sm sm:leading-6"
                    placeholder="URL or file path to patient photo"
                    value={form.photoUrl}
                    onChange={(e) => updateForm({ photoUrl: e.target.value })}
                  />
                </div>
              </div>
              
              <div className="mt-2">
                <label className="block text-sm font-medium mb-1">Or upload a new photo:</label>
                <input
                  type="file"
                  id="photo"
                  name="photo"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="w-full p-2 border border-gray-300 rounded"
                />
              </div>
              
              {imagePreview && (
                <div className="mt-2">
                  <p className="text-sm font-medium mb-1">Current photo:</p>
                  <div className="w-24 h-24 border rounded overflow-hidden">
                    {typeof imagePreview === 'string' && imagePreview.startsWith('http') ? (
                      <img src={imagePreview} alt="Patient" className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-gray-100 text-xs text-gray-500 p-1">
                        {form.photoUrl || 'No image'}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            {/* Pill Times Input */}
            <div className="sm:col-span-4">
              <label
                htmlFor="pillTimes"
                className="block text-sm font-medium leading-6 text-slate-900"
              >
                Pill Times
              </label>
              <div className="mt-2">
                <div className="flex rounded-md shadow-sm ring-1 ring-inset ring-slate-300 focus-within:ring-2 focus-within:ring-inset focus-within:ring-indigo-600 sm:max-w-md">
                  <input
                    type="text"
                    name="pillTimes"
                    id="pillTimes"
                    className="block flex-1 border-0 bg-transparent py-1.5 pl-1 text-slate-900 placeholder:text-slate-400 focus:ring-0 sm:text-sm sm:leading-6"
                    placeholder="Example: 8:00,12:00,18:00"
                    value={form.pillTimes}
                    onChange={(e) => updateForm({ pillTimes: e.target.value })}
                    required
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">Enter times separated by commas (24-hour format)</p>
              </div>
            </div>
            
            {/* Slot Number Input */}
            <div className="sm:col-span-4">
              <label
                htmlFor="slotNumber"
                className="block text-sm font-medium leading-6 text-slate-900"
              >
                Slot Number
              </label>
              <div className="mt-2">
                <div className="flex rounded-md shadow-sm ring-1 ring-inset ring-slate-300 focus-within:ring-2 focus-within:ring-inset focus-within:ring-indigo-600 sm:max-w-md">
                  <input
                    type="number"
                    name="slotNumber"
                    id="slotNumber"
                    min="1"
                    max="10"
                    className="block flex-1 border-0 bg-transparent py-1.5 pl-1 text-slate-900 placeholder:text-slate-400 focus:ring-0 sm:text-sm sm:leading-6"
                    placeholder="Enter slot number (1-10)"
                    value={form.slotNumber}
                    onChange={(e) => updateForm({ slotNumber: e.target.value })}
                    required
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
        <input
          type="submit"
          value={isNew ? "Create Patient Record" : "Update Patient Record"}
          className="inline-flex items-center justify-center whitespace-nowrap text-md font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-slate-100 hover:text-accent-foreground h-9 rounded-md px-3 cursor-pointer mt-4"
        />
      </form>
    </>
  );
}
