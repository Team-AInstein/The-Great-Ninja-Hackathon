import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {

    setSelectedFiles(Array.from(e.target.files));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      if (selectedFiles.length !== 4) {
        throw new Error('Please select exactly 4 images');
      }

      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append('images', file);
      });
  
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Server error occurred');
      }

      const data = await response.json();
      console.log("Raw response data:", data);
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      setResult({ error: error.message || 'An error occurred while processing the images' });
    } finally {
      setLoading(false);
    }
};

  return (
    <div className="App">
      <h1>🚗 AI Based Traffic Management</h1>
      <hr/>

      <div className='main-container'>
        <div className='left'>
          <section id="hero" className="hero">
            <h2>🚦 Optimize Traffic Flow with AI 🤖</h2>
            <p>Enhance your city's traffic management with our smart adaptive system. Our technology optimizes traffic light timings based on real-time data to reduce congestion and improve traffic flow.</p>
          </section>
          <section id="upload" className="upload">
            <h2>📹 Upload Your Traffic Photos</h2>
            <p>Select 4 Images showing different roads at an intersection. Our system will analyze these videos to provide optimized traffic light timings for smoother traffic flow.</p>
            <form onSubmit={handleSubmit}>
              <input 
                type="file" 
                multiple 
                accept="photo/*" 
                onChange={handleFileChange} 
              />
              <br/>
              <button type="submit">Analyze Images</button>
            </form>
          </section>
        </div>

        <section id="result" className="result">
          {!loading && !result && (
            <p className='placeholder'>Optimization results will show here <br/><span>🚦🚦🚦🚦</span></p>
          )}
          {loading && <p className='loader'>Processing Images, it may take a few minutes...</p>}
          

          {result && !result.error && (
            <>
              <h2>✅ Optimization Results</h2>
              <p>Your traffic light timings have been optimized. Here are the recommended green times for each direction:</p>
              <ul>
                <li className={result.ambulance_info.lanes.includes(0) ? 'ambulance-present' : ''}>
                  🚦 North Direction: <span>{result.timings.north}</span> seconds
                  {result.ambulance_info.lanes.includes(0) && 
                    <span className="ambulance-alert"> 🚑 Ambulance Present!</span>
                  }
                </li>
                <li className={result.ambulance_info.lanes.includes(1) ? 'ambulance-present' : ''}>
                  🚦 South Direction: <span>{result.timings.south}</span> seconds
                  {result.ambulance_info.lanes.includes(1) && 
                    <span className="ambulance-alert"> 🚑 Ambulance Present!</span>
                  }
                </li>
                <li className={result.ambulance_info.lanes.includes(2) ? 'ambulance-present' : ''}>
                  🚦 West Direction: <span>{result.timings.west}</span> seconds
                  {result.ambulance_info.lanes.includes(2) && 
                    <span className="ambulance-alert"> 🚑 Ambulance Present!</span>
                  }
                </li>
                <li className={result.ambulance_info.lanes.includes(3) ? 'ambulance-present' : ''}>
                  🚦 East Direction: <span>{result.timings.east}</span> seconds
                  {result.ambulance_info.lanes.includes(3) && 
                    <span className="ambulance-alert"> 🚑 Ambulance Present!</span>
                  }
                </li>
              </ul>
              {result.ambulance_info.detected && (
                <div className="ambulance-warning">
                  ⚠️ Emergency vehicle detected! Priority has been given to lanes with ambulances.
                </div>
              )}
            </>
          )}     
        </section>
        {result && result.error && (
          <div>
            <h2>Error:</h2>
            <p>{result.error}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
