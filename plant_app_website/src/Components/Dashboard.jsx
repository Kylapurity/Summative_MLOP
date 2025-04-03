import React, { useState, useRef, useEffect } from 'react';

// Mock data for plant disease trends (unchanged)
const trendData = [
  { id: 1, disease: 'Leaf Blight', occurrences: 235, trend: 'increasing', severity: 'high' },
  { id: 2, disease: 'Powdery Mildew', occurrences: 189, trend: 'stable', severity: 'medium' },
  { id: 3, disease: 'Root Rot', occurrences: 156, trend: 'decreasing', severity: 'high' },
  { id: 4, disease: 'Bacterial Spot', occurrences: 120, trend: 'increasing', severity: 'medium' },
  { id: 5, disease: 'Rust', occurrences: 98, trend: 'stable', severity: 'low' },
];

// Helper function to get recommendations based on disease name (updated to match backend CLASS_NAMES)
const getRecommendations = (diseaseName) => {
  const defaultRecommendations = [
    'Remove and destroy infected leaves',
    'Ensure proper air circulation around plants',
    'Avoid overhead watering',
  ];

  const diseaseRecommendations = {
    'Apple_Apple_scab': [
      'Remove and destroy fallen leaves',
      'Apply fungicide during the growing season',
      'Prune trees to improve air circulation',
      'Consider resistant apple varieties',
    ],
    'Apple_Black_rot': [
      'Prune out dead or diseased branches',
      'Apply fungicide during the growing season',
      'Remove mummified fruits from the tree and ground',
      'Ensure proper spacing between trees',
    ],
    'Apple_Cedar_apple_rust': [
      'Remove nearby cedar trees if possible',
      'Apply fungicide during the growing season',
      'Prune trees to improve air circulation',
      'Use resistant apple varieties',
    ],
    'Apple_healthy': [
      'Maintain good tree health through proper watering and fertilization',
      'Regularly inspect for signs of disease',
      'Prune trees to improve air circulation',
      'Apply preventive fungicide if necessary',
    ],
    'Blueberry_healthy': [
      'Maintain good soil pH (4.5-5.5)',
      'Regularly inspect for signs of disease',
      'Ensure proper irrigation and drainage',
      'Apply mulch to retain soil moisture',
    ],
    'Cherry_Powdery_mildew': [
      'Apply fungicide during the growing season',
      'Prune trees to improve air circulation',
      'Remove and destroy infected leaves',
      'Avoid overhead watering',
    ],
    'Cherry_healthy': [
      'Maintain good tree health through proper watering and fertilization',
      'Regularly inspect for signs of disease',
      'Prune trees to improve air circulation',
      'Apply preventive fungicide if necessary',
    ],
    'Corn_Cercospora_leaf_spot_Gray_leaf_spot': [
      'Rotate crops to non-host crops',
      'Apply fungicide during the growing season',
      'Remove and destroy infected plant debris',
      'Use resistant corn varieties',
    ],
    'Corn_Common_rust': [
      'Apply fungicide during the growing season',
      'Remove and destroy infected plant debris',
      'Use resistant corn varieties',
      'Ensure proper spacing for air circulation',
    ],
    'Corn_Northern_Leaf_Blight': [
      'Rotate crops to non-host crops',
      'Apply fungicide during the growing season',
      'Remove and destroy infected plant debris',
      'Use resistant corn varieties',
    ],
    'Corn_healthy': [
      'Maintain good soil fertility',
      'Regularly inspect for signs of disease',
      'Ensure proper irrigation and drainage',
      'Apply preventive fungicide if necessary',
    ],
    'Grape_Black_rot': [
      'Remove mummified berries and infected leaves',
      'Apply fungicide before bloom and after',
      'Ensure good canopy management and air circulation',
      'Use appropriate fungicide rotation',
    ],
    'Grape_Esca_Black_Measles': [
      'Prune out infected wood and destroy it',
      'Apply fungicide during the dormant season',
      'Ensure proper vine spacing for air circulation',
      'Avoid wounding the vine during pruning',
    ],
    'Grape_Leaf_blight_Isariopsis_Leaf_Spot': [
      'Apply fungicide during the growing season',
      'Remove and destroy infected leaves',
      'Ensure good canopy management and air circulation',
      'Use resistant grape varieties',
    ],
    'Grape_healthy': [
      'Maintain good vine health through proper watering and fertilization',
      'Regularly inspect for signs of disease',
      'Prune vines to improve air circulation',
      'Apply preventive fungicide if necessary',
    ],
    'Orange_Haunglongbing_Citrus_greening': [
      'Remove and destroy infected trees',
      'Control psyllid populations with insecticides',
      'Use disease-free planting material',
      'Regularly inspect for signs of disease',
    ],
    'Peach_Bacterial_spot': [
      'Apply copper-based bactericide during the growing season',
      'Prune trees to improve air circulation',
      'Remove and destroy infected leaves and fruit',
      'Use resistant peach varieties',
    ],
    'Peach_healthy': [
      'Maintain good tree health through proper watering and fertilization',
      'Regularly inspect for signs of disease',
      'Prune trees to improve air circulation',
      'Apply preventive fungicide if necessary',
    ],
    'Pepper_bell_Bacterial_spot': [
      'Apply copper-based bactericide during the growing season',
      'Remove and destroy infected plants',
      'Avoid overhead watering',
      'Use disease-free seeds and transplants',
    ],
    'Pepper_bell_healthy': [
      'Maintain good plant health through proper watering and fertilization',
      'Regularly inspect for signs of disease',
      'Ensure proper spacing for air circulation',
      'Apply preventive fungicide if necessary',
    ],
    'Potato_Early_blight': [
      'Rotate crops to non-host crops',
      'Apply fungicide during the growing season',
      'Remove and destroy infected plant debris',
      'Use resistant potato varieties',
    ],
    'Potato_Late_blight': [
      'Rotate crops to non-host crops',
      'Apply fungicide during the growing season',
      'Remove and destroy infected plant debris',
      'Use resistant potato varieties',
    ],
    'Potato_healthy': [
      'Maintain good soil fertility',
      'Regularly inspect for signs of disease',
      'Ensure proper irrigation and drainage',
      'Apply preventive fungicide if necessary',
    ],
    'Raspberry_healthy': [
      'Maintain good plant health through proper watering and fertilization',
      'Regularly inspect for signs of disease',
      'Prune plants to improve air circulation',
      'Apply preventive fungicide if necessary',
    ],
    'Soybean_healthy': [
      'Maintain good soil fertility',
      'Regularly inspect for signs of disease',
      'Ensure proper irrigation and drainage',
      'Apply preventive fungicide if necessary',
    ],
    'Squash_Powdery_mildew': [
      'Apply fungicide during the growing season',
      'Remove and destroy infected leaves',
      'Ensure proper spacing for air circulation',
      'Avoid overhead watering',
    ],
    'Strawberry_Leaf_scorch': [
      'Apply fungicide during the growing season',
      'Remove and destroy infected leaves',
      'Ensure proper spacing for air circulation',
      'Use resistant strawberry varieties',
    ],
    'Strawberry_healthy': [
      'Maintain good plant health through proper watering and fertilization',
      'Regularly inspect for signs of disease',
      'Ensure proper spacing for air circulation',
      'Apply preventive fungicide if necessary',
    ],
    'Tomato_Bacterial_spot': [
      'Apply copper-based bactericide during the growing season',
      'Remove and destroy infected plants',
      'Avoid overhead watering',
      'Use disease-free seeds and transplants',
    ],
    'Tomato_Early_blight': [
      'Rotate crops to non-host crops',
      'Apply fungicide during the growing season',
      'Remove and destroy infected plant debris',
      'Use resistant tomato varieties',
    ],
    'Tomato_Late_blight': [
      'Rotate crops to non-host crops',
      'Apply fungicide during the growing season',
      'Remove and destroy infected plant debris',
      'Use resistant tomato varieties',
    ],
    'Tomato_Leaf_Mold': [
      'Apply fungicide during the growing season',
      'Remove and destroy infected leaves',
      'Ensure proper spacing for air circulation',
      'Avoid overhead watering',
    ],
    'Tomato_Septoria_leaf_spot': [
      'Rotate crops to non-host crops',
      'Apply fungicide during the growing season',
      'Remove and destroy infected plant debris',
      'Use resistant tomato varieties',
    ],
    'Tomato_Spider_mites_Two_spotted_spider_mite': [
      'Apply miticide during the growing season',
      'Remove and destroy heavily infested leaves',
      'Ensure proper irrigation to reduce stress',
      'Introduce natural predators like ladybugs',
    ],
    'Tomato_Target_Spot': [
      'Rotate crops to non-host crops',
      'Apply fungicide during the growing season',
      'Remove and destroy infected plant debris',
      'Use resistant tomato varieties',
    ],
    'Tomato_Yellow_Leaf_Curl_Virus': [
      'Control whitefly populations with insecticides',
      'Remove and destroy infected plants',
      'Use disease-free seeds and transplants',
      'Plant resistant tomato varieties',
    ],
    'Tomato_mosaic_virus': [
      'Remove and destroy infected plants',
      'Control aphid populations with insecticides',
      'Use disease-free seeds and transplants',
      'Plant resistant tomato varieties',
    ],
    'Tomato_healthy': [
      'Maintain good plant health through proper watering and fertilization',
      'Regularly inspect for signs of disease',
      'Ensure proper spacing for air circulation',
      'Apply preventive fungicide if necessary',
    ],
  };

  return diseaseRecommendations[diseaseName] || defaultRecommendations;
};

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('trends');
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [scanHistory, setScanHistory] = useState([]);
  const [retrainingHistory, setRetrainingHistory] = useState([]);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);
  const [error, setError] = useState(null);
  const [trainingFiles, setTrainingFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainingBatch, setRetrainingBatch] = useState(null);
  const fileInputRef = useRef(null);
  const trainingFileInputRef = useRef(null);

  // Backend API URL
  const API_URL = "https://summativemlop-production.up.railway.app"; // Update to your backend URL if deployed

  // Load history from localStorage
  useEffect(() => {
    const loadHistory = () => {
      try {
        const savedScanHistory = localStorage.getItem('scanHistory');
        const savedRetrainingHistory = localStorage.getItem('retrainingHistory');
        if (savedScanHistory) setScanHistory(JSON.parse(savedScanHistory));
        if (savedRetrainingHistory) setRetrainingHistory(JSON.parse(savedRetrainingHistory));
      } catch (err) {
        console.error('Error loading history from localStorage:', err);
        setError('Failed to load history.');
      }
    };
    loadHistory();
  }, []);

  // Save history to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('scanHistory', JSON.stringify(scanHistory));
      localStorage.setItem('retrainingHistory', JSON.stringify(retrainingHistory));
    } catch (err) {
      console.error('Error saving history to localStorage:', err);
      setError('Failed to save history.');
    }
  }, [scanHistory, retrainingHistory]);

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'increasing': return '↑';
      case 'decreasing': return '↓';
      default: return '→';
    }
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case 'increasing': return 'text-red-500';
      case 'decreasing': return 'text-green-500';
      default: return 'text-yellow-500';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-green-100 text-green-800';
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnalysisResult(null);
      setError(null);
    }
  };

  const handleAnalyzeClick = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      console.log('Prediction Response:', data);

      const rawDiseaseName = data.prediction;
      let displayDiseaseName = rawDiseaseName.split('_').slice(1).join(' ');
      if (displayDiseaseName.toLowerCase() === 'healthy') {
        displayDiseaseName = 'Healthy - No Disease Detected';
      }

      const result = {
        disease: displayDiseaseName,
        rawDiseaseName: rawDiseaseName,
        confidence: Math.round(data.confidence * 100),
        timestamp: data.timestamp,
        recommendations: getRecommendations(rawDiseaseName),
      };

      setAnalysisResult(result);

      const newHistoryItem = {
        id: Date.now(),
        imageUrl: previewUrl,
        date: new Date().toISOString().split('T')[0],
        disease: displayDiseaseName,
        rawDiseaseName: rawDiseaseName,
        confidence: result.confidence,
        recommendations: result.recommendations,
      };
      setScanHistory([newHistoryItem, ...scanHistory]);
    } catch (err) {
      console.error('Prediction Error:', err);
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTrainingFileChange = (e) => {
    setTrainingFiles([...e.target.files]);
    setRetrainingBatch(null); // Reset batch when new files are selected
  };

  const handleUploadTrainingFiles = async (e) => {
    e.preventDefault();
    if (!trainingFiles.length) {
      setError('Please select a ZIP file for retraining');
      return;
    }

    if (!trainingFiles.every(file => file.name.endsWith('.zip'))) {
      setError('Please upload only ZIP files');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      trainingFiles.forEach((file) => formData.append('files', file));

      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      console.log('Upload Response:', data);
      setRetrainingBatch(data.retraining_batch);
      alert('Files uploaded successfully! Proceed to retrain.');
    } catch (err) {
      console.error('Upload Error:', err);
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRetrainSubmit = async (e) => {
    e.preventDefault();
    if (!retrainingBatch) {
      setError('Please upload training files first');
      return;
    }

    setIsRetraining(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          retraining_batch: retrainingBatch,
          learning_rate: 0.0001,
          epochs: 10,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Retraining failed');
      }

      const data = await response.json();
      console.log('Retraining Response:', data);

      const numClasses = data.classes_detected.length;
      const newRetraining = {
        id: Date.now(),
        text: `Retrained model with ${data.images_used} images across ${numClasses} classes`,
        training_accuracy: data.metrics.training_accuracy,
        validation_accuracy: data.metrics.validation_accuracy,
        class_metrics: data.metrics.class_metrics,
        visualizations: data.visualization_files,
        date: data.timestamp,
        retraining_batch: data.retraining_batch,
      };
      setRetrainingHistory([newRetraining, ...retrainingHistory]);
      setTrainingFiles([]);
      setRetrainingBatch(null);
      trainingFileInputRef.current.value = null;
      alert('Retraining completed successfully!');
    } catch (err) {
      console.error('Retraining Error:', err);
      setError(err.message);
    } finally {
      setIsRetraining(false);
    }
  };

  const viewHistoryDetails = (item) => setSelectedHistoryItem(item);
  const closeHistoryDetails = () => setSelectedHistoryItem(null);

  const handleNewScan = () => {
    setSelectedImage(null);
    setPreviewUrl('');
    setAnalysisResult(null);
    setError(null);
    setActiveTab('upload');
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-green-600">Plant Disease Detector</h1>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="mb-4 border-b border-gray-200">
            <ul className="flex flex-wrap -mb-px">
              <li className="mr-2">
                <button
                  className={`inline-block py-4 px-4 text-sm font-medium ${activeTab === 'trends' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('trends')}
                >
                  Disease Trends
                </button>
              </li>
              <li className="mr-2">
                <button
                  className={`inline-block py-4 px-4 text-sm font-medium ${activeTab === 'upload' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('upload')}
                >
                  Predict Disease
                </button>
              </li>
              <li className="mr-2">
                <button
                  className={`inline-block py-4 px-4 text-sm font-medium ${activeTab === 'history' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('history')}
                >
                  My History
                </button>
              </li>
              <li className="mr-2">
                <button
                  className={`inline-block py-4 px-4 text-sm font-medium ${activeTab === 'retrain' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('retrain')}
                >
                  Retrain Model
                </button>
              </li>
              <li className="mr-2">
                <button
                  className={`inline-block py-4 px-4 text-sm font-medium ${activeTab === 'retraining_history' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('retraining_history')}
                >
                  Retraining History
                </button>
              </li>
              <li className="mr-2">
                <button
                  className={`inline-block py-4 px-4 text-sm font-medium ${activeTab === 'retraining_visualizations' ? 'text-green-600 border-b-2 border-green-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('retraining_visualizations')}
                >
                  Visualizations
                </button>
              </li>
            </ul>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
              {error}
              <button
                className="ml-4 bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-4 rounded"
                onClick={() => setError(null)}
              >
                Dismiss
              </button>
            </div>
          )}

          {activeTab === 'trends' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Current Plant Disease Trends</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Disease</th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Occurrences</th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trend</th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Severity</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {trendData.map((item) => (
                      <tr key={item.id}>
                        <td className="px-6 py-4 whitespace-nowrap"><div className="text-sm font-medium text-gray-900">{item.disease}</div></td>
                        <td className="px-6 py-4 whitespace-nowrap"><div className="text-sm text-gray-500">{item.occurrences}</div></td>
                        <td className="px-6 py-4 whitespace-nowrap"><div className={`text-sm ${getTrendColor(item.trend)}`}>{item.trend} {getTrendIcon(item.trend)}</div></td>
                        <td className="px-6 py-4 whitespace-nowrap"><span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getSeverityColor(item.severity)}`}>{item.severity}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'upload' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Upload Plant Image for Disease Detection</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                <div className="mb-6">
                  <p className="text-gray-600 mb-4">Upload a clear image of the plant leaf or affected area.</p>
                  {!previewUrl ? (
                    <div className="flex flex-col items-center">
                      <label
                        htmlFor="dropzone-file"
                        className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
                      >
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                          <svg className="w-10 h-10 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                          </svg>
                          <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                          <p className="text-xs text-gray-500">PNG, JPG, or JPEG (MAX. 10MB)</p>
                        </div>
                        <input
                          id="dropzone-file"
                          type="file"
                          className="hidden"
                          accept="image/*"
                          ref={fileInputRef}
                          onChange={handleImageChange}
                        />
                      </label>
                      <button
                        type="button"
                        className="mt-4 bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                        onClick={() => {
                          if (!selectedImage) {
                            alert("Please upload an image first");
                            fileInputRef.current.click();
                          } else {
                            handleAnalyzeClick();
                          }
                        }}
                      >
                        Predict Plant Disease
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <div className="relative mb-4 w-full max-w-lg">
                        <img src={previewUrl} alt="Plant preview" className="rounded-lg shadow-md object-contain max-h-64 mx-auto" />
                        <button
                          onClick={() => { setSelectedImage(null); setPreviewUrl(''); setAnalysisResult(null); setError(null); }}
                          className="absolute top-2 right-2 bg-red-500 hover:bg-red-700 text-white rounded-full p-1"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                      {!analysisResult && !error && (
                        <button
                          type="button"
                          className={`bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded ${isAnalyzing ? 'opacity-75 cursor-not-allowed' : ''}`}
                          onClick={handleAnalyzeClick}
                          disabled={isAnalyzing}
                        >
                          {isAnalyzing ? 'Analyzing...' : 'Predict Disease'}
                        </button>
                      )}
                    </div>
                  )}
                </div>
                {analysisResult && (
                  <div className="mt-8 border-t pt-6">
                    <h3 className="text-xl font-bold mb-4">Analysis Results</h3>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <div className="mb-4">
                        <span className="font-semibold">Detected Disease:</span>{' '}
                        <span className={analysisResult.disease.includes('Healthy') ? 'text-green-600' : 'text-red-600'}>{analysisResult.disease}</span>
                      </div>
                      <div className="mb-4">
                        <span className="font-semibold">Confidence:</span> {analysisResult.confidence}%
                      </div>
                      <div className="mb-4">
                        <span className="font-semibold">Timestamp:</span> {new Date(analysisResult.timestamp).toLocaleString()}
                      </div>
                      <div className="mb-6">
                        <h4 className="font-semibold mb-2">Recommendations:</h4>
                        <ul className="list-disc pl-5">
                          {analysisResult.recommendations.map((rec, index) => (
                            <li key={index} className="text-gray-700 mb-1">{rec}</li>
                          ))}
                        </ul>
                      </div>
                      <div className="flex justify-center space-x-4">
                        <button
                          type="button"
                          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded"
                          onClick={() => setActiveTab('history')}
                        >
                          View History
                        </button>
                        <button
                          type="button"
                          className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                          onClick={handleNewScan}
                        >
                          Scan Another Plant
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'history' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Your Detection History</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                {scanHistory.length === 0 ? (
                  <div>
                    <p className="text-gray-600 mb-4 text-center">You haven't analyzed any plants yet.</p>
                    <div className="flex justify-center">
                      <button
                        type="button"
                        className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                        onClick={handleNewScan}
                      >
                        Upload and Predict
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    {selectedHistoryItem ? (
                      <div className="bg-white p-4 rounded-lg">
                        <div className="flex justify-between items-start mb-4">
                          <h3 className="text-xl font-bold">{selectedHistoryItem.disease}</h3>
                          <button onClick={closeHistoryDetails} className="text-gray-500 hover:text-gray-700">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                        <div className="flex flex-col md:flex-row md:space-x-6">
                          <div className="md:w-1/3 mb-4 md:mb-0">
                            <img src={selectedHistoryItem.imageUrl} alt={selectedHistoryItem.disease} className="rounded-lg shadow-md w-full h-auto" />
                            <p className="text-gray-500 mt-2 text-sm">Scanned on {selectedHistoryItem.date}</p>
                          </div>
                          <div className="md:w-2/3">
                            <div className="mb-4">
                              <span className="font-semibold">Confidence:</span> {selectedHistoryItem.confidence}%
                            </div>
                            <div>
                              <h4 className="font-semibold mb-2">Recommendations:</h4>
                              <ul className="list-disc pl-5">
                                {selectedHistoryItem.recommendations.map((rec, index) => (
                                  <li key={index} className="text-gray-700 mb-1">{rec}</li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </div>
                        <div className="mt-6 flex justify-center">
                          <button
                            type="button"
                            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                            onClick={handleNewScan}
                          >
                            Scan New Plant
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          {scanHistory.map((item) => (
                            <div
                              key={item.id}
                              className="bg-white overflow-hidden shadow-md rounded-lg hover:shadow-lg transition-shadow cursor-pointer"
                              onClick={() => viewHistoryDetails(item)}
                            >
                              <div className="relative h-48 bg-gray-200">
                                <img src={item.imageUrl} alt={item.disease} className="w-full h-full object-cover" />
                              </div>
                              <div className="p-4">
                                <p className="text-gray-500 text-sm">{item.date}</p>
                                <h3 className={`font-bold mt-1 ${item.disease.includes('Healthy') ? 'text-green-600' : 'text-red-600'}`}>{item.disease}</h3>
                                <p className="text-gray-700 mt-1">Confidence: {item.confidence}%</p>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="mt-6 flex justify-center">
                          <button
                            type="button"
                            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
                            onClick={handleNewScan}
                          >
                            Scan New Plant
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'retrain' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Retrain Model</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                <div className="mb-6">
                  <p className="text-gray-600 mb-4">Upload a ZIP file containing train/ and val/ folders with images to retrain the model.</p>
                  <form onSubmit={handleUploadTrainingFiles}>
                    <div className="mb-4">
                      <label className="block text-gray-700 mb-2">Upload Training ZIP File</label>
                      <input
                        type="file"
                        ref={trainingFileInputRef}
                        onChange={handleTrainingFileChange}
                        multiple
                        className="w-full p-2 border rounded"
                        accept=".zip"
                      />
                      <p className="text-xs text-gray-500 mt-1">ZIP file with train/ and val/ folders.</p>
                    </div>
                    <button
                      type="submit"
                      disabled={isUploading || isRetraining}
                      className={`bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded ${isUploading || isRetraining ? 'opacity-75 cursor-not-allowed' : ''}`}
                    >
                      {isUploading ? 'Uploading...' : 'Upload Files'}
                    </button>
                  </form>
                  {retrainingBatch && (
                    <div className="mt-4">
                      <p className="text-gray-700 mb-2">Files uploaded with batch ID: <strong>{retrainingBatch}</strong></p>
                      <form onSubmit={handleRetrainSubmit}>
                        <button
                          type="submit"
                          disabled={isRetraining}
                          className={`bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded ${isRetraining ? 'opacity-75 cursor-not-allowed' : ''}`}
                        >
                          {isRetraining ? 'Retraining...' : 'Retrain Model'}
                        </button>
                      </form>
                    </div>
                  )}
                  {trainingFiles.length > 0 && !retrainingBatch && (
                    <p className="mt-2 text-sm text-gray-600">{trainingFiles.length} file(s) selected</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'retraining_history' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Retraining History</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                {retrainingHistory.length === 0 ? (
                  <div>
                    <p className="text-gray-600 mb-4">You haven't retrained the model yet.</p>
                    <button
                      type="button"
                      className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded"
                      onClick={() => setActiveTab('retrain')}
                    >
                      Retrain Model
                    </button>
                  </div>
                ) : (
                  <>
                    <ul className="divide-y divide-gray-200">
                      {retrainingHistory.map((item) => (
                        <li
                          key={item.id}
                          className="py-4 cursor-pointer hover:bg-gray-50"
                          onClick={() => viewHistoryDetails(item)}
                        >
                          <p className="text-gray-700">Date: {new Date(item.date).toLocaleDateString()} | {item.text}</p>
                        </li>
                      ))}
                    </ul>
                    {selectedHistoryItem && (
                      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                        <div className="flex justify-between items-start mb-4">
                          <h3 className="text-lg font-semibold">{selectedHistoryItem.text}</h3>
                          <button onClick={closeHistoryDetails} className="text-gray-500 hover:text-gray-700">
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                        <div className="mb-4">
                          <p><span className="font-semibold">Training Accuracy:</span> {selectedHistoryItem.training_accuracy ? `${(selectedHistoryItem.training_accuracy * 100).toFixed(2)}%` : 'N/A'}</p>
                          <p><span className="font-semibold">Validation Accuracy:</span> {selectedHistoryItem.validation_accuracy ? `${(selectedHistoryItem.validation_accuracy * 100).toFixed(2)}%` : 'N/A'}</p>
                          <p><span className="font-semibold">Batch ID:</span> {selectedHistoryItem.retraining_batch}</p>
                        </div>
                        {selectedHistoryItem.class_metrics && (
                          <div className="mt-4">
                            <h4 className="font-semibold mb-2">Class Metrics</h4>
                            <ul className="list-disc pl-5">
                              {Object.entries(selectedHistoryItem.class_metrics).map(([className, metrics]) => (
                                <li key={className} className="text-gray-700 mb-1">
                                  {className}: Precision: {Math.round(metrics.precision * 100)}%, Recall: {Math.round(metrics.recall * 100)}%, F1: {Math.round(metrics['f1-score'] * 100)}%
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        <p className="mt-4 text-gray-600">
                          Visualizations available in the{' '}
                          <button className="text-blue-500 hover:underline" onClick={() => setActiveTab('retraining_visualizations')}>
                            Visualizations
                          </button> tab.
                        </p>
                        <div className="mt-6 flex justify-center">
                          <button
                            type="button"
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded"
                            onClick={() => setActiveTab('retrain')}
                          >
                            Retrain Again
                          </button>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {activeTab === 'retraining_visualizations' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Visualizations</h2>
              <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                {retrainingHistory.length === 0 ? (
                  <div>
                    <p className="text-gray-600 mb-4">No visualizations available yet.</p>
                    <button
                      type="button"
                      className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded"
                      onClick={() => setActiveTab('retrain')}
                    >
                      Retrain Model
                    </button>
                  </div>
                ) : (
                  <div className="space-y-8">
                    {retrainingHistory.map((session) => (
                      <div key={session.id}>
                        <h3 className="text-lg font-semibold mb-2">Retraining Session: {new Date(session.date).toLocaleDateString()} (Batch: {session.retraining_batch})</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {session.visualizations?.confusion_matrix && (
                            <div>
                              <h4 className="font-medium">Confusion Matrix</h4>
                              <img src={session.visualizations.confusion_matrix} alt="Confusion Matrix" className="mt-2 rounded-lg shadow-md w-full" />
                            </div>
                          )}
                          {session.visualizations?.classification_report && (
                            <div>
                              <h4 className="font-medium">Classification Report</h4>
                              <img src={session.visualizations.classification_report} alt="Classification Report" className="mt-2 rounded-lg shadow-md w-full" />
                            </div>
                          )}
                          {session.visualizations?.loss_plot && (
                            <div>
                              <h4 className="font-medium">Loss Plot</h4>
                              <img src={session.visualizations.loss_plot} alt="Loss Plot" className="mt-2 rounded-lg shadow-md w-full" />
                            </div>
                          )}
                          {session.visualizations?.accuracy_plot && (
                            <div>
                              <h4 className="font-medium">Accuracy Plot</h4>
                              <img src={session.visualizations.accuracy_plot} alt="Accuracy Plot" className="mt-2 rounded-lg shadow-md w-full" />
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;