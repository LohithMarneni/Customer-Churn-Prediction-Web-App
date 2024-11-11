import React, { useState } from 'react';
import axios from 'axios';

const Form = () => {
    const [formData, setFormData] = useState({
        CreditScore: '',
        Geography: '',
        Gender: '',
        Age: '',
        Tenure: '',
        Balance: '',
        NumOfProducts: '',
        HasCrCard: '',
        IsActiveMember: '',
        EstimatedSalary: ''
    });
    const [prediction, setPrediction] = useState(null);

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            // Parse formData values to correct types
            const parsedData = {
                ...formData,
                CreditScore: Number(formData.CreditScore),
                Age: Number(formData.Age),
                Tenure: Number(formData.Tenure),
                Balance: parseFloat(formData.Balance),
                NumOfProducts: Number(formData.NumOfProducts),
                HasCrCard: Number(formData.HasCrCard),
                IsActiveMember: Number(formData.IsActiveMember),
                EstimatedSalary: parseFloat(formData.EstimatedSalary),
            };
    
            console.log("Form data sent:", parsedData);
            const response = await axios.post('http://localhost:5000/predict', parsedData);
            console.log("Backend response:", response.data);
            setPrediction(response.data.churn_prediction === 1 ? 'Will Exit' : 'Won\'t Exit');
        } catch (error) {
            console.error('Error making prediction:', error);
        }
    };
    
    

    return (
        <div className="form-container bg-gray-700 p-8 rounded-lg shadow-lg w-full max-w-lg">
            <form onSubmit={handleSubmit} className="space-y-6">
                {Object.keys(formData).map((key) => (
                    <div key={key} className="form-group">
                        <label className="block text-white text-sm font-medium mb-2">{key}</label>
                        <input
                            type="text"
                            name={key}
                            value={formData[key]}
                            onChange={handleChange}
                            required
                            className="w-full p-2 text-gray-800 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
                        />
                    </div>
                ))}
                <button
                    type="submit"
                    className="w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-300"
                >
                    Predict Churn
                </button>
            </form>
            {prediction && (
                <div className="result mt-6 text-center">
                    <h2 className="text-white text-xl font-semibold">Prediction: {prediction}</h2>
                </div>
            )}
        </div>
    );
};

export default Form;
