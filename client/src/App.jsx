import React, { useState } from 'react';
import './index.css';
import Form from './components/Form';

const App = () => {
    return (
        <div className="app bg-slate-800 min-h-screen flex flex-col items-center justify-center">
            <h1 className="text-4xl text-white text-center mb-6 mt-5">Customer Churn Prediction</h1>
            <Form />
        </div>
    );
};

export default App;
