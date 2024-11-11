// console.log("Hello world")
// server/server.js
import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import axios from 'axios';

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({
    origin: 'http://localhost:5173', 
    methods: ['GET', 'POST'], // Allow only React app running on localhost:3000
}));
app.use(bodyParser.json());
app.post('/predict', async (req, res) => {
    try {
        console.log("upto there");
        const response = await axios.post('http://127.0.0.1:5000/api/predict', req.body);
        console.log("upto here");
        res.json(response.data);
    } catch (error) {
        res.status(500).send(error.message);
    }
});
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
