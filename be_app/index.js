const express = require("express");
const cors = require("cors");

const app = express();

// Enable CORS
app.use(cors());
app.use(express.json()); // Middleware to parse JSON

app.post("/ask", async (req, res) => {
    try {
        const response = await fetch("http://127.0.0.1:5000/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: req.body.question, topic: req.body.topic }),
        });

        if (!response.ok) {
            throw new Error(`Flask API returned status: ${response.status}`);
        }

        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error("Error communicating with Flask API:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

// Start the server
const PORT = 5001; // Ensure it's different from Flask
app.listen(PORT, () => console.log(`Express server running on port ${PORT}`));
