import React, { Fragment, useState } from "react";
import "./App.css";
import axios from "axios";
import { Card, CardContent } from "@mui/material";
import Button from "@mui/material/Button";

export default function QnAApp() {
    const [question, setQuestion] = useState("");
    const [answer, setAnswer] = useState("");
    const [topic, setTopic] = useState("Machine Learning and Technology");
    const [loading, setLoading] = useState(false);

    const topics = ["Machine Learning & Technology", "Foods", "Kpop & South Korean Culture"];

    const handleAsk = async () => {
        if (!question) return;

        setLoading(true);
        setAnswer(""); // Reset previous answer

        try {
            const response = await axios.post("http://127.0.0.1:5001/ask", { question, topic });

            // Ensure a valid answer is received
            const receivedAnswer = response.data.answer ? response.data.answer.trim() : "No answer found.";
            setAnswer(receivedAnswer);
        } catch (error) {
            console.error("Error fetching answer:", error);
            setAnswer("Failed to fetch answer. Please check your connection.");
        }

        setLoading(false);
    };

    return (
        <Fragment>
            <div className="container p-5">
                <div className="text-center">
                    <h1 className="text-2xl font-bold mb-2">Tiny Knowledgde 🚀</h1>
                    <p className="text-gray-600 mb-4">Ask me a question about Machine Learning, Food, or K-pop! 🙂</p>
                </div>
                <div className="card content-grid">
                    <div className="">
                        <div className="input-content">
                            <h6>Please select your topic</h6>
                            <select 
                                value={topic} 
                                onChange={(e) => setTopic(e.target.value)} 
                                className="p-2 form-select text-center">
                                {topics.map((t) => <option key={t} value={t}>{t}</option>)}
                            </select>
                            <input 
                                type="text" 
                                value={question} 
                                onChange={(e) => setQuestion(e.target.value)}
                                placeholder="Type your question..." 
                                className="p-2 form-control text-center" 
                            />
                            <button className="button-input" onClick={handleAsk} disabled={loading}>{loading ? "Thinking..." : "Ask me 🤩"}</button>
                        </div>
                    </div>
                    <div className="">
                        <div className="result-content">
                            <p className="text-gray-700 font-mono whitespace-pre-wrap">
                                {loading ? "Generating answer..." : answer || "Your answer will appear here..."}
                            </p>
                        </div>
                    </div>
                </div>
                <div className="text-center mt-3">
                    <small>Made with 💖 by Group 66666</small>
                    <br></br>
                    <small>NLP Mini Project | 2025</small>
                </div>
            </div>
        </Fragment>
    );
}
