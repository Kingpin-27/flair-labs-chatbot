/**
 * This is not a production server yet!
 * This is only a minimal backend to get started.
 */

require("fix-esm").register();

import { appRouter, createContext } from "@flair-labs-chatbot/backend/chatbot-core";
import * as trpcExpress from "@trpc/server/adapters/express";
import axios from "axios";
import cors from "cors";
import express from "express";
import fs from "fs";
import * as path from "path";
import { uploadDirectory, uploadMiddleware } from "./upload-middleware";

const app = express();

app.use(cors({ origin: ["http://localhost:4200", "http://127.0.0.1:4200"], credentials: false }));

app.use((req, _res, next) => {
    console.log("🚀", req.method, req.path, req.body ?? req.query);
    next();
});

app.use("/assets", express.static(path.join(__dirname, "assets")));

app.use("/api", trpcExpress.createExpressMiddleware({ router: appRouter, createContext }));

app.use("/uploads", express.static(path.join(__dirname, "/" + uploadDirectory)));

app.post("/upload", uploadMiddleware, async (req, res) => {
    const files = req.files as any;
    if (!files) {
        res.status(500).json({ message: "No file to upload" });
        return;
    }
    files.forEach((file) => {
        const filePath = `${uploadDirectory}/${file.filename}`;
        fs.rename(file.path, filePath, (err) => {
            if (err) {
                return res.status(500).json({ error: "Failed to store the file" });
            }
        });
    });

    try {
        const uploadResult = await axios.get("http://localhost:4444/load_docs");
        console.log("Upload Result:", uploadResult.data);
    } catch (error) {
        let errorMessage = error.message;
        if (axios.isAxiosError(error)) {
            errorMessage = error.response?.data || error.message;
        }
        console.error("Error uploading docs to vector db:", errorMessage);
    }

    res.status(200).json({ message: "File upload successful" });
});

const port = process.env.PORT || 3333;
const server = app.listen(port, () => {
    console.log(`Listening at http://localhost:${port}`);
});
server.on("error", console.error);
