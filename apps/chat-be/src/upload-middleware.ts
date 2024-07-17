import fs from "fs";
import { fileUploadApiKey } from "libs/common-utils";
import multer from "multer";
import path from "path";

export const uploadDirectory = "uploads";

export const removePreviousFiles = () => {
    fs.readdir(uploadDirectory, (err, files) => {
        if (err) throw err;

        for (const file of files) {
            fs.unlink(path.join(uploadDirectory, file), (err) => {
                if (err) throw err;
            });
        }
    });
};

const storage = multer.diskStorage({
    destination: (_req, _file, cb) => {
        cb(null, uploadDirectory + "/");
    },
    filename: (_req, file, cb) => {
        cb(null, Date.now() + "-" + file.originalname);
    },
});

const upload = multer({ storage: storage });

export const uploadMiddleware = (req, res, next) => {
    removePreviousFiles();
    upload.array(fileUploadApiKey)(req, res, (err) => {
        if (err) {
            return res.status(400).json({ error: err.message });
        }
        next();
    });
};
