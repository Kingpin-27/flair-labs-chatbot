import { Injectable } from "@angular/core";
import { HttpClient } from "@angular/common/http";

@Injectable({
    providedIn: "root",
})
export class FileUploadService {
    private uploadUrl = "http://localhost:3333/upload";

    constructor(private readonly http: HttpClient) {}

    uploadFiles(files: FormData) {
        return this.http.post<any>(this.uploadUrl, files);
    }
}
