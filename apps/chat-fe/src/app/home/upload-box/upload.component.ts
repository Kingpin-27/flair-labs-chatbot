import { Component, ElementRef, EventEmitter, Output, ViewChild } from "@angular/core";
import { FileUploadService } from "@flair-labs-chatbot/frontend/utils/api-client";

type FileListOptions = {
    [key: string]: any;
};

const constants = {
    SIZE_1_KB: 1024,
    SIZE_1_MB: 1048576,
    SIZE_20_MB: 20_971_520,
};

@Component({
    selector: "app-upload",
    styleUrl: "./upload.component.scss",
    templateUrl: "./upload.component.html",
})
export class UploadComponent {
    @Output() public areFilesUploaded = new EventEmitter();

    filesDetails: FileListOptions = {};
    fileUploadLoading = false;

    @ViewChild("fileTemplate") fileTempl!: ElementRef;
    @ViewChild("imageTemplate") imageTempl!: ElementRef;
    @ViewChild("empty") empty!: ElementRef;
    @ViewChild("hiddenInput") hidden!: ElementRef;
    @ViewChild("gallery") gallery!: ElementRef;
    @ViewChild("overlay") overlay!: ElementRef;

    constructor(private readonly fileUploadService: FileUploadService) {}

    onSubmitClick(): void {
        if (Object.keys(this.filesDetails).length === 0) return;

        this.fileUploadLoading = true;
        const formData = new FormData();
        Object.values(this.filesDetails).forEach((file) => formData.append("files", file, file.name));

        this.fileUploadService.uploadFiles(formData).subscribe({
            next: () => this.areFilesUploaded.emit(true),
            error: (err) => {
                console.error("Error Uploading File", err);
                this.fileUploadLoading = false;
            },
        });
    }

    // use to check if a file is being dragged
    private hasFiles = ({ dataTransfer: { types = [] } }) => (types as any).indexOf("Files") > -1;

    // check if file is of type image and prepend the initialied
    // template to the target element
    private addFile(target: any, file: any): void {
        const isImage = file.type.match("image.*");
        const objectURL = URL.createObjectURL(file);

        const clone = isImage
            ? this.imageTempl?.nativeElement.content.cloneNode(true)
            : this.fileTempl?.nativeElement.content.cloneNode(true);

        const fileName = file.name.length > 35 ? file.name.slice(0, 35) + "..." : file.name;

        clone.querySelector("h1").textContent = fileName;
        clone.querySelector("li").id = objectURL;
        clone.querySelector(".delete").dataset.target = objectURL;
        clone.querySelector(".size").textContent =
            file.size > constants.SIZE_1_KB
                ? file.size > constants.SIZE_1_MB
                    ? Math.round(file.size / constants.SIZE_1_MB) + "mb"
                    : Math.round(file.size / constants.SIZE_1_KB) + "kb"
                : file.size + "b";

        isImage &&
            Object.assign(clone.querySelector("img"), {
                src: objectURL,
                alt: fileName,
            });

        this.empty?.nativeElement.classList.add("hidden");
        target.nativeElement.prepend(clone);

        this.filesDetails[objectURL] = file;
    }

    onHiddenChange(event: any): void {
        for (const file of event.target?.files) {
            this.addFile(this.gallery, file);
        }
    }

    // use to drag dragenter and dragleave events.
    // this is to know if the outermost parent is dragged over
    // without issues due to drag events on its children
    counter = 0;

    // reset counter and append file to gallery when file is dropped
    dropHandler(event: any): void {
        event.preventDefault();
        const oversizeFilesList = [];
        console.log("Before adding files: ", event);
        for (const file of event.dataTransfer.files) {
            if (file.size > constants.SIZE_20_MB) {
                oversizeFilesList.push(file.name);
                this.overlay.nativeElement.classList.remove("disabled-area");
                this.counter = 0;
                continue;
            }
            this.addFile(this.gallery, file);
            this.overlay.nativeElement.classList.remove("disabled-area");
            this.counter = 0;
        }
        if (oversizeFilesList.length > 0) {
            alert(`Files that have exceeded 20MB size limit:\n` + oversizeFilesList);
        }
    }

    // only react to actual files being dragged
    dragEnterHandler(event: any): void {
        event.preventDefault();
        if (!this.hasFiles(event)) {
            return;
        }
        this.counter += 1;
        this.overlay.nativeElement.classList.add("disabled-area");
    }

    dragLeaveHandler(): void {
        this.counter -= 1;
        if (this.counter < 1) {
            this.overlay.nativeElement.classList.remove("disabled-area");
        }
    }

    dragOverHandler(event: any): void {
        if (this.hasFiles(event)) {
            event.preventDefault();
        }
    }

    // event delegation to caputre delete events
    // fron the waste buckets in the file preview cards
    onGalleryClick(event: any): void {
        const { target } = event;
        if (target.classList.contains("delete")) {
            const objectURL = target.dataset.target;
            (document.getElementById(objectURL) as any)?.remove(objectURL);
            this.gallery.nativeElement.children.length === 1 && this.empty.nativeElement.classList.remove("hidden");
            delete this.filesDetails[objectURL];
        }
    }

    // clear entire selection
    onCancelClick(): void {
        while (this.gallery.nativeElement.children.length > 0) {
            this.gallery.nativeElement.lastChild.remove();
        }
        this.filesDetails = {};
        this.empty.nativeElement.classList.remove("hidden");
        this.gallery.nativeElement.append(this.empty.nativeElement);
    }
}
