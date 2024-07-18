import { CommonModule } from "@angular/common";
import { NgModule } from "@angular/core";
import { BrowserModule } from "@angular/platform-browser";
import { RouterModule } from "@angular/router";
import { ApiClientService, FileUploadService } from "@flair-labs-chatbot/frontend/utils/api-client";
import { AppRoutingModule } from "./app-routing.module";
import { AppComponent } from "./app.component";
import { ChatInterfaceComponent } from "./home/chat-interface/chat-interface.component";
import { HomeComponent } from "./home/home.component";
import { UploadComponent } from "./home/upload-box/upload.component";
import { FormsModule } from "@angular/forms";
import { provideHttpClient, withInterceptorsFromDi } from "@angular/common/http";
import { BrowserAnimationsModule, provideAnimations } from "@angular/platform-browser/animations";

@NgModule({
    declarations: [AppComponent, HomeComponent, UploadComponent, ChatInterfaceComponent],
    imports: [RouterModule, CommonModule, AppRoutingModule, BrowserModule, FormsModule, BrowserAnimationsModule],
    providers: [ApiClientService, FileUploadService, provideAnimations(), provideHttpClient(withInterceptorsFromDi())],
    bootstrap: [AppComponent],
})
export class AppModule {}
