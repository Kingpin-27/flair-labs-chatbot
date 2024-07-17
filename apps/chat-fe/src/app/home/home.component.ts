import { animate, style, transition, trigger } from "@angular/animations";
import { Component } from "@angular/core";

@Component({
    selector: "app-home",
    templateUrl: "./home.component.html",
    animations: [
        trigger("enterAnimation", [
            transition(":enter", [
                style({ transform: "translateX(100%)", opacity: 0 }),
                animate("500ms", style({ transform: "translateX(0)", opacity: 1 })),
            ]),
            transition(":leave", [
                style({ transform: "translateX(0)", opacity: 1 }),
                animate("500ms", style({ transform: "translateX(100%)", opacity: 0 })),
            ]),
        ]),
    ],
})
export class HomeComponent {
    title = "banner_mgmt";
    isFileUploadCompleted = false;
}
