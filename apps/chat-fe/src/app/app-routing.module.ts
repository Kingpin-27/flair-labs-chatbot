import { Route, RouterModule } from "@angular/router";
import { HomeComponent } from "./home/home.component";
import { NgModule } from "@angular/core";

export const appRoutes: Route[] = [
    {
        path: "home",
        component: HomeComponent,
    },
    {
        path: "**",
        redirectTo: "home",
    },
];

@NgModule({
    imports: [RouterModule.forRoot(appRoutes)],
    exports: [RouterModule],
})
export class AppRoutingModule {}
