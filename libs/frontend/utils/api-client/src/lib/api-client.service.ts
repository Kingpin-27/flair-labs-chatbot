import { Injectable } from "@angular/core";
import { AppRouter } from "@flair-labs-chatbot/backend/chatbot-core";
import { createTRPCProxyClient, httpBatchLink } from "@trpc/client";
import { transformer } from "../../../../../common-utils";

@Injectable({ providedIn: "root" })
export class ApiClientService {
    apiClient = createTRPCProxyClient<AppRouter>({
        transformer,
        links: [httpBatchLink({ url: "http://localhost:3333/api", headers: () => this.getHeaders() })],
    });

    private getHeaders() {
        return {
            test: "header from fe",
        };
    }
}
