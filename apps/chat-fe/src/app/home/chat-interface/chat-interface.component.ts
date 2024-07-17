import { AfterViewInit, Component, ElementRef, OnInit, ViewChild } from "@angular/core";
import { ApiClientService } from "@flair-labs-chatbot/frontend/utils/api-client";

type ChatMessage = {
    type: string;
    content: string;
};

@Component({
    selector: "app-chat",
    templateUrl: "./chat-interface.component.html",
})
export class ChatInterfaceComponent implements AfterViewInit {
    userChatQuery = "";

    @ViewChild("messagesTemplate") private messagesContiner!: ElementRef;

    messages: ChatMessage[] = [];

    constructor(private apiClientService: ApiClientService) {}

    ngAfterViewInit(): void {
        this.scrollToBottom();
    }

    scrollToBottom(): void {
        try {
            this.messagesContiner.nativeElement.scrollTop = this.messagesContiner.nativeElement.scrollHeight;
        } catch (err) {}
    }

    async onSendClicked(): Promise<void> {
        if (this.userChatQuery?.length === 0) return;

        const chatQuery = this.userChatQuery;
        this.messages.push(
            ...[
                { type: "user", content: chatQuery },
                { type: "system", content: "" },
            ],
        );
        this.userChatQuery = "";
        await this.loadSystemMessage(chatQuery);
    }

    async loadSystemMessage(query: string): Promise<void> {
        const systemResponse = await this.apiClientService.apiClient.fetch_result.query({ query });
        const lastMessageIndex = this.messages.length - 1;
        let lastMessage = this.messages[lastMessageIndex];
        if (lastMessage) {
            lastMessage.content = systemResponse;
        }
        setTimeout(() => this.scrollToBottom(), 0);
    }
}
