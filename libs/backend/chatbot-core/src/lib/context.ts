import { inferAsyncReturnType, initTRPC } from "@trpc/server";
import { CreateExpressContextOptions } from "@trpc/server/adapters/express";
import { transformer } from "../../../../common-utils";

export function createContext(event: CreateExpressContextOptions) {
    return {
        test: event.req.headers["test"] || "empty",
    };
}

export type Context = inferAsyncReturnType<typeof createContext>;

export const trpcServer = initTRPC.context<Context>().create({ transformer });
