import { Activity, User } from "lucide-react";
import clsx from "clsx";

type Message = {
    id: string;
    role: "user" | "assistant";
    content: string | React.ReactNode;
    isExplanation?: boolean;
};

export default function ChatMessage({ message }: { message: Message }) {
    const isUser = message.role === "user";

    return (
        <div className={clsx(
            "flex gap-4 md:gap-5 w-full max-w-[90%] md:max-w-[85%]",
            isUser ? "self-end flex-row-reverse" : "self-start"
        )}>

            {/* Avatar */}
            <div className={clsx(
                "w-9 h-9 md:w-11 md:h-11 rounded-full flex-shrink-0 flex items-center justify-center shadow-sm border",
                isUser
                    ? "bg-slate-100 border-slate-200"
                    : "bg-primary-light border-primary/20 text-primary"
            )}>
                {isUser ? <User className="w-5 h-5 text-slate-500" /> : <Activity className="w-5 h-5" />}
            </div>

            {/* Bubble */}
            <div className={clsx(
                "px-5 py-4 md:px-6 md:py-4 rounded-[20px] text-base leading-relaxed break-words relative shadow-sm",
                isUser
                    ? "bg-bubble-tint border border-primary/10 text-slate-800 rounded-tr-none"
                    : message.isExplanation
                        ? "bg-gradient-to-br from-white to-primary-light/20 border border-primary/20 rounded-tl-none font-medium text-slate-700 shadow-md"
                        : "healthcare-card rounded-tl-none bg-white font-normal"
            )}>
                {message.content}

                {/* Badge for AI Explanations */}
                {message.isExplanation && (
                    <div className="mt-4 inline-flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-white/70 border border-primary/15 text-xs font-bold uppercase tracking-widest text-primary">
                        <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                        AI Analysis
                    </div>
                )}
            </div>

        </div>
    );
}
