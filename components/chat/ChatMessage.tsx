import { Activity, User } from "lucide-react";
import clsx from "clsx";

export type ValueToRange = { metric?: string; value?: string; medical_range?: string };

type Message = {
    id: string;
    role: "user" | "assistant";
    content: string | React.ReactNode;
    isExplanation?: boolean;
    value_to_range?: ValueToRange[];
    suggested_follow_ups?: string[];
};

function metricStatusClass(medicalRange: string): string {
    const r = (medicalRange || "").toLowerCase();
    if (r.includes("normal") && !r.includes("elevated") && !r.includes("high") && !r.includes("low")) return "bg-emerald-100 border-emerald-300 text-emerald-800";
    if (r.includes("elevated") || r.includes("borderline") || r.includes("prediabetes") || r.includes("mild")) return "bg-amber-100 border-amber-400 text-amber-900";
    return "bg-red-100 border-red-300 text-red-800";
}

export default function ChatMessage({ message, onSuggestionClick }: { message: Message; onSuggestionClick?: (text: string) => void }) {
    const isUser = message.role === "user";
    const valueToRange = message.value_to_range ?? [];
    const suggestions = message.suggested_follow_ups ?? [];

    return (
        <div className={clsx(
            "flex gap-4 md:gap-5 w-full max-w-[90%] md:max-w-[85%]",
            isUser ? "self-end flex-row-reverse" : "self-start"
        )}>

            <div className={clsx(
                "w-9 h-9 md:w-11 md:h-11 rounded-full flex-shrink-0 flex items-center justify-center shadow-sm border",
                isUser ? "bg-slate-100 border-slate-200" : "bg-primary-light border-primary/20 text-primary"
            )}>
                {isUser ? <User className="w-5 h-5 text-slate-500" /> : <Activity className="w-5 h-5" />}
            </div>

            <div className="flex flex-col gap-2 flex-1 min-w-0">
                {/* Metric highlights */}
                {!isUser && valueToRange.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-1">
                        {valueToRange.map((v, i) => (
                            <span
                                key={i}
                                className={clsx(
                                    "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm font-medium",
                                    metricStatusClass(v.medical_range ?? "")
                                )}
                            >
                                <span className="font-semibold">{v.metric?.replace(/_/g, " ")}:</span>
                                <span>{v.value}</span>
                                <span className="opacity-90 text-xs">({v.medical_range})</span>
                            </span>
                        ))}
                    </div>
                )}

                <div className={clsx(
                    "px-5 py-4 md:px-6 md:py-4 rounded-[20px] text-base leading-relaxed break-words relative shadow-sm",
                    isUser
                        ? "bg-bubble-tint border border-primary/10 text-slate-800 rounded-tr-none"
                        : message.isExplanation
                            ? "bg-gradient-to-br from-white to-primary-light/20 border border-primary/20 rounded-tl-none font-medium text-slate-700 shadow-md"
                            : "healthcare-card rounded-tl-none bg-white font-normal"
                )}>
                    {message.content}

                    {message.isExplanation && (
                        <div className="mt-4 inline-flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-white/70 border border-primary/15 text-xs font-bold uppercase tracking-widest text-primary">
                            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                            AI Analysis
                        </div>
                    )}
                </div>

                {/* Follow-up suggestions */}
                {!isUser && suggestions.length > 0 && (
                    <div className="mt-2">
                        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">You may also want to ask:</p>
                        <ul className="flex flex-wrap gap-2">
                            {suggestions.map((s, i) => (
                                <li key={i}>
                                    <button
                                        type="button"
                                        onClick={() => onSuggestionClick?.(s)}
                                        className="text-left px-3 py-2 rounded-xl bg-slate-50 hover:bg-primary-light/50 border border-slate-200 hover:border-primary/30 text-sm text-slate-700 hover:text-slate-900 transition-colors"
                                    >
                                        • {s}
                                    </button>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
}
