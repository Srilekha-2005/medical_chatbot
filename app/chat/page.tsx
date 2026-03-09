"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Plus, FileText, ImageIcon, Loader2, User, MessageSquare, PlusCircle } from "lucide-react";
import clsx from "clsx";
import ChatMessage from "@/components/chat/ChatMessage";
import { sendChatMessage, uploadReport } from "@/services/api";

const STORAGE_KEY = "health-assistant-chats";

export type ValueToRange = { metric?: string; value?: string; medical_range?: string };

type Message = {
    id: string;
    role: "user" | "assistant";
    content: string | React.ReactNode;
    isExplanation?: boolean;
    value_to_range?: ValueToRange[];
    suggested_follow_ups?: string[];
};

type SerializedMessage = {
    id: string;
    role: "user" | "assistant";
    content: string;
    isExplanation?: boolean;
    value_to_range?: ValueToRange[];
    suggested_follow_ups?: string[];
};

type PreviousChat = { id: string; title: string; messages: SerializedMessage[] };

const WELCOME_MESSAGE: Message = {
    id: "welcome",
    role: "assistant",
    content: "Hello! I am your Health Assistant. How can I help you today? You can safely upload a PDF or image of your health records for me to analyze."
};

function serializeMessage(m: Message): SerializedMessage {
    const content = typeof m.content === "string" ? m.content : "[File upload]";
    return {
        id: m.id,
        role: m.role,
        content,
        isExplanation: m.isExplanation,
        value_to_range: m.value_to_range,
        suggested_follow_ups: m.suggested_follow_ups,
    };
}

function deserializeMessage(m: SerializedMessage): Message {
    return {
        id: m.id,
        role: m.role,
        content: m.content,
        isExplanation: m.isExplanation,
        value_to_range: m.value_to_range,
        suggested_follow_ups: m.suggested_follow_ups,
    };
}

function loadPreviousChats(): PreviousChat[] {
    if (typeof window === "undefined") return [];
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return [];
        return JSON.parse(raw);
    } catch {
        return [];
    }
}

function savePreviousChats(chats: PreviousChat[]) {
    if (typeof window === "undefined") return;
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    } catch {}
}

function generateId() {
    return `chat-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function chatTitle(messages: SerializedMessage[]): string {
    const firstUser = messages.find(m => m.role === "user");
    if (firstUser && firstUser.content) {
        const text = firstUser.content.replace(/^\[File upload\]$/i, "Report upload");
        return text.length > 30 ? text.slice(0, 30) + "…" : text;
    }
    return "New chat";
}

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const [currentChatId, setCurrentChatId] = useState<string | null>(null);
    const [previousChats, setPreviousChats] = useState<PreviousChat[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    useEffect(() => scrollToBottom(), [messages, isTyping]);

    const persistCurrentChat = useCallback((msgs: Message[]) => {
        if (msgs.length <= 1) return;
        const serialized = msgs.map(serializeMessage);
        const title = chatTitle(serialized);
        const chats = loadPreviousChats();
        const id = currentChatId || generateId();
        const existing = chats.findIndex(c => c.id === id);
        const entry: PreviousChat = { id, title, messages: serialized };
        const next = existing >= 0 ? chats.map((c, i) => (i === existing ? entry : c)) : [entry, ...chats].slice(0, 50);
        setPreviousChats(next);
        savePreviousChats(next);
        if (!currentChatId) setCurrentChatId(id);
    }, [currentChatId]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage = input.trim();
        setInput("");
        const newUserMsg: Message = { id: Date.now().toString(), role: "user", content: userMessage };
        setMessages(prev => [...prev, newUserMsg]);
        setIsTyping(true);

        try {
            const result = await sendChatMessage(userMessage);
            const responseText = result.response ?? "I couldn't generate a response. Please try again.";
            const insight = result.insight as { value_to_range?: ValueToRange[]; suggested_follow_ups?: string[] } | undefined;
            const nextMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: "assistant",
                content: responseText,
                value_to_range: insight?.value_to_range,
                suggested_follow_ups: insight?.suggested_follow_ups,
            };
            setMessages(prev => [...prev, nextMsg]);
            persistCurrentChat([...messages, newUserMsg, nextMsg]);
        } catch (err) {
            const errorText = err instanceof Error ? err.message : "Something went wrong. Please try again.";
            setMessages(prev => [
                ...prev,
                { id: `err-${Date.now()}`, role: "assistant", content: `Sorry, an error occurred: ${errorText}` }
            ]);
        } finally {
            setIsTyping(false);
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const uploadId = Date.now().toString();
        const isImage = file.type.startsWith("image/");

        setMessages(prev => [
            ...prev,
            {
                id: uploadId,
                role: "user",
                content: (
                    <div className="flex items-center gap-3 bg-white/40 p-3 rounded-xl border border-primary/20 shadow-sm transition">
                        <div className="bg-primary/10 p-2 rounded-lg">
                            {isImage ? <ImageIcon className="w-6 h-6 text-primary" /> : <FileText className="w-6 h-6 text-primary" />}
                        </div>
                        <span className="text-base font-medium truncate max-w-[200px] text-slate-800">{file.name}</span>
                    </div>
                )
            }
        ]);
        setIsTyping(true);
        if (fileInputRef.current) fileInputRef.current.value = "";

        try {
            const result = await uploadReport(file);
            const responseText = result.response ?? "I've received your report. Here’s what I found.";
            const insight = result.insight as { value_to_range?: ValueToRange[]; suggested_follow_ups?: string[] } | undefined;
            const nextMsg: Message = {
                id: `${Date.now() + 1}_explanation`,
                role: "assistant",
                isExplanation: true,
                content: responseText,
                value_to_range: insight?.value_to_range,
                suggested_follow_ups: insight?.suggested_follow_ups,
            };
            setMessages(prev => [...prev, nextMsg]);
            persistCurrentChat([
                ...messages,
                { id: uploadId, role: "user", content: `[File: ${file.name}]` },
                { ...nextMsg, id: "", content: responseText },
            ]);
        } catch (err) {
            const errorText = err instanceof Error ? err.message : "Upload failed. Please try again.";
            setMessages(prev => [
                ...prev,
                { id: `err-${Date.now()}`, role: "assistant", content: `Sorry, an error occurred: ${errorText}` }
            ]);
        } finally {
            setIsTyping(false);
        }
    };

    const handleNewChat = () => {
        persistCurrentChat(messages);
        setMessages([WELCOME_MESSAGE]);
        setCurrentChatId(null);
    };

    const handleLoadChat = (chat: PreviousChat) => {
        persistCurrentChat(messages);
        setMessages(chat.messages.map(deserializeMessage));
        setCurrentChatId(chat.id);
    };

    useEffect(() => {
        setPreviousChats(loadPreviousChats());
    }, []);

    return (
        <div className="flex h-screen w-full bg-white text-slate-900 font-sans overflow-hidden">

            {/* Left Sidebar */}
            <aside className="w-[260px] hidden md:flex flex-col flex-shrink-0 bg-primary-light border-r border-slate-200">
                <div className="p-4 flex flex-col gap-6">
                    <h1 className="text-xl font-black text-slate-900 tracking-tight flex items-center gap-2">
                        <span className="w-7 h-7 rounded-lg bg-white flex items-center justify-center shadow-sm">
                            <span className="w-2.5 h-2.5 rounded-full bg-primary relative">
                                <span className="absolute inline-flex h-full w-full rounded-full bg-primary opacity-75 animate-ping"></span>
                            </span>
                        </span>
                        Health Assistant
                    </h1>
                    <button
                        onClick={handleNewChat}
                        className="w-full flex items-center gap-3 bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 font-bold py-3 px-4 rounded-xl transition-all shadow-[0_2px_10px_#00000008] group"
                    >
                        <PlusCircle className="w-5 h-5 text-primary group-hover:scale-110 transition-transform" />
                        New Chat
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
                    <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 px-2 mt-2">Previous Chats</div>

                    {previousChats.length === 0 && (
                        <p className="text-sm text-slate-400 px-2">No previous chats yet.</p>
                    )}
                    {previousChats.map((chat) => (
                        <button
                            key={chat.id}
                            onClick={() => handleLoadChat(chat)}
                            className={clsx(
                                "w-full flex items-center gap-3 text-left p-3 rounded-xl transition-colors border shadow-sm",
                                currentChatId === chat.id
                                    ? "bg-white border-slate-200 text-slate-800"
                                    : "bg-white/60 hover:bg-white border-transparent hover:border-slate-200 text-slate-700"
                            )}
                        >
                            <MessageSquare className="w-4 h-4 text-slate-400 flex-shrink-0" />
                            <span className="text-sm font-medium truncate">{chat.title}</span>
                        </button>
                    ))}
                </div>
            </aside>

            {/* Main Content */}
            <div className="flex-1 flex flex-col w-full h-full relative bg-chat-bg">

                <header className="h-16 flex items-center justify-between px-4 md:px-8 border-b border-slate-100 flex-shrink-0 bg-white/95 backdrop-blur-sm z-10 w-full">
                    <div className="font-bold text-lg text-slate-800">
                        <span className="md:hidden">Health Assistant</span>
                    </div>
                    <div className="flex items-center gap-3">
                        <button className="w-9 h-9 md:w-10 md:h-10 rounded-full bg-slate-100 border-2 border-slate-200 flex items-center justify-center overflow-hidden hover:border-primary transition-colors cursor-pointer shadow-sm">
                            <User className="w-4 h-4 md:w-5 md:h-5 text-slate-400" />
                        </button>
                    </div>
                </header>

                <main className="flex-1 overflow-y-auto px-4 py-6 md:py-8 w-full scroll-smooth">
                    <div className="w-full max-w-4xl mx-auto flex flex-col gap-8 pb-4">
                        {messages.map((msg) => (
                            <ChatMessage
                                key={msg.id}
                                message={msg}
                                onSuggestionClick={(text) => setInput(text)}
                            />
                        ))}

                        {isTyping && (
                            <div className="flex gap-4 max-w-[85%] self-start animate-fade-in pl-2">
                                <div className="w-9 h-9 rounded-full bg-primary-light flex-shrink-0 flex items-center justify-center border border-primary/20 shadow-sm">
                                    <Loader2 className="w-5 h-5 text-primary animate-spin" />
                                </div>
                                <div className="bg-white px-5 py-3.5 rounded-[20px] rounded-tl-none border border-slate-100 text-slate-500 font-medium text-base flex items-center gap-1 shadow-sm">
                                    Analyzing clinical data<span className="animate-pulse tracking-widest text-primary ml-1">...</span>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} className="h-2" />
                    </div>
                </main>

                <div className="flex-shrink-0 bg-white border-t border-slate-100 px-4 pt-4 pb-3 z-20 w-full relative">
                    <div className="max-w-4xl mx-auto flex flex-col gap-3 relative">
                        <div className="flex items-end gap-2 md:gap-3 w-full">
                            <input
                                type="file"
                                className="hidden"
                                ref={fileInputRef}
                                accept=".pdf,.png,.jpg,.jpeg,.gif,.bmp,.tiff,.tif"
                                onChange={handleFileUpload}
                            />
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                disabled={isTyping}
                                className="h-12 w-12 md:h-[58px] md:w-[60px] flex-shrink-0 flex items-center justify-center rounded-2xl bg-slate-50 border border-slate-200 text-slate-500 hover:text-primary hover:bg-primary-light/50 hover:border-primary/50 transition-all duration-200 ease-out shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                                title="Upload Health Record"
                            >
                                <Plus className="w-6 h-6" />
                            </button>
                            <div className="flex-1 relative group bg-slate-50 rounded-3xl border border-slate-200 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary/20 shadow-inner flex items-center transition-all">
                                <textarea
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === "Enter" && !e.shiftKey) {
                                            e.preventDefault();
                                            handleSend();
                                        }
                                    }}
                                    placeholder="Ask a health question or upload a document..."
                                    className="w-full resize-none min-h-[48px] md:min-h-[58px] py-3.5 md:py-[18px] px-4 md:px-5 bg-transparent border-none focus:outline-none focus:ring-0 text-base text-slate-800 placeholder-slate-400 overflow-y-hidden"
                                    rows={1}
                                    disabled={isTyping}
                                />
                            </div>
                            <button
                                onClick={handleSend}
                                disabled={!input.trim() || isTyping}
                                className="h-12 w-12 md:h-[58px] md:w-[60px] flex-shrink-0 flex items-center justify-center rounded-2xl bg-primary text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#2eaaaf] transition-colors shadow-md active:scale-95"
                                title="Send Message"
                            >
                                <Send className="w-5 h-5 md:w-6 md:h-6 ml-0.5" />
                            </button>
                        </div>
                        <div className="text-center mt-0.5">
                            <p className="text-[11px] md:text-xs text-slate-400 font-medium tracking-wide">
                                AI Health Assistant provides educational insights only and does not provide medical diagnosis.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
