"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Plus, FileText, ImageIcon, Loader2, User, MessageSquare, PlusCircle } from "lucide-react";
import ChatMessage from "@/components/chat/ChatMessage";
import MetricsDisplay from "@/components/chat/MetricsDisplay";

type Message = {
    id: string;
    role: "user" | "assistant";
    content: string | React.ReactNode;
    isExplanation?: boolean;
};

export default function ChatPage() {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: "welcome",
            role: "assistant",
            content: "Hello! I am your Health Assistant. How can I help you today? You can safely upload a PDF or image of your health records for me to analyze."
        }
    ]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => scrollToBottom(), [messages, isTyping]);

    const handleSend = () => {
        if (!input.trim()) return;

        const newUserMsg: Message = { id: Date.now().toString(), role: "user", content: input };
        setMessages(prev => [...prev, newUserMsg]);
        setInput("");
        setIsTyping(true);

        setTimeout(() => {
            setMessages(prev => [...prev, {
                id: (Date.now() + 1).toString(),
                role: "assistant",
                content: "I understand your concern. Monitoring your vital metrics consistently is the best way to handle your lifestyle changes effectively. Let me know if you want me to analyze recent clinical data for you."
            }]);
            setIsTyping(false);
        }, 1500);
    };

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const uploadId = Date.now().toString();
        const isImage = file.type.startsWith('image/');

        setMessages(prev => [...prev, {
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
        }]);

        setIsTyping(true);

        setTimeout(() => {
            setIsTyping(false);

            const responseBatchId = (Date.now() + 1).toString();

            setMessages(prev => [
                ...prev,
                {
                    id: responseBatchId + "_widget",
                    role: "assistant",
                    content: <MetricsDisplay />
                },
                {
                    id: responseBatchId + "_explanation",
                    role: "assistant",
                    isExplanation: true,
                    content: "I've carefully analyzed your document. Your blood pressure currently sits slightly elevated at 128/82 mmHg, but your heart rate and glucose levels are well within the normal range. Your sleep duration is quite optimal at 7.5 hours. Excellent job prioritizing your rest!"
                }
            ]);

        }, 2800);

        if (fileInputRef.current) fileInputRef.current.value = "";
    };


    return (
        <div className="flex h-screen w-full bg-white text-slate-900 font-sans overflow-hidden">

            {/* 2. Left Sidebar */}
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
                    {/* New Chat Button */}
                    <button className="w-full flex items-center gap-3 bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 font-bold py-3 px-4 rounded-xl transition-all shadow-[0_2px_10px_#00000008] group">
                        <PlusCircle className="w-5 h-5 text-primary group-hover:scale-110 transition-transform" />
                        New Chat
                    </button>
                </div>

                {/* Scrollable chat history */}
                <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
                    <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 px-2 mt-2">Previous Chats</div>

                    <button className="w-full flex items-center gap-3 bg-white/60 hover:bg-white text-slate-700 p-3 rounded-xl transition-colors border border-transparent hover:border-slate-200 shadow-sm">
                        <MessageSquare className="w-4 h-4 text-slate-400" />
                        <span className="text-sm font-medium truncate">Blood test analysis</span>
                    </button>

                    <button className="w-full flex items-center gap-3 hover:bg-white/50 text-slate-600 p-3 rounded-xl transition-colors">
                        <MessageSquare className="w-4 h-4 text-slate-400" />
                        <span className="text-sm font-medium truncate">Diet plan discussion</span>
                    </button>

                    <button className="w-full flex items-center gap-3 hover:bg-white/50 text-slate-600 p-3 rounded-xl transition-colors">
                        <MessageSquare className="w-4 h-4 text-slate-400" />
                        <span className="text-sm font-medium truncate">Sleep cycle review</span>
                    </button>
                </div>
            </aside>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col w-full h-full relative bg-chat-bg">

                {/* 4. Top Header */}
                <header className="h-16 flex items-center justify-between px-4 md:px-8 border-b border-slate-100 flex-shrink-0 bg-white/95 backdrop-blur-sm z-10 w-full">
                    <div className="font-bold text-lg text-slate-800">
                        <span className="md:hidden">Health Assistant</span> {/* Title fallback for mobile */}
                    </div>

                    <div className="flex items-center gap-3">
                        <button className="w-9 h-9 md:w-10 md:h-10 rounded-full bg-slate-100 border-2 border-slate-200 flex items-center justify-center overflow-hidden hover:border-primary transition-colors cursor-pointer shadow-sm">
                            <User className="w-4 h-4 md:w-5 md:h-5 text-slate-400" />
                        </button>
                    </div>
                </header>

                {/* 3. Main Chat Area */}
                <main className="flex-1 overflow-y-auto px-4 py-6 md:py-8 w-full scroll-smooth">
                    <div className="w-full max-w-4xl mx-auto flex flex-col gap-8 pb-4">

                        {messages.map((msg) => (
                            <ChatMessage key={msg.id} message={msg} />
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

                {/* 5. Input Area & 6. Footer */}
                <div className="flex-shrink-0 bg-white border-t border-slate-100 px-4 pt-4 pb-3 z-20 w-full relative">
                    <div className="max-w-4xl mx-auto flex flex-col gap-3 relative">

                        {/* Input Row */}
                        <div className="flex items-end gap-2 md:gap-3 w-full">
                            <input
                                type="file"
                                className="hidden"
                                ref={fileInputRef}
                                accept=".pdf,.png,.jpg,.jpeg"
                                onChange={handleFileUpload}
                            />

                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="h-12 w-12 md:h-[58px] md:w-[60px] flex-shrink-0 flex items-center justify-center rounded-2xl bg-slate-50 border border-slate-200 text-slate-500 hover:text-primary hover:bg-primary-light/50 hover:border-primary/50 transition-all duration-200 ease-out shadow-sm"
                                title="Upload Health Record"
                            >
                                <Plus className="w-6 h-6" />
                            </button>

                            <div className="flex-1 relative group bg-slate-50 rounded-3xl border border-slate-200 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary/20 shadow-inner flex items-center transition-all">
                                <textarea
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                            e.preventDefault();
                                            handleSend();
                                        }
                                    }}
                                    placeholder="Ask a health question or upload a document..."
                                    className="w-full resize-none min-h-[48px] md:min-h-[58px] py-3.5 md:py-[18px] px-4 md:px-5 bg-transparent border-none focus:outline-none focus:ring-0 text-base text-slate-800 placeholder-slate-400 overflow-y-hidden"
                                    rows={1}
                                />
                            </div>

                            <button
                                onClick={handleSend}
                                disabled={!input.trim()}
                                className="h-12 w-12 md:h-[58px] md:w-[60px] flex-shrink-0 flex items-center justify-center rounded-2xl bg-primary text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#2eaaaf] transition-colors shadow-md active:scale-95"
                                title="Send Message"
                            >
                                <Send className="w-5 h-5 md:w-6 md:h-6 ml-0.5" />
                            </button>
                        </div>

                        {/* Footer */}
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
