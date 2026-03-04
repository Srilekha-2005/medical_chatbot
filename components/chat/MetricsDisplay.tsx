import { Activity } from "lucide-react";
import { HeartPulse, Droplet, Moon as MoonIcon, Thermometer } from "lucide-react";

export default function MetricsDisplay() {
    const metrics = [
        {
            label: "Blood Pressure",
            value: "128/82",
            unit: "mmHg",
            status: "normal",
            icon: <HeartPulse className="w-6 h-6 text-rose-500" />,
            bg: "bg-rose-50 border border-rose-100"
        },
        {
            label: "Heart Rate",
            value: "72",
            unit: "bpm",
            status: "good",
            icon: <Activity className="w-6 h-6 text-secondary" />,
            bg: "bg-cyan-50 border border-cyan-100"
        },
        {
            label: "Glucose",
            value: "95",
            unit: "mg/dL",
            status: "good",
            icon: <Droplet className="w-6 h-6 text-primary" />,
            bg: "bg-blue-50 border border-blue-100"
        },
        {
            label: "Sleep Duration",
            value: "7.5",
            unit: "hrs",
            status: "optimal",
            icon: <MoonIcon className="w-6 h-6 text-indigo-500" />,
            bg: "bg-indigo-50 border border-indigo-100"
        }
    ];

    return (
        <div className="healthcare-card p-6 md:p-8 animate-fade-in-up w-full mt-2 mb-2">
            <div className="flex items-center justify-between mb-8 border-b border-slate-100 pb-4">
                <h3 className="font-bold text-slate-900 text-lg flex items-center gap-3">
                    <span className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse shadow-[0_0_8px_#35c5cf99]"></span>
                    Extracted Health Metrics
                </h3>
                <span className="text-[11px] font-bold uppercase tracking-widest text-slate-500 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-full">Automated Extraction</span>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
                {metrics.map((metric, i) => (
                    <div key={i} className="bg-white border-2 border-slate-50/80 rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow hover:border-slate-100 flex flex-col justify-between">
                        <div className="flex items-center gap-3 mb-4">
                            <div className={`p-2.5 rounded-xl ${metric.bg} shadow-inner`}>
                                {metric.icon}
                            </div>
                        </div>

                        <div>
                            <div className="flex items-baseline gap-1.5 mt-2">
                                <span className="text-2xl md:text-3xl font-extrabold text-slate-800 tracking-tight">{metric.value}</span>
                                <span className="text-sm font-semibold text-slate-500">{metric.unit}</span>
                            </div>
                            <p className="text-sm text-slate-400 font-semibold mt-1 truncate">{metric.label}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
