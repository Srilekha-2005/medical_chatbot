import Link from "next/link";
import { Activity, ShieldCheck, HeartPulse } from "lucide-react";

export default function LoginPage() {
  return (
    <div className="min-h-screen w-full flex items-center justify-center p-4 md:p-8 bg-background-gray">
      <div className="w-full max-w-lg bg-white rounded-[32px] shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-100 p-10 md:p-14 flex flex-col items-center">

        {/* Logo/Icon Container */}
        <div className="w-20 h-20 bg-primary-light rounded-3xl flex items-center justify-center mb-8 shadow-inner">
          <Activity className="w-10 h-10 text-primary" />
        </div>

        {/* Header content */}
        <div className="text-center mb-12">
          <h1 className="text-3xl md:text-5xl font-extrabold text-slate-900 tracking-tight mb-4 leading-tight">
            Health <br />
            <span className="text-primary">Assistant</span>
          </h1>
          <p className="text-slate-500 text-base md:text-lg leading-relaxed max-w-sm mx-auto">
            Your intelligent companion for understanding health metrics and educational insights.
          </p>
        </div>

        {/* Feature Highlights Minimal */}
        <div className="flex gap-8 mb-12 w-full justify-center text-slate-400">
          <div className="flex flex-col items-center gap-3">
            <div className="p-3 rounded-2xl bg-slate-50 border border-slate-100 drop-shadow-sm"><ShieldCheck className="w-6 h-6 text-slate-500" /></div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Secure</span>
          </div>
          <div className="flex flex-col items-center gap-3">
            <div className="p-3 rounded-2xl bg-slate-50 border border-slate-100 drop-shadow-sm"><HeartPulse className="w-6 h-6 text-secondary" /></div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Accurate</span>
          </div>
        </div>

        {/* Action bounds */}
        <div className="w-full max-w-sm mx-auto">
          <Link
            href="/chat"
            className="w-full flex items-center justify-center gap-4 bg-white border-2 border-slate-100 hover:border-primary hover:bg-primary-light/10 text-slate-800 text-lg font-bold py-4 px-6 rounded-2xl transition-all duration-200 shadow-sm hover:shadow-md active:scale-[0.98]"
          >
            <svg viewBox="0 0 24 24" className="w-6 h-6" aria-hidden="true">
              <path d="M12.0003 4.75C13.7703 4.75 15.3553 5.36002 16.6053 6.54998L20.0303 3.125C17.9502 1.19 15.2353 0 12.0003 0C7.31028 0 3.25527 2.69 1.25027 6.65L5.26528 9.765C6.22528 6.86 8.86528 4.75 12.0003 4.75Z" fill="#EA4335"></path>
              <path d="M23.49 12.275C23.49 11.49 23.415 10.73 23.3 10H12V14.51H18.47C18.18 15.99 17.34 17.25 16.08 18.1L19.945 21.1C22.2 19.01 23.49 15.92 23.49 12.275Z" fill="#4285F4"></path>
              <path d="M5.26498 14.235C5.02498 13.505 4.88498 12.73 4.88498 11.935C4.88498 11.14 5.01998 10.365 5.26498 9.63498L1.24998 6.51998C0.454982 8.13498 0 9.96 0 11.935C0 13.91 0.454982 15.735 1.24998 17.35L5.26498 14.235Z" fill="#FBBC05"></path>
              <path d="M12.0004 24C15.2404 24 17.9654 22.935 19.9454 21.095L16.0804 18.095C15.0054 18.82 13.6204 19.245 12.0004 19.245C8.86536 19.245 6.22536 17.135 5.26536 14.23L1.25036 17.345C3.25536 21.305 7.31036 24 12.0004 24Z" fill="#34A853"></path>
            </svg>
            Continue with Google
          </Link>

          <p className="mt-8 text-center text-sm font-medium text-slate-400">
            By continuing, you agree to our <a href="#" className="underline decoration-slate-300 hover:text-primary transition-colors">Terms of Service</a> & <a href="#" className="underline decoration-slate-300 hover:text-primary transition-colors">Privacy Policy</a>
          </p>
        </div>

      </div>
    </div>
  );
}
