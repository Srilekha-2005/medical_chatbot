const API_BASE = "http://localhost:8000";

export type ValueToRange = {
  metric?: string;
  value?: string;
  medical_range?: string;
};

export type ChatResponse = {
  response: string;
  query?: string;
  insight?: {
    value_to_range?: ValueToRange[];
    suggested_follow_ups?: string[];
    [key: string]: unknown;
  };
  retrieval_summary?: unknown;
};

export type UploadReportResponse = {
  response: string;
  report?: { filename?: string; source?: string; metrics?: unknown; raw_text_length?: number };
  insight?: unknown;
  retrieval_summary?: unknown;
};

export type ApiError = {
  detail: string | { message?: string; detail?: string }[];
};

async function handleResponse<T>(res: Response): Promise<T> {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      typeof (data as ApiError).detail === "string"
        ? (data as ApiError).detail
        : Array.isArray((data as ApiError).detail)
          ? (data as ApiError).detail.map((d: { message?: string }) => d.message || JSON.stringify(d)).join(", ")
          : (data as ApiError).detail?.message ?? res.statusText;
    throw new Error(detail || `Request failed: ${res.status}`);
  }
  return data as T;
}

export async function sendChatMessage(message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  return handleResponse<ChatResponse>(res);
}

export async function uploadReport(file: File): Promise<UploadReportResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload-report`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<UploadReportResponse>(res);
}
