import { env } from "../config/env.js";

export interface ModalJobParams {
  jobId: string;
  videoS3Key: string;
  callbackUrl: string;
}

export async function triggerModalDetection(
  params: ModalJobParams
): Promise<void> {
  if (!env.MODAL_FUNCTION_URL) {
    throw new Error("MODAL_FUNCTION_URL is not configured");
  }
  const response = await fetch(env.MODAL_FUNCTION_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      job_id: params.jobId,
      video_key: params.videoS3Key,
      callback_url: params.callbackUrl,
      webhook_secret: env.MODAL_WEBHOOK_SECRET,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Modal function call failed: ${response.status} - ${text}`);
  }
}
