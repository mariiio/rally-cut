import compression from "compression";
import cors from "cors";
import express from "express";
import helmet from "helmet";

import { env } from "./config/env.js";
import { errorHandler } from "./middleware/errorHandler.js";
import { requestLogger } from "./middleware/requestLogger.js";
import { resolveUser } from "./middleware/resolveUser.js";
import confirmationRouter from "./routes/confirmation.js";
import detectionRouter from "./routes/detection.js";
import exportsRouter from "./routes/exports.js";
import feedbackRouter from "./routes/feedback.js";
import healthRouter from "./routes/health.js";
import highlightsRouter from "./routes/highlights.js";
import authRouter from "./routes/auth.js";
import identityRouter from "./routes/identity.js";
import ralliesRouter from "./routes/rallies.js";
import sessionsRouter from "./routes/sessions.js";
import sharesRouter from "./routes/shares.js";
import videoSharesRouter from "./routes/videoShares.js";
import accessRequestsRouter from "./routes/accessRequests.js";
import videosRouter from "./routes/videos.js";
import webhooksRouter from "./routes/webhooks.js";

const app = express();

app.use(helmet());

// CORS: Allow configured origin + Label Studio for development
const allowedOrigins = [env.CORS_ORIGIN];
if (env.LABEL_STUDIO_URL) {
  allowedOrigins.push(env.LABEL_STUDIO_URL);
}
app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (like mobile apps or curl)
      if (!origin) return callback(null, true);
      if (allowedOrigins.includes(origin)) {
        return callback(null, true);
      }
      return callback(new Error("Not allowed by CORS"));
    },
    credentials: true,
  })
);
app.use(compression());
app.use(express.json({ limit: "1mb" }));

app.use(requestLogger);

// Health check before user resolution
app.use(healthRouter);

// Webhooks don't need user context
app.use(webhooksRouter);

// Resolve user from X-Visitor-Id header for all other routes
app.use(resolveUser);

// Auth routes (register, verify, link)
app.use(authRouter);

// Identity routes
app.use(identityRouter);

// Main API routes
// videosRouter must come before sessionsRouter to ensure /videos/upload-url
// matches before /videos/:videoId (which would interpret "upload-url" as a videoId)
app.use(videosRouter);
app.use(sessionsRouter);
app.use(sharesRouter);
app.use(videoSharesRouter);
app.use(accessRequestsRouter);
app.use(detectionRouter);
app.use(confirmationRouter);
app.use(exportsRouter);
app.use(ralliesRouter);
app.use(highlightsRouter);
app.use(feedbackRouter);

app.use(errorHandler);

if (process.env["NODE_ENV"] !== "test") {
  app.listen(env.PORT, () => {
    console.log(`Server running on port ${env.PORT}`);
  });
}

export default app;
