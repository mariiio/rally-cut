import compression from "compression";
import cors from "cors";
import express from "express";
import helmet from "helmet";

import { env } from "./config/env.js";
import { errorHandler } from "./middleware/errorHandler.js";
import { requestLogger } from "./middleware/requestLogger.js";
import { resolveUser } from "./middleware/resolveUser.js";
import detectionRouter from "./routes/detection.js";
import exportsRouter from "./routes/exports.js";
import healthRouter from "./routes/health.js";
import highlightsRouter from "./routes/highlights.js";
import identityRouter from "./routes/identity.js";
import ralliesRouter from "./routes/rallies.js";
import sessionsRouter from "./routes/sessions.js";
import sharesRouter from "./routes/shares.js";
import videosRouter from "./routes/videos.js";
import webhooksRouter from "./routes/webhooks.js";

const app = express();

app.use(helmet());
app.use(
  cors({
    origin: env.CORS_ORIGIN,
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

// Identity routes
app.use(identityRouter);

// Main API routes
app.use(sessionsRouter);
app.use(sharesRouter);
app.use(videosRouter);
app.use(detectionRouter);
app.use(exportsRouter);
app.use(ralliesRouter);
app.use(highlightsRouter);

app.use(errorHandler);

if (process.env["NODE_ENV"] !== "test") {
  app.listen(env.PORT, () => {
    console.log(`Server running on port ${env.PORT}`);
  });
}

export default app;
