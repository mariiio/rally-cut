import compression from "compression";
import cors from "cors";
import express from "express";
import helmet from "helmet";

import { env } from "./config/env.js";
import { errorHandler } from "./middleware/errorHandler.js";
import { requestLogger } from "./middleware/requestLogger.js";
import detectionRouter from "./routes/detection.js";
import healthRouter from "./routes/health.js";
import highlightsRouter from "./routes/highlights.js";
import ralliesRouter from "./routes/rallies.js";
import sessionsRouter from "./routes/sessions.js";
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

app.use(healthRouter);
app.use(sessionsRouter);
app.use(videosRouter);
app.use(detectionRouter);
app.use(ralliesRouter);
app.use(highlightsRouter);
app.use(webhooksRouter);

app.use(errorHandler);

if (process.env["NODE_ENV"] !== "test") {
  app.listen(env.PORT, () => {
    console.log(`Server running on port ${env.PORT}`);
  });
}

export default app;
