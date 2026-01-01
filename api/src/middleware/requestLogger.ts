import type { RequestHandler } from "express";

export const requestLogger: RequestHandler = (req, _res, next) => {
  const start = Date.now();

  _res.on("finish", () => {
    const duration = Date.now() - start;
    const log = {
      method: req.method,
      path: req.path,
      status: _res.statusCode,
      duration: `${duration}ms`,
    };
    console.log(JSON.stringify(log));
  });

  next();
};
