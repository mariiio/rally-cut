import type { RequestHandler } from "express";
import { z } from "zod";

interface ValidateOptions {
  body?: z.ZodType;
  params?: z.ZodType;
  query?: z.ZodType;
}

export function validateRequest(options: ValidateOptions): RequestHandler {
  return (req, _res, next) => {
    if (options.body !== undefined) {
      req.body = options.body.parse(req.body);
    }
    if (options.params !== undefined) {
      req.params = options.params.parse(req.params);
    }
    if (options.query !== undefined) {
      req.query = options.query.parse(req.query);
    }
    next();
  };
}
