import { getSignedCookies } from "@aws-sdk/cloudfront-signer";
import { env } from "../config/env.js";

export interface SignedCookies {
  "CloudFront-Policy": string;
  "CloudFront-Signature": string;
  "CloudFront-Key-Pair-Id": string;
}

export function generateSignedCookies(sessionId: string): SignedCookies | null {
  if (env.NODE_ENV === "development" && !env.CLOUDFRONT_PRIVATE_KEY.startsWith("-----BEGIN")) {
    return null;
  }

  const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000);

  const policy = {
    Statement: [
      {
        Resource: `https://${env.CLOUDFRONT_DOMAIN}/videos/${sessionId}/*`,
        Condition: {
          DateLessThan: {
            "AWS:EpochTime": Math.floor(expiresAt.getTime() / 1000),
          },
        },
      },
    ],
  };

  const cookies = getSignedCookies({
    keyPairId: env.CLOUDFRONT_KEY_PAIR_ID,
    privateKey: env.CLOUDFRONT_PRIVATE_KEY,
    policy: JSON.stringify(policy),
  });

  return cookies as SignedCookies;
}

export function getCloudFrontUrl(s3Key: string): string {
  return `https://${env.CLOUDFRONT_DOMAIN}/${s3Key}`;
}
