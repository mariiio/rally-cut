import { SESv2Client, SendEmailCommand } from "@aws-sdk/client-sesv2";
import { env } from "../config/env.js";

// Lazy-init SES client (only when actually sending)
let sesClient: SESv2Client | null = null;

function getSesClient(): SESv2Client {
  if (!sesClient) {
    sesClient = new SESv2Client({ region: env.SES_REGION });
  }
  return sesClient;
}

// Cached disposable domain set for O(1) lookups
let disposableDomainSet: Set<string> | null = null;

async function getDisposableDomains(): Promise<Set<string>> {
  if (disposableDomainSet) return disposableDomainSet;
  try {
    const { default: domains } = await import("disposable-email-domains");
    disposableDomainSet = new Set(domains as string[]);
  } catch {
    disposableDomainSet = new Set();
  }
  return disposableDomainSet;
}

/**
 * Check if an email domain is disposable/temporary.
 */
export async function isDisposableEmail(email: string): Promise<boolean> {
  const domain = email.split("@")[1]?.toLowerCase();
  if (!domain) return false;

  const domains = await getDisposableDomains();
  return domains.has(domain);
}

/**
 * Send a verification email with a link containing the token.
 */
export async function sendVerificationEmail(
  email: string,
  token: string
): Promise<void> {
  const verifyUrl = `${env.FRONTEND_URL}/auth/verify?token=${encodeURIComponent(token)}`;

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d0e12; color: #e0e0e0; padding: 40px 20px; }
    .container { max-width: 480px; margin: 0 auto; }
    .logo { font-size: 24px; font-weight: 700; color: #ff6b4a; margin-bottom: 24px; }
    .button { display: inline-block; background: #ff6b4a; color: #fff !important; text-decoration: none; padding: 12px 32px; border-radius: 8px; font-weight: 600; margin: 24px 0; }
    .footer { margin-top: 32px; font-size: 13px; color: #888; }
  </style>
</head>
<body>
  <div class="container">
    <div class="logo">RallyCut</div>
    <h2 style="color: #fff; margin-bottom: 16px;">Verify your email</h2>
    <p>Click the button below to verify your email address and complete your registration.</p>
    <a href="${verifyUrl}" class="button">Verify Email</a>
    <p style="font-size: 14px; color: #aaa;">Or copy this link: ${verifyUrl}</p>
    <div class="footer">
      <p>This link expires in 24 hours.</p>
      <p>If you didn't create a RallyCut account, you can ignore this email.</p>
    </div>
  </div>
</body>
</html>`;

  // In development, log the token instead of sending email
  if (env.NODE_ENV === "development") {
    console.log(`\n=== VERIFICATION EMAIL ===`);
    console.log(`To: ${email}`);
    console.log(`Token: ${token}`);
    console.log(`URL: ${verifyUrl}`);
    console.log(`=========================\n`);
    return;
  }

  const client = getSesClient();
  await client.send(
    new SendEmailCommand({
      FromEmailAddress: env.SES_FROM_EMAIL,
      Destination: {
        ToAddresses: [email],
      },
      Content: {
        Simple: {
          Subject: {
            Data: "Verify your RallyCut email",
            Charset: "UTF-8",
          },
          Body: {
            Html: {
              Data: html,
              Charset: "UTF-8",
            },
          },
        },
      },
    })
  );
}
