// Google Analytics 4 event tracking utilities

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void;
  }
}

type GTagEvent = {
  action: string;
  category: string;
  label?: string;
  value?: number;
};

/**
 * Track a custom event in Google Analytics
 */
export function trackEvent({ action, category, label, value }: GTagEvent) {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', action, {
      event_category: category,
      event_label: label,
      value: value,
    });
  }
}

/**
 * Track CTA button clicks
 */
export function trackCTAClick(location: string, destination: string) {
  trackEvent({
    action: 'cta_click',
    category: 'engagement',
    label: `${location}_to_${destination}`,
  });
}

/**
 * Track waitlist signups
 */
export function trackWaitlistJoin(tier: string) {
  trackEvent({
    action: 'waitlist_join',
    category: 'conversion',
    label: tier,
  });
}

/**
 * Track video upload started
 */
export function trackVideoUploadStarted() {
  trackEvent({
    action: 'video_upload_started',
    category: 'engagement',
  });
}

/**
 * Track video upload completed
 */
export function trackVideoUploadCompleted() {
  trackEvent({
    action: 'video_upload_completed',
    category: 'conversion',
  });
}

/**
 * Track pricing section view
 */
export function trackPricingView() {
  trackEvent({
    action: 'pricing_view',
    category: 'engagement',
  });
}

/**
 * Track FAQ interaction
 */
export function trackFAQExpand(question: string) {
  trackEvent({
    action: 'faq_expand',
    category: 'engagement',
    label: question,
  });
}
