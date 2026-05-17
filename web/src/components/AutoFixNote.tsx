'use client';

import { useEffect, useState } from 'react';
import { IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import type { AutoFix } from '@/types/rally';

function isDismissed(videoId: string): boolean {
  try {
    return localStorage.getItem(`autofix-note-dismissed-${videoId}`) === '1';
  } catch {
    return false;
  }
}

export function AutoFixNote({
  fixes,
  videoId,
}: {
  fixes: AutoFix[] | undefined;
  videoId: string | undefined;
}) {
  const [dismissed, setDismissed] = useState(() => (videoId ? isDismissed(videoId) : false));

  useEffect(() => {
    setDismissed(videoId ? isDismissed(videoId) : false);
  }, [videoId]);

  if (!fixes || fixes.length === 0 || dismissed) return null;

  const handleDismiss = () => {
    setDismissed(true);
    if (!videoId) return;
    try {
      localStorage.setItem(`autofix-note-dismissed-${videoId}`, '1');
    } catch {
      // Ignore storage errors.
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        margin: '4px 0 0 0',
        fontSize: 12,
        color: '#4caf50',
      }}
    >
      <ul style={{ margin: 0, padding: 0, listStyle: 'none', flex: 1 }}>
        {fixes.map((fx) => (
          <li key={fx.id}>✓ {fx.message}</li>
        ))}
      </ul>
      <IconButton
        size="small"
        onClick={handleDismiss}
        aria-label="dismiss"
        sx={{ color: '#4caf50', p: 0.25 }}
      >
        <CloseIcon sx={{ fontSize: 14 }} />
      </IconButton>
    </div>
  );
}
