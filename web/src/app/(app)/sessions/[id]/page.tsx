'use client';

import { use } from 'react';
import { EditorLayout } from '@/components/EditorLayout';

interface SessionPageProps {
  params: Promise<{ id: string }>;
}

export default function SessionPage({ params }: SessionPageProps) {
  const { id } = use(params);
  return <EditorLayout sessionId={id} />;
}
