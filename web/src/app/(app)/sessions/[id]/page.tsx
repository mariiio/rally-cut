'use client';

import { use } from 'react';
import { useSearchParams } from 'next/navigation';
import { EditorLayout } from '@/components/EditorLayout';

interface SessionPageProps {
  params: Promise<{ id: string }>;
}

export default function SessionPage({ params }: SessionPageProps) {
  const { id } = use(params);
  const searchParams = useSearchParams();
  const initialVideoId = searchParams.get('video') || undefined;
  return <EditorLayout sessionId={id} initialVideoId={initialVideoId} />;
}
