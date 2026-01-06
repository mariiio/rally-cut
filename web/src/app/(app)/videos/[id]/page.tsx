'use client';

import { use } from 'react';
import { EditorLayout } from '@/components/EditorLayout';

interface VideoEditorPageProps {
  params: Promise<{ id: string }>;
}

export default function VideoEditorPage({ params }: VideoEditorPageProps) {
  const { id } = use(params);
  return <EditorLayout videoId={id} />;
}
