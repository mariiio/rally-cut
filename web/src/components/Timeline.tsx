'use client';

import { useMemo, useCallback } from 'react';
import { Box, Typography } from '@mui/material';
import {
  Timeline as TimelineEditor,
  TimelineRow,
  TimelineAction,
  TimelineEffect,
} from '@xzdarcy/react-timeline-editor';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';

// Custom effect for rally segments
const effects: Record<string, TimelineEffect> = {
  rally: {
    id: 'rally',
    name: 'Rally',
  },
};

export function Timeline() {
  const { segments, updateSegment, selectSegment, selectedSegmentId } =
    useEditorStore();
  const { currentTime, duration, seek } = usePlayerStore();

  // Convert Rally[] to TimelineRow[] format
  const editorData: TimelineRow[] = useMemo(() => {
    return [
      {
        id: 'segments',
        actions: segments.map((seg) => ({
          id: seg.id,
          start: seg.start_time,
          end: seg.end_time,
          effectId: 'rally',
          selected: seg.id === selectedSegmentId,
        })),
      },
    ];
  }, [segments, selectedSegmentId]);

  // Handle segment changes (resize/move)
  const handleChange = useCallback(
    (data: TimelineRow[]) => {
      const actions = data[0]?.actions || [];
      actions.forEach((action: TimelineAction) => {
        const segment = segments.find((s) => s.id === action.id);
        if (segment) {
          // Only update if times changed
          if (
            segment.start_time !== action.start ||
            segment.end_time !== action.end
          ) {
            updateSegment(action.id, {
              start_time: action.start,
              end_time: action.end,
            });
          }
        }
      });
    },
    [segments, updateSegment]
  );

  // Handle clicking on timeline to seek
  const handleClickTimeArea = useCallback(
    (time: number) => {
      seek(time);
      return true;
    },
    [seek]
  );

  // Handle clicking on an action to select it
  const handleClickAction = useCallback(
    (_e: React.MouseEvent, action: { action: TimelineAction }) => {
      selectSegment(action.action.id);
      seek(action.action.start);
    },
    [selectSegment, seek]
  );

  if (segments.length === 0) {
    return (
      <Box
        sx={{
          height: 80,
          bgcolor: 'background.paper',
          borderRadius: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'text.secondary',
        }}
      >
        <Typography variant="body2">
          Load a JSON file to see the timeline
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        height: 100,
        bgcolor: 'background.paper',
        borderRadius: 1,
        overflow: 'hidden',
        '& .timeline-editor': {
          background: 'transparent !important',
        },
        '& .timeline-editor-action': {
          background: '#1976d2 !important',
          borderRadius: '4px !important',
        },
        '& .timeline-editor-action-selected': {
          background: '#2196f3 !important',
          border: '2px solid #fff !important',
        },
        '& .timeline-editor-cursor': {
          background: '#f44336 !important',
          width: '2px !important',
        },
      }}
    >
      <TimelineEditor
        editorData={editorData}
        effects={effects}
        onChange={handleChange}
        onClickTimeArea={handleClickTimeArea}
        onClickAction={handleClickAction}
        scale={Math.max(1, duration / 100)}
        scaleWidth={100}
        startLeft={20}
        autoScroll={true}
        getActionRender={(action) => {
          const segment = segments.find((s) => s.id === action.id);
          return (
            <Box
              sx={{
                width: '100%',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 10,
                color: 'white',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                px: 0.5,
              }}
            >
              {segment?.id}
            </Box>
          );
        }}
      />
    </Box>
  );
}
