export interface TutorialStep {
  id: string;
  title: string;
  description: string;
  placement: 'top' | 'bottom' | 'left' | 'right';
  /** If provided, step is skipped when this returns false */
  shouldShow?: (context: TutorialContext) => boolean;
}

export interface TutorialContext {
  userRole: 'owner' | 'member' | null;
  hasRallies: boolean;
}

export const tutorialSteps: TutorialStep[] = [
  {
    id: 'detect-rallies',
    title: 'Add Rallies',
    description: 'Click here to auto-detect rallies with ML, or press M to manually mark rally start/end while playing',
    placement: 'bottom',
    // Only show for owners, or members on sessions without rallies
    shouldShow: (ctx) => ctx.userRole === 'owner' || !ctx.hasRallies,
  },
  {
    id: 'rally-list',
    title: 'Rally List',
    description: 'Your rallies appear here. Click to select and view on timeline',
    placement: 'right',
  },
  {
    id: 'timeline',
    title: 'Edit Rallies',
    description: 'Drag rally edges to adjust timing, or use arrow keys when selected',
    placement: 'top',
    // Only owners can edit rallies
    shouldShow: (ctx) => ctx.userRole === 'owner',
  },
  {
    id: 'camera-panel',
    title: 'Camera Effects',
    description: 'Add zoom and pan effects to make highlights dynamic',
    placement: 'left',
  },
  {
    id: 'keyboard-shortcuts',
    title: 'Keyboard Shortcuts',
    description: 'Click here for all shortcuts. Space to play, arrow keys to seek, M to mark rallies',
    placement: 'bottom',
  },
];
