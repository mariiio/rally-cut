'use client';

import { createContext, useContext, useState, useEffect, useCallback, ReactNode, useMemo } from 'react';
import { tutorialSteps, TutorialStep, TutorialContext as StepContext } from './tutorialSteps';
import { TutorialOverlay } from './TutorialOverlay';

const STORAGE_KEY_COMPLETED = 'rallycut_tutorial_completed';
const STORAGE_KEY_DISMISSED = 'rallycut_tutorial_dismissed';
const STORAGE_KEY_VERSION = 'rallycut_tutorial_version';
const CURRENT_VERSION = '1';
const AUTO_START_DELAY = 1500;

interface TutorialContextValue {
  isActive: boolean;
  currentStep: number;
  currentStepData: TutorialStep | null;
  visibleSteps: TutorialStep[];
  startTutorial: () => void;
  nextStep: () => void;
  prevStep: () => void;
  skipTutorial: () => void;
  markStepUnavailable: (stepId: string) => void;
}

const TutorialContext = createContext<TutorialContextValue | null>(null);

export function useTutorial() {
  const context = useContext(TutorialContext);
  if (!context) {
    throw new Error('useTutorial must be used within a TutorialProvider');
  }
  return context;
}

interface TutorialProviderProps {
  children: ReactNode;
  autoStart?: boolean;
  context?: StepContext;
}

export function TutorialProvider({
  children,
  autoStart = true,
  context = { userRole: 'owner', hasRallies: false },
}: TutorialProviderProps) {
  const [isActive, setIsActive] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [hasCheckedStorage, setHasCheckedStorage] = useState(false);
  const [unavailableStepIds, setUnavailableStepIds] = useState<Set<string>>(new Set());

  // Filter steps based on context conditions and DOM availability
  const visibleSteps = useMemo(() => {
    return tutorialSteps.filter((step) => {
      // Check shouldShow condition
      if (step.shouldShow && !step.shouldShow(context)) {
        return false;
      }
      // Check if marked unavailable (DOM element not found)
      if (unavailableStepIds.has(step.id)) {
        return false;
      }
      return true;
    });
  }, [context, unavailableStepIds]);

  const currentStepData = visibleSteps[currentStepIndex] ?? null;

  // Check localStorage on mount
  useEffect(() => {
    const completed = localStorage.getItem(STORAGE_KEY_COMPLETED);
    const dismissed = localStorage.getItem(STORAGE_KEY_DISMISSED);
    const version = localStorage.getItem(STORAGE_KEY_VERSION);
    const shouldShow = !completed && !dismissed && (!version || version === CURRENT_VERSION);

    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: one-time initialization
    setHasCheckedStorage(true);

    if (shouldShow && autoStart) {
      const timer = setTimeout(() => setIsActive(true), AUTO_START_DELAY);
      return () => clearTimeout(timer);
    }
  }, [autoStart]);

  const startTutorial = useCallback(() => {
    setCurrentStepIndex(0);
    setUnavailableStepIds(new Set());
    setIsActive(true);
  }, []);

  const nextStep = useCallback(() => {
    if (currentStepIndex < visibleSteps.length - 1) {
      setCurrentStepIndex((i) => i + 1);
    } else {
      localStorage.setItem(STORAGE_KEY_COMPLETED, 'true');
      localStorage.setItem(STORAGE_KEY_VERSION, CURRENT_VERSION);
      setIsActive(false);
    }
  }, [currentStepIndex, visibleSteps.length]);

  const prevStep = useCallback(() => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex((i) => i - 1);
    }
  }, [currentStepIndex]);

  const skipTutorial = useCallback(() => {
    localStorage.setItem(STORAGE_KEY_DISMISSED, 'true');
    setIsActive(false);
  }, []);

  const markStepUnavailable = useCallback((stepId: string) => {
    setUnavailableStepIds((prev) => {
      if (prev.has(stepId)) return prev;
      const next = new Set(prev);
      next.add(stepId);
      return next;
    });
  }, []);

  // Auto-advance if current step becomes unavailable
  useEffect(() => {
    if (isActive && currentStepData && unavailableStepIds.has(currentStepData.id)) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: auto-advance when step becomes unavailable
      nextStep();
    }
  }, [isActive, currentStepData, unavailableStepIds, nextStep]);

  const value = useMemo<TutorialContextValue>(() => ({
    isActive,
    currentStep: currentStepIndex,
    currentStepData,
    visibleSteps,
    startTutorial,
    nextStep,
    prevStep,
    skipTutorial,
    markStepUnavailable,
  }), [
    isActive,
    currentStepIndex,
    currentStepData,
    visibleSteps,
    startTutorial,
    nextStep,
    prevStep,
    skipTutorial,
    markStepUnavailable,
  ]);

  return (
    <TutorialContext.Provider value={value}>
      {children}
      {hasCheckedStorage && isActive && currentStepData && <TutorialOverlay />}
    </TutorialContext.Provider>
  );
}
