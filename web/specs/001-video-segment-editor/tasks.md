# Tasks: Video Segment Editor

**Input**: Design documents from `/specs/001-video-segment-editor/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: No unit tests per constitution (MVP scope). Manual testing only.

**Organization**: Tasks grouped by user story for independent implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1-US7)
- Include exact file paths

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Initialize Next.js project with all dependencies

- [ ] T001 Initialize Next.js project with TypeScript, Tailwind, App Router in web/
- [ ] T002 Install dependencies: @mui/material @emotion/react @emotion/styled @mui/icons-material react-player @xzdarcy/react-timeline-editor zustand
- [ ] T003 [P] Configure MUI theme provider in src/app/layout.tsx
- [ ] T004 [P] Create TypeScript interfaces in src/types/segment.ts

---

## Phase 2: Foundational (Core Stores)

**Purpose**: State management infrastructure needed by all user stories

**CRITICAL**: All user stories depend on these stores

- [ ] T005 Create editorStore with segment CRUD operations in src/stores/editorStore.ts
- [ ] T006 [P] Create playerStore with playback state in src/stores/playerStore.ts
- [ ] T007 [P] Create file utility functions in src/utils/fileHandlers.ts
- [ ] T008 [P] Create time formatting utilities in src/utils/timeFormat.ts

**Checkpoint**: Stores ready - component implementation can begin

---

## Phase 3: User Story 1+2 - Load & Review (Priority: P1) MVP

**Goal**: Load video + JSON, review segments by clicking to seek

**Independent Test**: Upload video + JSON, click segment, verify video seeks correctly

### Implementation

- [ ] T009 [US1] Create VideoPlayer component with react-player in src/components/VideoPlayer.tsx
- [ ] T010 [P] [US1] Create PlayerControls component (play/pause, time display) in src/components/PlayerControls.tsx
- [ ] T011 [P] [US1] Create FileControls component (upload video, load JSON) in src/components/FileControls.tsx
- [ ] T012 [US2] Create SegmentList component (clickable segments) in src/components/SegmentList.tsx
- [ ] T013 [US2] Create Timeline component with react-timeline-editor in src/components/Timeline.tsx
- [ ] T014 [US1+2] Create EditorLayout component (grid layout) in src/components/EditorLayout.tsx
- [ ] T015 [US1+2] Wire up main page with all components in src/app/page.tsx
- [ ] T016 [US2] Implement segment highlighting based on currentTime in Timeline and SegmentList

**Checkpoint**: Can load video+JSON and navigate by clicking segments

---

## Phase 4: User Story 7 - Export JSON (Priority: P1)

**Goal**: Export edited segments as JSON file matching RallyCut format

**Independent Test**: Make any edit, export, verify JSON has correct format and recalculated stats

### Implementation

- [ ] T017 [US7] Add export button to FileControls in src/components/FileControls.tsx
- [ ] T018 [US7] Implement exportToJson() with stats recalculation in src/stores/editorStore.ts
- [ ] T019 [US7] Implement JSON download trigger in src/utils/fileHandlers.ts

**Checkpoint**: Full read-only workflow complete (load, review, export)

---

## Phase 5: User Story 3 - Trim Segments (Priority: P2)

**Goal**: Adjust segment start/end times via timeline drag or direct input

**Independent Test**: Drag segment edge, verify times update; edit in panel, verify timeline updates

### Implementation

- [ ] T020 [US3] Add segment resize handling to Timeline (onChangeEnd callback) in src/components/Timeline.tsx
- [ ] T021 [US3] Create SegmentEditPanel component with time inputs in src/components/SegmentEditPanel.tsx
- [ ] T022 [US3] Add "Set to current time" buttons for start/end in SegmentEditPanel
- [ ] T023 [US3] Implement segment validation (end > start) in src/stores/editorStore.ts
- [ ] T024 [US3] Add hasUnsavedChanges tracking to editorStore

**Checkpoint**: Can trim any segment via timeline or panel

---

## Phase 6: User Story 4+5 - Add/Remove Segments (Priority: P2)

**Goal**: Add new segments and delete existing ones

**Independent Test**: Add segment with times, verify appears; delete segment, verify removed

### Implementation

- [ ] T025 [US4] Create SegmentForm dialog for adding segments in src/components/SegmentForm.tsx
- [ ] T026 [US4] Implement addSegment with auto-generated ID in src/stores/editorStore.ts
- [ ] T027 [P] [US5] Add delete button with confirmation to SegmentList in src/components/SegmentList.tsx
- [ ] T028 [US5] Implement removeSegment in src/stores/editorStore.ts

**Checkpoint**: Can add and remove segments

---

## Phase 7: User Story 6 - Reorder Segments (Priority: P3)

**Goal**: Change segment order via drag-and-drop

**Independent Test**: Drag segment to new position, export, verify order changed

### Implementation

- [ ] T029 [US6] Add drag-and-drop reordering to SegmentList using MUI DnD
- [ ] T030 [US6] Implement reorderSegments in src/stores/editorStore.ts

**Checkpoint**: Can reorder segments

---

## Phase 8: Polish & Edge Cases

**Purpose**: Handle edge cases and improve UX

- [ ] T031 [P] Add unsaved changes warning on page unload in src/app/page.tsx
- [ ] T032 [P] Add error handling for invalid JSON with MUI Snackbar in src/components/FileControls.tsx
- [ ] T033 [P] Add loading states while video initializes
- [ ] T034 Style timeline segments with distinct colors
- [ ] T035 Add segment count and duration stats display in EditorLayout

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup (T001-T004)
- **Load & Review (Phase 3)**: Depends on Foundational (T005-T008)
- **Export (Phase 4)**: Depends on Load & Review (file controls exist)
- **Trim (Phase 5)**: Depends on Load & Review
- **Add/Remove (Phase 6)**: Depends on Load & Review
- **Reorder (Phase 7)**: Depends on Load & Review
- **Polish (Phase 8)**: Depends on all features

### Parallel Opportunities

Setup phase:
```
T003 (MUI theme) + T004 (types) can run in parallel
```

Foundational phase:
```
T006 (playerStore) + T007 (fileHandlers) + T008 (timeFormat) can run in parallel after T005
```

Phase 3:
```
T010 (PlayerControls) + T011 (FileControls) can run in parallel after T009
T012 + T013 can run in parallel
```

---

## Implementation Strategy

### MVP First (Phases 1-4)

1. Complete Setup + Foundational
2. Complete Load & Review (US1+2)
3. Complete Export (US7)
4. **STOP and VALIDATE**: Full read-only workflow works
5. Demo to user

### Incremental Delivery

1. MVP (Load/Review/Export) → Demo
2. Add Trim (US3) → Demo
3. Add Add/Remove (US4+5) → Demo
4. Add Reorder (US6) → Demo
5. Polish → Final release

---

## Summary

- **Total tasks**: 35
- **Phase 1 (Setup)**: 4 tasks
- **Phase 2 (Foundational)**: 4 tasks
- **Phase 3 (Load & Review)**: 8 tasks
- **Phase 4 (Export)**: 3 tasks
- **Phase 5 (Trim)**: 5 tasks
- **Phase 6 (Add/Remove)**: 4 tasks
- **Phase 7 (Reorder)**: 2 tasks
- **Phase 8 (Polish)**: 5 tasks

**MVP Scope**: Phases 1-4 (19 tasks) - Load video + JSON, review, export
