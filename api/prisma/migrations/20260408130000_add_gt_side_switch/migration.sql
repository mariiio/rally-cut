-- Session 5 follow-up: manual side-switch override per rally.
-- null = use analysis flag, true = force switch at this rally, false = force no switch.
ALTER TABLE "rallies"
  ADD COLUMN "gt_side_switch" BOOLEAN;
