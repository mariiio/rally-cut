/**
 * Label Studio integration service for ground truth labeling.
 *
 * Provides seamless integration between RallyCut and Label Studio:
 * - Export tracking predictions as pre-annotations (player_1-4 + ball)
 * - Import corrected annotations as ground truth
 * - Task reuse to avoid duplicates (stored in playerTrack.groundTruthTaskId)
 * - Rally-bounded labels (hidden after rally ends via enabled: false)
 *
 * Requires Label Studio running locally with LABEL_STUDIO_API_KEY configured.
 * Project "RallyCut Ground Truth" is auto-created on first export.
 *
 * Frame timing: All frame numbers are calculated at 30fps (LABEL_STUDIO_FPS)
 * regardless of actual video fps. This ensures correct sync since Label Studio
 * interprets frame numbers at its default 30fps rate.
 */

import { prisma } from "../lib/prisma.js";
import { env } from "../config/env.js";

// Label Studio API configuration
const LABEL_STUDIO_URL = env.LABEL_STUDIO_URL || "http://localhost:8082";
const LABEL_STUDIO_API_KEY = env.LABEL_STUDIO_API_KEY;

// Frame rate used for Label Studio frame number calculations.
// Label Studio interprets frame numbers at ~30fps regardless of actual video fps.
// Using a fixed rate ensures correct timing across all video framerates.
const LABEL_STUDIO_FPS = 30;

// Label config for video object tracking with individual player labels
const LABEL_CONFIG = `
<View>
  <Video name="video" value="$video"/>
  <VideoRectangle name="box" toName="video"/>
  <Labels name="label" toName="video">
    <Label value="player_1" background="#4CAF50"/>
    <Label value="player_2" background="#2196F3"/>
    <Label value="player_3" background="#FF9800"/>
    <Label value="player_4" background="#9C27B0"/>
    <Label value="ball" background="#f44336"/>
  </Labels>
</View>
`;

interface LabelStudioConfig {
  url: string;
  apiKey: string;
  projectId?: number;
}

interface ExportResult {
  success: boolean;
  taskId?: number;
  projectId?: number;
  taskUrl?: string;
  error?: string;
}

interface ImportResult {
  success: boolean;
  playerCount?: number;
  ballCount?: number;
  frameCount?: number;
  error?: string;
}

interface GroundTruthData {
  positions: Array<{
    frameNumber: number;
    trackId: number;
    label: string;
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
  }>;
  frameCount: number;
  videoWidth: number;
  videoHeight: number;
}

/**
 * Build Label Studio predictions from tracking data.
 *
 * Converts RallyCut tracking positions to Label Studio videorectangle format.
 * Only includes primary tracks (4 players) and ball, bounded to rally duration.
 *
 * @param playerTrack - The player track data from database
 * @param rallyStartTime - Time in seconds where the rally starts in the full video
 * @param rallyEndTime - Time in seconds where the rally ends in the full video
 * @param fps - Video frames per second (used for frame calculation)
 * @param videoDuration - Total video duration in seconds (for Label Studio metadata)
 * @returns Array of Label Studio prediction results
 */
function buildPredictions(
  playerTrack: any,
  rallyStartTime: number = 0,
  rallyEndTime: number = 0,
  fps: number = 30,
  videoDuration: number = 0
): any[] {
  const results: any[] = [];

  // Video metadata for Label Studio
  const framesCount = Math.round(videoDuration * fps);
  const duration = videoDuration;

  // Calculate max frame number for rally (relative to rally start)
  const rallyDuration = rallyEndTime - rallyStartTime;
  const maxRallyFrame = Math.ceil(rallyDuration * fps);

  // Parse positions and primary track IDs
  const positions = playerTrack.positionsJson || [];
  const primaryTrackIds: number[] = playerTrack.primaryTrackIds || [];

  // Create a map from trackId to player number (1-4)
  const trackIdToPlayerNum = new Map<number, number>();
  primaryTrackIds.forEach((trackId: number, index: number) => {
    trackIdToPlayerNum.set(trackId, index + 1);
  });

  // Group by track ID, only including primary tracks
  const trackGroups: Record<number, any[]> = {};
  for (const pos of positions) {
    const trackId = pos.trackId;
    if (!trackIdToPlayerNum.has(trackId)) continue; // Only primary tracks
    if (!trackGroups[trackId]) trackGroups[trackId] = [];
    trackGroups[trackId].push(pos);
  }

  // Build video rectangle annotations for each primary player
  for (const trackId of Object.keys(trackGroups).map(Number).sort()) {
    const trackPositions = trackGroups[trackId]
      .filter((pos: any) => pos.frameNumber <= maxRallyFrame)
      .sort((a: any, b: any) => a.frameNumber - b.frameNumber);

    const playerNum = trackIdToPlayerNum.get(trackId) || 1;

    const sequence = trackPositions.map((pos: any) => {
      // Calculate time from rally start + position frame offset
      const time = rallyStartTime + pos.frameNumber / fps;
      // Frame calculated at LABEL_STUDIO_FPS (not video fps) for correct LS interpretation
      const frame = Math.round(time * LABEL_STUDIO_FPS) + 1;

      // Convert normalized center to percentage top-left
      const xPct = (pos.x - pos.width / 2) * 100;
      const yPct = (pos.y - pos.height / 2) * 100;
      const wPct = pos.width * 100;
      const hPct = pos.height * 100;

      return {
        frame,
        time,
        enabled: true,
        x: Math.max(0, Math.min(100, xPct)),
        y: Math.max(0, Math.min(100, yPct)),
        width: wPct,
        height: hPct,
        rotation: 0,
      };
    });

    if (sequence.length > 0) {
      // Add a final keyframe with enabled: false to hide after rally ends
      const lastKeyframe = sequence[sequence.length - 1];
      const endTime = rallyEndTime + 0.001; // Just after rally end
      const endFrame = Math.round(endTime * LABEL_STUDIO_FPS) + 1;
      sequence.push({
        frame: endFrame,
        time: endTime,
        enabled: false,
        x: lastKeyframe.x,
        y: lastKeyframe.y,
        width: lastKeyframe.width,
        height: lastKeyframe.height,
        rotation: 0,
      });

      results.push({
        id: `player_${playerNum}`,
        type: "videorectangle",
        value: {
          sequence,
          labels: [`player_${playerNum}`],
          framesCount,
          duration,
        },
        origin: "prediction",
        to_name: "video",
        from_name: "box",
      });
    }
  }

  // Add ball track if available
  const ballPositions = playerTrack.ballPositionsJson || [];
  if (ballPositions.length > 0) {
    const ballSequence = ballPositions
      .filter((pos: any) => pos.frameNumber <= maxRallyFrame)
      .sort((a: any, b: any) => a.frameNumber - b.frameNumber)
      .map((pos: any) => {
        // Calculate time from rally start + position frame offset
        const time = rallyStartTime + pos.frameNumber / fps;
        // Frame calculated at LABEL_STUDIO_FPS (not video fps) for correct LS interpretation
        const frame = Math.round(time * LABEL_STUDIO_FPS) + 1;
        const ballSizePct = 2.0;

        return {
          frame,
          time,
          enabled: true,
          x: Math.max(0, Math.min(100, pos.x * 100 - ballSizePct / 2)),
          y: Math.max(0, Math.min(100, pos.y * 100 - ballSizePct / 2)),
          width: ballSizePct,
          height: ballSizePct,
          rotation: 0,
        };
      });

    if (ballSequence.length > 0) {
      // Add a final keyframe with enabled: false to hide after rally ends
      const lastKeyframe = ballSequence[ballSequence.length - 1];
      const endTime = rallyEndTime + 0.001;
      const endFrame = Math.round(endTime * LABEL_STUDIO_FPS) + 1;
      ballSequence.push({
        frame: endFrame,
        time: endTime,
        enabled: false,
        x: lastKeyframe.x,
        y: lastKeyframe.y,
        width: lastKeyframe.width,
        height: lastKeyframe.height,
        rotation: 0,
      });

      results.push({
        id: "ball_0",
        type: "videorectangle",
        value: {
          sequence: ballSequence,
          labels: ["ball"],
          framesCount,
          duration,
        },
        origin: "prediction",
        to_name: "video",
        from_name: "box",
      });
    }
  }

  return results;
}

/**
 * Get or create a Label Studio project for RallyCut.
 */
async function getOrCreateProject(config: LabelStudioConfig): Promise<number> {
  const headers = {
    Authorization: `Token ${config.apiKey}`,
    "Content-Type": "application/json",
  };

  // Check for existing project
  const listResp = await fetch(`${config.url}/api/projects`, { headers });
  if (!listResp.ok) {
    throw new Error(`Failed to list projects: ${listResp.statusText}`);
  }

  const projects = (await listResp.json()) as { results?: Array<{ id: number; title: string }> };
  const existing = projects.results?.find(
    (p) => p.title === "RallyCut Ground Truth"
  );

  if (existing) {
    return existing.id;
  }

  // Create new project
  const createResp = await fetch(`${config.url}/api/projects`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      title: "RallyCut Ground Truth",
      label_config: LABEL_CONFIG,
    }),
  });

  if (!createResp.ok) {
    throw new Error(`Failed to create project: ${createResp.statusText}`);
  }

  const project = (await createResp.json()) as { id: number };
  return project.id;
}

/**
 * Create a labeling task in Label Studio.
 */
async function createTask(
  config: LabelStudioConfig,
  projectId: number,
  videoUrl: string,
  predictions: any[]
): Promise<number> {
  const headers = {
    Authorization: `Token ${config.apiKey}`,
    "Content-Type": "application/json",
  };

  // First create the task
  const taskData = {
    data: { video: videoUrl },
  };

  const resp = await fetch(`${config.url}/api/projects/${projectId}/tasks`, {
    method: "POST",
    headers,
    body: JSON.stringify(taskData),
  });

  if (!resp.ok) {
    throw new Error(`Failed to create task: ${resp.statusText}`);
  }

  const task = (await resp.json()) as { id: number };

  // Then add predictions separately (Label Studio API quirk)
  if (predictions.length > 0) {
    console.log(`[LabelStudio] Adding ${predictions.length} predictions to task ${task.id}`);
    const predResp = await fetch(`${config.url}/api/predictions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        task: task.id,
        model_version: "rallycut",
        result: predictions,
      }),
    });

    if (!predResp.ok) {
      const errorText = await predResp.text();
      console.error(`[LabelStudio] Failed to add predictions: ${predResp.status} ${errorText}`);
    } else {
      console.log(`[LabelStudio] Predictions added successfully`);
    }
  }

  return task.id;
}

/**
 * Get annotations for a task.
 */
async function getTaskAnnotations(
  config: LabelStudioConfig,
  taskId: number
): Promise<any[]> {
  const headers = {
    Authorization: `Token ${config.apiKey}`,
  };

  const resp = await fetch(`${config.url}/api/tasks/${taskId}`, { headers });
  if (!resp.ok) {
    throw new Error(`Failed to get task: ${resp.statusText}`);
  }

  const task = (await resp.json()) as { annotations?: Array<{ id?: number; result?: any[] }> };
  const annotations = task.annotations || [];

  if (annotations.length === 0) {
    return [];
  }

  // Get latest annotation
  const latest = annotations.reduce((a: any, b: any) =>
    (a.id || 0) > (b.id || 0) ? a : b
  );

  return latest.result || [];
}

/**
 * Parse Label Studio annotations to ground truth format.
 */
function parseAnnotations(
  annotations: any[],
  videoWidth: number,
  videoHeight: number
): GroundTruthData {
  const positions: GroundTruthData["positions"] = [];
  let maxFrame = 0;

  for (const result of annotations) {
    if (result.type !== "videorectangle") continue;

    const value = result.value || {};
    const sequence = value.sequence || [];
    const labels = value.labels || ["player"];
    const label = labels[0] || "player";

    // Extract track ID from result ID (format: "player_0", "ball_0", etc.)
    const resultId = String(result.id || "0");
    let trackId: number;
    try {
      const parts = resultId.split("_");
      trackId = parseInt(parts[parts.length - 1], 10);
      // If parsing fails, generate a simple hash from the string
      if (isNaN(trackId)) {
        trackId = Math.abs(
          resultId.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0)
        ) % 1000;
      }
    } catch {
      trackId = 0;
    }

    for (const keyframe of sequence) {
      if (!keyframe.enabled) continue;

      const frame = Number(keyframe.frame || 1) - 1; // Convert to 0-indexed

      // Convert percentage to normalized
      const xPct = Number(keyframe.x || 0);
      const yPct = Number(keyframe.y || 0);
      const wPct = Number(keyframe.width || 5);
      const hPct = Number(keyframe.height || 10);

      const x = (xPct + wPct / 2) / 100;
      const y = (yPct + hPct / 2) / 100;
      const width = wPct / 100;
      const height = hPct / 100;

      positions.push({
        frameNumber: frame,
        trackId,
        label,
        x,
        y,
        width,
        height,
        confidence: 1.0,
      });

      maxFrame = Math.max(maxFrame, frame);
    }
  }

  return {
    positions,
    frameCount: maxFrame + 1,
    videoWidth,
    videoHeight,
  };
}

interface ExportOptions {
  config?: Partial<LabelStudioConfig>;
  forceRegenerate?: boolean;
}

/**
 * Export tracking predictions to Label Studio for labeling.
 */
export async function exportToLabelStudio(
  rallyId: string,
  userId: string,
  videoUrl: string,
  options?: ExportOptions
): Promise<ExportResult> {
  const { config, forceRegenerate = false } = options || {};
  // Get Label Studio config
  const lsConfig: LabelStudioConfig = {
    url: config?.url || LABEL_STUDIO_URL,
    apiKey: config?.apiKey || LABEL_STUDIO_API_KEY || "",
    projectId: config?.projectId,
  };

  if (!lsConfig.apiKey) {
    return {
      success: false,
      error: "Label Studio API key not configured",
    };
  }

  try {
    // Get rally and verify ownership
    const rally = await prisma.rally.findUnique({
      where: { id: rallyId },
      include: { video: true },
    });

    if (!rally) {
      return { success: false, error: "Rally not found" };
    }

    if (rally.video.userId !== userId) {
      return { success: false, error: "Permission denied" };
    }

    // Get player track data
    const playerTrack = await prisma.playerTrack.findUnique({
      where: { rallyId },
    });

    if (!playerTrack || playerTrack.status !== "COMPLETED") {
      return {
        success: false,
        error: "No tracking data available. Run player tracking first.",
      };
    }

    // Check if we already have a task for this rally (unless force regenerate)
    if (playerTrack.groundTruthTaskId && !forceRegenerate) {
      const projectId = lsConfig.projectId || (await getOrCreateProject(lsConfig));
      const taskUrl = `${lsConfig.url}/projects/${projectId}/data?task=${playerTrack.groundTruthTaskId}`;
      console.log(`[LabelStudio] Reusing existing task ${playerTrack.groundTruthTaskId}`);
      return {
        success: true,
        taskId: playerTrack.groundTruthTaskId,
        projectId,
        taskUrl,
      };
    }

    if (forceRegenerate && playerTrack.groundTruthTaskId) {
      console.log(`[LabelStudio] Force regenerating task (old task ID: ${playerTrack.groundTruthTaskId})`);
    }

    // Get or create project
    const projectId = lsConfig.projectId || (await getOrCreateProject(lsConfig));

    // Build predictions with rally start time
    const fps = playerTrack.fps || 30;
    const rallyStartTime = rally.startMs / 1000;
    const rallyEndTime = rally.endMs / 1000;
    const videoDuration = (rally.video.durationMs || 0) / 1000;
    const predictions = buildPredictions(playerTrack, rallyStartTime, rallyEndTime, fps, videoDuration);

    console.log(`[LabelStudio] Rally: ${rallyStartTime}s - ${rallyEndTime}s, video duration ${videoDuration}s, FPS: ${fps}`);
    console.log(`[LabelStudio] Building predictions from ${(playerTrack.positionsJson as any[])?.length || 0} positions, ${(playerTrack.ballPositionsJson as any[])?.length || 0} ball positions`);

    // Debug: Log first position timing calculation
    const positions = (playerTrack.positionsJson || []) as any[];
    if (Array.isArray(positions) && positions.length > 0) {
      const firstPos = [...positions].sort((a, b) => a.frameNumber - b.frameNumber)[0];
      const firstTime = rallyStartTime + firstPos.frameNumber / fps;
      const firstFrame = Math.round(firstTime * LABEL_STUDIO_FPS) + 1;
      console.log(`[LabelStudio] First position: frameNumber=${firstPos.frameNumber} (segment-relative)`);
      console.log(`[LabelStudio] Calculated: time=${firstTime.toFixed(3)}s, LS frame=${firstFrame} (at ${LABEL_STUDIO_FPS}fps)`);
    }

    console.log(`[LabelStudio] Created ${predictions.length} prediction tracks`);

    // Create task
    const taskId = await createTask(lsConfig, projectId, videoUrl, predictions);

    // Save task ID to player track for future reference
    await prisma.playerTrack.update({
      where: { rallyId },
      data: { groundTruthTaskId: taskId },
    });

    const taskUrl = `${lsConfig.url}/projects/${projectId}/data?task=${taskId}`;

    return {
      success: true,
      taskId,
      projectId,
      taskUrl,
    };
  } catch (error) {
    console.error("Error exporting to Label Studio:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

/**
 * Import annotations from Label Studio as ground truth.
 */
export async function importFromLabelStudio(
  rallyId: string,
  userId: string,
  taskId: number,
  config?: Partial<LabelStudioConfig>
): Promise<ImportResult> {
  // Get Label Studio config
  const lsConfig: LabelStudioConfig = {
    url: config?.url || LABEL_STUDIO_URL,
    apiKey: config?.apiKey || LABEL_STUDIO_API_KEY || "",
  };

  if (!lsConfig.apiKey) {
    return {
      success: false,
      error: "Label Studio API key not configured",
    };
  }

  try {
    // Get rally and verify ownership
    const rally = await prisma.rally.findUnique({
      where: { id: rallyId },
      include: { video: true },
    });

    if (!rally) {
      return { success: false, error: "Rally not found" };
    }

    if (rally.video.userId !== userId) {
      return { success: false, error: "Permission denied" };
    }

    // Get annotations from Label Studio
    const annotations = await getTaskAnnotations(lsConfig, taskId);

    if (annotations.length === 0) {
      return {
        success: false,
        error: "No annotations found. Complete labeling in Label Studio first.",
      };
    }

    // Parse to ground truth format
    const groundTruth = parseAnnotations(
      annotations,
      rally.video.width || 1920,
      rally.video.height || 1080
    );

    // Save ground truth to player track
    await prisma.playerTrack.update({
      where: { rallyId },
      data: {
        groundTruthJson: JSON.parse(JSON.stringify(groundTruth)),
        groundTruthTaskId: taskId,
        groundTruthSyncedAt: new Date(),
      },
    });

    const playerPositions = groundTruth.positions.filter(
      (p) => p.label.startsWith("player")
    );
    const ballPositions = groundTruth.positions.filter(
      (p) => p.label === "ball"
    );

    return {
      success: true,
      playerCount: playerPositions.length,
      ballCount: ballPositions.length,
      frameCount: groundTruth.frameCount,
    };
  } catch (error) {
    console.error("Error importing from Label Studio:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

/**
 * Get Label Studio integration status for a rally.
 */
export async function getLabelStudioStatus(
  rallyId: string,
  userId: string
): Promise<{
  hasTrackingData: boolean;
  hasGroundTruth: boolean;
  taskId?: number;
  syncedAt?: Date;
}> {
  const rally = await prisma.rally.findUnique({
    where: { id: rallyId },
    include: { video: true },
  });

  if (!rally || rally.video.userId !== userId) {
    return { hasTrackingData: false, hasGroundTruth: false };
  }

  const playerTrack = await prisma.playerTrack.findUnique({
    where: { rallyId },
  });

  return {
    hasTrackingData: playerTrack?.status === "COMPLETED",
    hasGroundTruth: !!playerTrack?.groundTruthJson,
    taskId: playerTrack?.groundTruthTaskId ?? undefined,
    syncedAt: playerTrack?.groundTruthSyncedAt ?? undefined,
  };
}
