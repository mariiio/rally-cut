type FrameField = 'frame' | 'frameNumber';

type AnyFrameEntry = Record<string, unknown>;

export type SlicePlayerTrackInput = {
  fps: number;
  frameCount: number;
  courtSplitY: number | null;
  processingTimeMs: number | null;
  modelVersion: string | null;
  status: string;
  needsRetrack: boolean;
  positionsJson: AnyFrameEntry[] | null;
  rawPositionsJson: AnyFrameEntry[] | null;
  ballPositionsJson: AnyFrameEntry[] | null;
  contactsJson: AnyFrameEntry[] | null;
  actionsJson: AnyFrameEntry[] | null;
  groundTruthJson: AnyFrameEntry[] | null;
  actionGroundTruthJson: AnyFrameEntry[] | null;
  qualityReportJson: unknown;
};

export type SlicedPlayerTrack = Omit<SlicePlayerTrackInput, 'frameCount'> & {
  frameCount: number;
  detectionRate: number;
  avgConfidence: number;
  avgPlayerCount: number;
  uniqueTrackCount: number;
};

function partitionByFrame(
  arr: AnyFrameEntry[] | null,
  field: FrameField,
  firstEndFrame: number,
  secondStartFrame: number,
): { firstArr: AnyFrameEntry[]; secondArr: AnyFrameEntry[] } {
  if (!arr) return { firstArr: [], secondArr: [] };
  const firstArr: AnyFrameEntry[] = [];
  const secondArr: AnyFrameEntry[] = [];
  for (const entry of arr) {
    const f = entry[field];
    if (typeof f !== 'number') continue;
    if (f < firstEndFrame) firstArr.push(entry);
    else if (f >= secondStartFrame) {
      secondArr.push({ ...entry, [field]: f - secondStartFrame });
    }
    // frames in [firstEndFrame, secondStartFrame) are discarded
  }
  return { firstArr, secondArr };
}

function recomputeMetadata(
  positionsJson: AnyFrameEntry[],
  frameCount: number,
): Pick<SlicedPlayerTrack, 'detectionRate' | 'avgConfidence' | 'avgPlayerCount' | 'uniqueTrackCount'> {
  const framesWithDetection = new Set<number>();
  const trackIds = new Set<string | number>();
  let confSum = 0;
  let confN = 0;
  const playersPerFrame = new Map<number, Set<string | number>>();
  for (const p of positionsJson) {
    const frame = p.frameNumber as number;
    const tid = (p.trackId ?? p.track_id) as string | number | undefined;
    framesWithDetection.add(frame);
    if (tid !== undefined) {
      trackIds.add(tid);
      if (!playersPerFrame.has(frame)) playersPerFrame.set(frame, new Set());
      playersPerFrame.get(frame)!.add(tid);
    }
    if (typeof p.confidence === 'number') { confSum += p.confidence; confN++; }
  }
  const detectionRate = frameCount > 0 ? framesWithDetection.size / frameCount : 0;
  const avgConfidence = confN > 0 ? confSum / confN : 0;
  const avgPlayerCount = playersPerFrame.size > 0
    ? [...playersPerFrame.values()].reduce((a, s) => a + s.size, 0) / playersPerFrame.size
    : 0;
  return {
    detectionRate,
    avgConfidence,
    avgPlayerCount,
    uniqueTrackCount: trackIds.size,
  };
}

export function slicePlayerTrack(
  pt: SlicePlayerTrackInput,
  firstEndFrame: number,
  secondStartFrame: number,
): { first: SlicedPlayerTrack; second: SlicedPlayerTrack } {
  const pos = partitionByFrame(pt.positionsJson, 'frameNumber', firstEndFrame, secondStartFrame);
  const raw = partitionByFrame(pt.rawPositionsJson, 'frameNumber', firstEndFrame, secondStartFrame);
  const ball = partitionByFrame(pt.ballPositionsJson, 'frameNumber', firstEndFrame, secondStartFrame);
  const contacts = partitionByFrame(pt.contactsJson, 'frame', firstEndFrame, secondStartFrame);
  const actions = partitionByFrame(pt.actionsJson, 'frame', firstEndFrame, secondStartFrame);
  const gt = partitionByFrame(pt.groundTruthJson, 'frame', firstEndFrame, secondStartFrame);
  const actGt = partitionByFrame(pt.actionGroundTruthJson, 'frame', firstEndFrame, secondStartFrame);

  const firstCount = firstEndFrame;
  const secondCount = Math.max(0, pt.frameCount - secondStartFrame);

  const firstMeta = recomputeMetadata(pos.firstArr, firstCount);
  const secondMeta = recomputeMetadata(pos.secondArr, secondCount);

  const base = {
    fps: pt.fps,
    courtSplitY: pt.courtSplitY,
    processingTimeMs: pt.processingTimeMs,
    modelVersion: pt.modelVersion,
    status: pt.status,
    needsRetrack: pt.needsRetrack,
    qualityReportJson: pt.qualityReportJson,
  };

  return {
    first: {
      ...base,
      frameCount: firstCount,
      ...firstMeta,
      positionsJson: pos.firstArr,
      rawPositionsJson: raw.firstArr,
      ballPositionsJson: ball.firstArr,
      contactsJson: contacts.firstArr,
      actionsJson: actions.firstArr,
      groundTruthJson: pt.groundTruthJson ? gt.firstArr : null,
      actionGroundTruthJson: pt.actionGroundTruthJson ? actGt.firstArr : null,
    },
    second: {
      ...base,
      frameCount: secondCount,
      ...secondMeta,
      positionsJson: pos.secondArr,
      rawPositionsJson: raw.secondArr,
      ballPositionsJson: ball.secondArr,
      contactsJson: contacts.secondArr,
      actionsJson: actions.secondArr,
      groundTruthJson: pt.groundTruthJson ? gt.secondArr : null,
      actionGroundTruthJson: pt.actionGroundTruthJson ? actGt.secondArr : null,
    },
  };
}
