import { prisma } from '../lib/prisma';
import { reresolveRallyGt } from '../services/actionGroundTruthService';

async function main() {
  const videoId = process.env.VIDEO_ID;
  if (!videoId) { console.error('Set VIDEO_ID=<uuid>'); process.exit(1); }
  const rallies = await prisma.rally.findMany({
    where: { videoId },
    include: { playerTrack: true },
  });
  for (const rally of rallies) {
    if (!rally.playerTrack?.rawPositionsJson) continue;
    const rawPositions = rally.playerTrack.rawPositionsJson as unknown as Array<{
      frameNumber: number; trackId: number; x: number; y: number;
      width: number; height: number; confidence?: number; embedding?: number[] | null;
    }>;
    await prisma.$transaction(async (tx) => { await reresolveRallyGt(tx, rally.id, rawPositions); });
    process.stdout.write('.');
  }
  console.log(' done');
  await prisma.$disconnect();
}
main().catch((e) => { console.error(e); process.exit(1); });
