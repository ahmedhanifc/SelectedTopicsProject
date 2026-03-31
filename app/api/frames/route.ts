import { NextResponse } from "next/server";
import { getFrameSequence } from "@/lib/frameStreamer";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const video = searchParams.get("video") ?? "video01";
  const clip = searchParams.get("clip") ?? "video01";

  try {
    const payload = await getFrameSequence({ video, clip });
    return NextResponse.json(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown frame route error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
