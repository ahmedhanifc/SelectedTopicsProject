import path from "node:path";
import { NextResponse } from "next/server";
import { runPythonPipeline } from "@/lib/pythonRunner";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = (await request.json().catch(() => ({}))) as { input?: string };
    const input = body.input ?? "public/data/video01/video01_00160";
    const absoluteInput = path.isAbsolute(input) ? input : path.join(process.cwd(), input);
    const result = await runPythonPipeline(absoluteInput);
    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown python route error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
