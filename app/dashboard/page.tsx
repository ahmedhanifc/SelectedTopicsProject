"use client";

import { useEffect, useMemo, useState } from "react";
import Controls, { type DisplayMode } from "@/components/Controls";
import VideoPanel from "@/components/VideoPanel";
import VitalsPanel from "@/components/VitalsPanel";

type FramePayload = {
  clipKey: string;
  fps: number;
  frames: string[];
  masks: string[];
};

export default function DashboardPage() {
  const [mode, setMode] = useState<DisplayMode>("RAW");
  const [payload, setPayload] = useState<FramePayload | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [confidenceOverlayEnabled, setConfidenceOverlayEnabled] = useState(true);
  const [pipelineStatus, setPipelineStatus] = useState("Live session");

  useEffect(() => {
    let cancelled = false;

    async function loadFrames() {
      const response = await fetch("/api/frames?video=video01&clip=video01_00160");
      if (!response.ok) {
        throw new Error("Failed to load frame sequence");
      }
      const data = (await response.json()) as FramePayload;
      if (!cancelled) {
        setPayload(data);
        setPipelineStatus("Live session");
      }
    }

    loadFrames().catch((error: unknown) => {
      if (!cancelled) {
        const message = error instanceof Error ? error.message : "Frame route unavailable";
        setPipelineStatus(message);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!payload || !playing || payload.frames.length === 0) {
      return;
    }

    const interval = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % payload.frames.length);
    }, Math.round(1000 / payload.fps));

    return () => window.clearInterval(interval);
  }, [payload, playing]);

  const currentFrame = useMemo(() => {
    if (!payload || payload.frames.length === 0) {
      return null;
    }
    return {
      raw: payload.frames[frameIndex] ?? payload.frames[0],
      mask: payload.masks[frameIndex] ?? payload.masks[0]
    };
  }, [frameIndex, payload]);

  return (
    <main className="clinical-shell">
      <section className="monitor-frame">
        <header className="monitor-header">
          <div>
            <h1>John Doe: Cholecystectomy</h1>
          </div>
          <div className="session-meta">
            <span>Operating room 4</span>
            <strong>
              {pipelineStatus}
              <i />
            </strong>
          </div>
        </header>

        <VideoPanel
          mode={mode}
          rawSrc={currentFrame?.raw ?? null}
          maskSrc={currentFrame?.mask ?? null}
          confidenceOverlayEnabled={confidenceOverlayEnabled}
        />

        <section className="info-grid">
          <VitalsPanel />

          <article className="info-card">
            <span className="mini-heading">AI focus zone</span>
            <div className="focus-card">
              {currentFrame?.raw ? <img src={currentFrame.raw} alt="AI focus region" /> : null}
              <div className="focus-caption">
                <span>Micro-focus crop</span>
                <strong>2.5x zoom</strong>
              </div>
            </div>
          </article>

          <article className="info-card">
            <span className="mini-heading">Robotic telemetry</span>
            <div className="telemetry-list">
              <div className="telemetry-row">
                <label>Arm α force</label>
                <div className="meter-track">
                  <span style={{ width: "56%" }} />
                </div>
                <strong>1.2N</strong>
              </div>
              <div className="telemetry-row">
                <label>Arm β force</label>
                <div className="meter-track">
                  <span style={{ width: "34%" }} />
                </div>
                <strong>0.8N</strong>
              </div>
              <div className="telemetry-row">
                <label>Latency</label>
                <div className="meter-track quiet">
                  <span style={{ width: "12%" }} />
                </div>
                <strong>12ms</strong>
              </div>
            </div>
          </article>

          <article className="info-card">
            <span className="mini-heading">Confidence legend</span>
            <ul className="legend-list">
              <li>
                <i className="legend-dot high" />
                <span>High</span>
                <strong>95-100%</strong>
              </li>
              <li>
                <i className="legend-dot medium" />
                <span>Medium</span>
                <strong>70-94%</strong>
              </li>
              <li>
                <i className="legend-dot low" />
                <span>Low</span>
                <strong>&lt; 70%</strong>
              </li>
            </ul>
          </article>
        </section>

        <Controls
          mode={mode}
          onModeChange={setMode}
          playing={playing}
          onTogglePlaying={() => setPlaying((value) => !value)}
          confidenceOverlayEnabled={confidenceOverlayEnabled}
          onToggleConfidenceOverlay={() => setConfidenceOverlayEnabled((value) => !value)}
        />
      </section>
    </main>
  );
}
