"use client";

import { useEffect, useMemo, useState } from "react";
import Controls, { type DisplayMode } from "@/components/Controls";
import FocusZone, { type FocusPoint } from "@/components/FocusZone";
import VideoPanel, { type FocusSource } from "@/components/VideoPanel";
import VitalsPanel from "@/components/VitalsPanel";

type FramePayload = {
  clipKey: string;
  fps: number;
  frames: string[];
  masks: string[];
  confidenceMasks?: Array<string | null>;
};

export default function DashboardPage() {
  const [mode, setMode] = useState<DisplayMode>("RAW");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [payload, setPayload] = useState<FramePayload | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [confidenceOverlayEnabled, setConfidenceOverlayEnabled] = useState(true);
  const [pipelineStatus, setPipelineStatus] = useState("Live session");
  const [focusPoint, setFocusPoint] = useState<FocusPoint>({ x: 0.5, y: 0.5 });
  const [focusSource, setFocusSource] = useState<FocusSource>("raw");

  useEffect(() => {
    let cancelled = false;

    async function loadFrames() {
      const response = await fetch("/api/frames?video=video01");
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
      mask: payload.masks[frameIndex] ?? payload.masks[0],
      confidence: payload.confidenceMasks?.[frameIndex] ?? null
    };
  }, [frameIndex, payload]);

  const focusSrc = focusSource === "mask" ? currentFrame?.mask ?? null : currentFrame?.raw ?? null;

  return (
    <main className="clinical-shell">
      <section className={sidebarCollapsed ? "monitor-frame sidebar-collapsed" : "monitor-frame"}>
        <aside className={sidebarCollapsed ? "sidebar-shell collapsed" : "sidebar-shell"}>
          <div className="sidebar-topbar">
            <button
              type="button"
              className={sidebarCollapsed ? "sidebar-toggle" : "sidebar-toggle close"}
              onClick={() => setSidebarCollapsed((value) => !value)}
              aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              <span className="sidebar-toggle-icon" aria-hidden="true">
                {sidebarCollapsed ? "\u203A" : "\u2039"}
              </span>
            </button>
          </div>

          <section className="sidebar-card">
            <span className={sidebarCollapsed ? "sidebar-card-title hidden" : "sidebar-card-title"}>
              Modes
            </span>
            <Controls
              mode={mode}
              onModeChange={setMode}
              playing={playing}
              onTogglePlaying={() => setPlaying((value) => !value)}
              confidenceOverlayEnabled={confidenceOverlayEnabled}
              onToggleConfidenceOverlay={() => setConfidenceOverlayEnabled((value) => !value)}
              collapsed={sidebarCollapsed}
              showActions={false}
            />
          </section>

          <section className="sidebar-actions-block">
            <Controls
              mode={mode}
              onModeChange={setMode}
              playing={playing}
              onTogglePlaying={() => setPlaying((value) => !value)}
              confidenceOverlayEnabled={confidenceOverlayEnabled}
              onToggleConfidenceOverlay={() => setConfidenceOverlayEnabled((value) => !value)}
              collapsed={sidebarCollapsed}
              showModes={false}
              showConfidenceToggle={false}
            />
          </section>
        </aside>

        <section className="content-shell">
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
            confidenceSrc={currentFrame?.confidence ?? null}
            confidenceOverlayEnabled={confidenceOverlayEnabled}
            focusPoint={focusPoint}
            onFocusChange={(source, point) => {
              setFocusSource(source);
              setFocusPoint(point);
            }}
          />

          <section className="info-grid">
            <VitalsPanel />

            <article className="info-card">
              <span className="mini-heading">Focused area</span>
              <div className="focus-card">
                <FocusZone src={focusSrc} point={focusPoint} onFocusChange={setFocusPoint} />
              </div>
            </article>

            <article className="info-card">
              <div className="legend-header">
                <span className="mini-heading">Confidence legend</span>
                <button
                  type="button"
                  className={
                    confidenceOverlayEnabled ? "legend-switch active" : "legend-switch"
                  }
                  onClick={() => setConfidenceOverlayEnabled((value) => !value)}
                  aria-pressed={confidenceOverlayEnabled}
                  aria-label="Toggle confidence overlay"
                >
                  <span className="legend-switch-track">
                    <span className="legend-switch-thumb" />
                  </span>
                  <span className="legend-switch-label">
                    {confidenceOverlayEnabled ? "On" : "Off"}
                  </span>
                </button>
              </div>
              <ul className="legend-list">
                <li>
                  <i className="legend-dot high" />
                  <span>High</span>
                  <strong>80-100%</strong>
                </li>
                <li>
                  <i className="legend-dot medium" />
                  <span>Medium</span>
                  <strong>50-79%</strong>
                </li>
                <li>
                  <i className="legend-dot low" />
                  <span>Low</span>
                  <strong>&lt; 50%</strong>
                </li>
              </ul>
            </article>
          </section>
        </section>
      </section>
    </main>
  );
}
