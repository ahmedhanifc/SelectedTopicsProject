"use client";

import type { RefObject } from "react";
import { useEffect, useRef } from "react";
import type { DisplayMode } from "@/components/Controls";

type VideoPanelProps = {
  mode: DisplayMode;
  rawSrc: string | null;
  maskSrc: string | null;
  confidenceOverlayEnabled: boolean;
};

function AnalysisMedia({
  rawSrc,
  maskSrc,
  canvasRef
}: {
  rawSrc: string | null;
  maskSrc: string | null;
  canvasRef: RefObject<HTMLCanvasElement | null>;
}) {
  return <canvas ref={canvasRef} className="blend-canvas full-canvas" />;
}

export default function VideoPanel({
  mode,
  rawSrc,
  maskSrc,
  confidenceOverlayEnabled
}: VideoPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (mode !== "BLENDED" || !rawSrc || !maskSrc || !canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const rawImage = new Image();
    const maskImage = new Image();

    let rawReady = false;
    let maskReady = false;

    const draw = () => {
      if (!rawReady || !maskReady) {
        return;
      }

      canvas.width = rawImage.width;
      canvas.height = rawImage.height;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.globalAlpha = 1;
      context.drawImage(rawImage, 0, 0);
      context.globalAlpha = confidenceOverlayEnabled ? 0.42 : 0.18;
      context.drawImage(maskImage, 0, 0, canvas.width, canvas.height);
      context.globalAlpha = 1;
    };

    rawImage.onload = () => {
      rawReady = true;
      draw();
    };
    maskImage.onload = () => {
      maskReady = true;
      draw();
    };

    rawImage.src = rawSrc;
    maskImage.src = maskSrc;
  }, [confidenceOverlayEnabled, maskSrc, mode, rawSrc]);

  const stageLabel =
    mode === "RAW"
      ? "Raw feed"
      : mode === "SEGMENTED"
        ? "Segmented view"
        : mode === "SPLIT"
          ? "Split comparison"
          : "Blended overlay";

  return (
    <section className="monitor-panel">
      {mode === "SPLIT" ? (
        <div className="stage-grid">
          <article className="stage-card raw-card">
            <span className="stage-label">Raw feed</span>
            {rawSrc ? <img className="surgical-image" src={rawSrc} alt="Raw surgical feed" /> : null}
          </article>

          <article className="stage-card analysis-card">
            <div className="analysis-header">
              <span className="mode-pill">Segmented view</span>
            </div>
            {maskSrc ? <img className="surgical-image" src={maskSrc} alt="Segmented frame" /> : null}
          </article>
        </div>
      ) : (
        <article className={`stage-card ${mode === "RAW" ? "raw-card" : "analysis-card"} full-stage`}>
          <div className="analysis-header">
            <span className={mode === "RAW" ? "stage-label top-label" : "mode-pill"}>{stageLabel}</span>
          </div>

          {mode === "RAW" && rawSrc ? <img className="surgical-image" src={rawSrc} alt="Raw surgical feed" /> : null}

          {mode === "SEGMENTED" ? (
            <>{maskSrc ? <img className="surgical-image" src={maskSrc} alt="Segmented frame" /> : null}</>
          ) : null}

          {mode === "BLENDED" ? (
            <AnalysisMedia rawSrc={rawSrc} maskSrc={maskSrc} canvasRef={canvasRef} />
          ) : null}
        </article>
      )}
    </section>
  );
}
