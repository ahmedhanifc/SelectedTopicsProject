"use client";

import type { PointerEvent as ReactPointerEvent, RefObject } from "react";
import { useEffect, useRef, useState } from "react";
import type { DisplayMode } from "@/components/Controls";
import type { FocusPoint } from "@/components/FocusZone";

export type FocusSource = "raw" | "mask";

type VideoPanelProps = {
  mode: DisplayMode;
  rawSrc: string | null;
  maskSrc: string | null;
  confidenceOverlayEnabled: boolean;
  focusPoint: FocusPoint;
  onFocusChange: (source: FocusSource, point: FocusPoint) => void;
};

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function getContainedBounds(
  mediaElement: HTMLElement,
  mediaWidth: number,
  mediaHeight: number
): { offsetX: number; offsetY: number; drawWidth: number; drawHeight: number } {
  const rect = mediaElement.getBoundingClientRect();
  const styles = window.getComputedStyle(mediaElement);
  const paddingLeft = Number.parseFloat(styles.paddingLeft) || 0;
  const paddingRight = Number.parseFloat(styles.paddingRight) || 0;
  const paddingTop = Number.parseFloat(styles.paddingTop) || 0;
  const paddingBottom = Number.parseFloat(styles.paddingBottom) || 0;

  const contentWidth = Math.max(rect.width - paddingLeft - paddingRight, 1);
  const contentHeight = Math.max(rect.height - paddingTop - paddingBottom, 1);
  const mediaAspect = mediaWidth / mediaHeight;
  const boxAspect = contentWidth / contentHeight;

  let drawWidth = contentWidth;
  let drawHeight = contentHeight;
  let offsetX = paddingLeft;
  let offsetY = paddingTop;

  if (mediaAspect > boxAspect) {
    drawHeight = drawWidth / mediaAspect;
    offsetY += (contentHeight - drawHeight) / 2;
  } else {
    drawWidth = drawHeight * mediaAspect;
    offsetX += (contentWidth - drawWidth) / 2;
  }

  return {
    offsetX,
    offsetY,
    drawWidth,
    drawHeight
  };
}

function getContainedPoint(
  clientX: number,
  clientY: number,
  mediaElement: HTMLElement,
  mediaWidth: number,
  mediaHeight: number
): FocusPoint {
  const rect = mediaElement.getBoundingClientRect();
  const { offsetX, offsetY, drawWidth, drawHeight } = getContainedBounds(mediaElement, mediaWidth, mediaHeight);

  return {
    x: clamp01((clientX - rect.left - offsetX) / drawWidth),
    y: clamp01((clientY - rect.top - offsetY) / drawHeight)
  };
}

function InteractiveImage({
  src,
  alt,
  source,
  focusPoint,
  showTarget,
  onFocusChange
}: {
  src: string | null;
  alt: string;
  source: FocusSource;
  focusPoint: FocusPoint;
  showTarget?: boolean;
  onFocusChange: (source: FocusSource, point: FocusPoint) => void;
}) {
  const imageRef = useRef<HTMLImageElement | null>(null);
  const dragActiveRef = useRef(false);
  const [targetStyle, setTargetStyle] = useState<{ left: number; top: number } | null>(null);

  useEffect(() => {
    const image = imageRef.current;
    if (!image) {
      return;
    }

    const updateTarget = () => {
      if (image.naturalWidth === 0 || image.naturalHeight === 0) {
        return;
      }

      const { offsetX, offsetY, drawWidth, drawHeight } = getContainedBounds(
        image,
        image.naturalWidth,
        image.naturalHeight
      );

      setTargetStyle({
        left: offsetX + focusPoint.x * drawWidth,
        top: offsetY + focusPoint.y * drawHeight
      });
    };

    updateTarget();

    const resizeObserver = new ResizeObserver(() => {
      updateTarget();
    });

    resizeObserver.observe(image);
    image.addEventListener("load", updateTarget);

    return () => {
      resizeObserver.disconnect();
      image.removeEventListener("load", updateTarget);
    };
  }, [focusPoint, src]);

  const updateFocus = (event: ReactPointerEvent<HTMLImageElement>) => {
    const image = imageRef.current;
    if (!image || image.naturalWidth === 0 || image.naturalHeight === 0) {
      return;
    }

    onFocusChange(source, getContainedPoint(event.clientX, event.clientY, image, image.naturalWidth, image.naturalHeight));
  };

  if (!src) {
    return null;
  }

  return (
    <>
    <img
      ref={imageRef}
      className="surgical-image interactive-stage"
      src={src}
      alt={alt}
      draggable={false}
      onPointerDown={(event) => {
        dragActiveRef.current = true;
        event.currentTarget.setPointerCapture(event.pointerId);
        updateFocus(event);
      }}
        onPointerMove={(event) => {
          if (!dragActiveRef.current) {
            return;
          }
          updateFocus(event);
        }}
        onPointerUp={(event) => {
          dragActiveRef.current = false;
          event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => {
          dragActiveRef.current = false;
        }}
      />
      {showTarget && targetStyle ? (
        <span className="focus-target" style={{ left: `${targetStyle.left}px`, top: `${targetStyle.top}px` }} />
      ) : null}
    </>
  );
}

function AnalysisMedia({
  rawSrc,
  maskSrc,
  canvasRef,
  focusPoint,
  showTarget,
  onFocusChange
}: {
  rawSrc: string | null;
  maskSrc: string | null;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  focusPoint: FocusPoint;
  showTarget?: boolean;
  onFocusChange: (source: FocusSource, point: FocusPoint) => void;
}) {
  const dragActiveRef = useRef(false);
  const [targetStyle, setTargetStyle] = useState<{ left: number; top: number } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) {
      return;
    }

    const updateTarget = () => {
      const { offsetX, offsetY, drawWidth, drawHeight } = getContainedBounds(canvas, canvas.width, canvas.height);
      setTargetStyle({
        left: offsetX + focusPoint.x * drawWidth,
        top: offsetY + focusPoint.y * drawHeight
      });
    };

    updateTarget();

    const resizeObserver = new ResizeObserver(() => {
      updateTarget();
    });

    resizeObserver.observe(canvas);

    return () => {
      resizeObserver.disconnect();
    };
  }, [canvasRef, focusPoint, rawSrc, maskSrc]);

  const updateFocus = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) {
      return;
    }

    onFocusChange("raw", getContainedPoint(event.clientX, event.clientY, canvas, canvas.width, canvas.height));
  };

  return (
    <>
      <canvas
        ref={canvasRef}
        className="blend-canvas full-canvas interactive-stage"
        onPointerDown={(event) => {
          dragActiveRef.current = true;
          event.currentTarget.setPointerCapture(event.pointerId);
          updateFocus(event);
        }}
        onPointerMove={(event) => {
          if (!dragActiveRef.current) {
            return;
          }
          updateFocus(event);
        }}
        onPointerUp={(event) => {
          dragActiveRef.current = false;
          event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => {
          dragActiveRef.current = false;
        }}
      />
      {showTarget && targetStyle ? (
        <span className="focus-target" style={{ left: `${targetStyle.left}px`, top: `${targetStyle.top}px` }} />
      ) : null}
    </>
  );
}

export default function VideoPanel({
  mode,
  rawSrc,
  maskSrc,
  confidenceOverlayEnabled,
  focusPoint,
  onFocusChange
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
            <div className="analysis-header">
              <span className="stage-label">Raw feed</span>
            </div>
            <InteractiveImage
              src={rawSrc}
              alt="Raw surgical feed"
              source="raw"
              focusPoint={focusPoint}
              showTarget
              onFocusChange={onFocusChange}
            />
          </article>

          <article className="stage-card analysis-card">
            <div className="analysis-header">
              <span className="stage-label">Segmented view</span>
            </div>
            <InteractiveImage
              src={maskSrc}
              alt="Segmented frame"
              source="mask"
              focusPoint={focusPoint}
              onFocusChange={onFocusChange}
            />
          </article>
        </div>
      ) : (
        <article className={`stage-card ${mode === "RAW" ? "raw-card" : "analysis-card"} full-stage`}>
          <div className="analysis-header">
            <span className="stage-label">{stageLabel}</span>
          </div>

          {mode === "RAW" ? (
            <InteractiveImage
              src={rawSrc}
              alt="Raw surgical feed"
              source="raw"
              focusPoint={focusPoint}
              showTarget
              onFocusChange={onFocusChange}
            />
          ) : null}

          {mode === "SEGMENTED" ? (
            <InteractiveImage
              src={maskSrc}
              alt="Segmented frame"
              source="mask"
              focusPoint={focusPoint}
              onFocusChange={onFocusChange}
            />
          ) : null}

          {mode === "BLENDED" ? (
            <AnalysisMedia
              rawSrc={rawSrc}
              maskSrc={maskSrc}
              canvasRef={canvasRef}
              focusPoint={focusPoint}
              showTarget
              onFocusChange={onFocusChange}
            />
          ) : null}
        </article>
      )}
    </section>
  );
}
