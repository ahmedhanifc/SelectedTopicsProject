"use client";

import { useEffect, useRef } from "react";

export type FocusPoint = {
  x: number;
  y: number;
};

type FocusZoneProps = {
  src: string | null;
  point: FocusPoint;
  zoom?: number;
  onFocusChange: (point: FocusPoint) => void;
};

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export default function FocusZone({ src, point, zoom = 2.5, onFocusChange }: FocusZoneProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragActiveRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !src) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const image = new Image();

    image.onload = () => {
      const width = canvas.clientWidth || 320;
      const height = canvas.clientHeight || 120;
      const pixelRatio = window.devicePixelRatio || 1;

      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, width, height);

      const cropWidth = image.width / zoom;
      const cropHeight = image.height / zoom;
      const centerX = point.x * image.width;
      const centerY = point.y * image.height;

      const maxStartX = Math.max(image.width - cropWidth, 0);
      const maxStartY = Math.max(image.height - cropHeight, 0);
      const sourceX = Math.min(Math.max(centerX - cropWidth / 2, 0), maxStartX);
      const sourceY = Math.min(Math.max(centerY - cropHeight / 2, 0), maxStartY);

      context.drawImage(image, sourceX, sourceY, cropWidth, cropHeight, 0, 0, width, height);

      context.strokeStyle = "rgba(255, 255, 255, 0.7)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(width / 2, 0);
      context.lineTo(width / 2, height);
      context.moveTo(0, height / 2);
      context.lineTo(width, height / 2);
      context.stroke();
    };

    image.src = src;
  }, [point, src, zoom]);

  const updateFromPointer = (clientX: number, clientY: number, element: HTMLElement) => {
    const rect = element.getBoundingClientRect();
    const x = clamp01((clientX - rect.left) / rect.width);
    const y = clamp01((clientY - rect.top) / rect.height);
    onFocusChange({ x, y });
  };

  return (
    <div
      className="focus-preview-shell"
      onPointerDown={(event) => {
        dragActiveRef.current = true;
        event.currentTarget.setPointerCapture(event.pointerId);
        updateFromPointer(event.clientX, event.clientY, event.currentTarget);
      }}
      onPointerMove={(event) => {
        if (!dragActiveRef.current) {
          return;
        }
        updateFromPointer(event.clientX, event.clientY, event.currentTarget);
      }}
      onPointerUp={(event) => {
        dragActiveRef.current = false;
        event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={() => {
        dragActiveRef.current = false;
      }}
    >
      <canvas ref={canvasRef} className="focus-preview-canvas" aria-label="AI focus zoom preview" />
    </div>
  );
}
