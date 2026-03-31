"use client";

export type DisplayMode = "RAW" | "SEGMENTED" | "SPLIT" | "BLENDED";

type ControlsProps = {
  mode: DisplayMode;
  onModeChange: (mode: DisplayMode) => void;
  playing: boolean;
  onTogglePlaying: () => void;
  confidenceOverlayEnabled: boolean;
  onToggleConfidenceOverlay: () => void;
};

const MODES: Array<{ id: DisplayMode; label: string; icon: string }> = [
  { id: "RAW", label: "Raw", icon: "◧" },
  { id: "SEGMENTED", label: "Segmented", icon: "◈" },
  { id: "SPLIT", label: "Split", icon: "☷" },
  { id: "BLENDED", label: "Blended", icon: "≈" }
];

export default function Controls({
  mode,
  onModeChange,
  playing,
  onTogglePlaying,
  confidenceOverlayEnabled,
  onToggleConfidenceOverlay
}: ControlsProps) {
  return (
    <section className="control-rail">
      <div className="control-mode-row">
        {MODES.map((entry) => (
          <button
            key={entry.id}
            type="button"
            className={entry.id === mode ? "rail-mode active" : "rail-mode"}
            onClick={() => onModeChange(entry.id)}
          >
            <span className="rail-icon">{entry.icon}</span>
            <span>{entry.label}</span>
          </button>
        ))}
      </div>

      <div className="control-actions">
        <button type="button" className="ghost-action" onClick={onTogglePlaying}>
          {playing ? "Pause stream" : "Resume stream"}
        </button>

        <button type="button" className="toggle-pill" onClick={onToggleConfidenceOverlay}>
          <span>Confidence overlay</span>
          <span className={confidenceOverlayEnabled ? "toggle-indicator active" : "toggle-indicator"}>
            <i />
          </span>
        </button>
      </div>
    </section>
  );
}
