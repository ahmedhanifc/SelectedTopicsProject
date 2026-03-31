"use client";

export type DisplayMode = "RAW" | "SEGMENTED" | "SPLIT" | "BLENDED";

type ControlsProps = {
  mode: DisplayMode;
  onModeChange: (mode: DisplayMode) => void;
  playing: boolean;
  onTogglePlaying: () => void;
  confidenceOverlayEnabled: boolean;
  onToggleConfidenceOverlay: () => void;
  collapsed?: boolean;
  showModes?: boolean;
  showActions?: boolean;
};

const MODES: Array<{ id: DisplayMode; label: string; icon: string }> = [
  { id: "RAW", label: "Raw", icon: "\u25A1" },
  { id: "SEGMENTED", label: "Segmented", icon: "\u25C8" },
  { id: "SPLIT", label: "Split", icon: "\u2637" },
  { id: "BLENDED", label: "Blended", icon: "\u2248" }
];

export default function Controls({
  mode,
  onModeChange,
  playing,
  onTogglePlaying,
  confidenceOverlayEnabled,
  onToggleConfidenceOverlay,
  collapsed = false,
  showModes = true,
  showActions = true
}: ControlsProps) {
  return (
    <section className="control-rail">
      {showModes ? (
        <div className="control-mode-row">
          {MODES.map((entry) => (
            <button
              key={entry.id}
              type="button"
              className={entry.id === mode ? "rail-mode active" : "rail-mode"}
              onClick={() => onModeChange(entry.id)}
              aria-label={entry.label}
            >
              <span className="rail-icon">{entry.icon}</span>
              {!collapsed ? <span>{entry.label}</span> : null}
            </button>
          ))}
        </div>
      ) : null}

      {showActions ? (
        <div className="control-actions">
          <button
            type="button"
            className="ghost-action"
            onClick={onTogglePlaying}
            aria-label={playing ? "Pause stream" : "Resume stream"}
          >
            <span className="action-icon" aria-hidden="true">
              {playing ? "\u275A\u275A" : "\u25B6"}
            </span>
            {!collapsed ? <span>{playing ? "Pause stream" : "Resume stream"}</span> : null}
          </button>

          <button
            type="button"
            className="toggle-pill"
            onClick={onToggleConfidenceOverlay}
            aria-label="Confidence overlay"
          >
            <span
              className={confidenceOverlayEnabled ? "overlay-icon active" : "overlay-icon"}
              aria-hidden="true"
            >
              <i className="overlay-back" />
              <i className="overlay-front" />
            </span>
            {!collapsed ? <span>Confidence overlay</span> : null}
          </button>
        </div>
      ) : null}
    </section>
  );
}
