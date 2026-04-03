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
  showConfidenceToggle?: boolean;
};

const MODES: Array<{ id: DisplayMode; label: string; icon: string }> = [
  { id: "RAW", label: "Raw", icon: "\u25C9" },
  { id: "SEGMENTED", label: "Segmented", icon: "\u25D4" },
  { id: "SPLIT", label: "Split", icon: "\u25EB" },
  { id: "BLENDED", label: "Blended", icon: "\u25CD" }
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
  showActions = true,
  showConfidenceToggle = true
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
              <span
                className={
                  `rail-icon ${
                    entry.id === "SEGMENTED" || entry.id === "SPLIT" || entry.id === "BLENDED"
                      ? "detail-icon"
                    : ""
                  } ${entry.id === "SEGMENTED" ? "segmented-icon" : ""} ${
                    entry.id === "RAW"
                      ? "raw-icon"
                      : entry.id === "SEGMENTED"
                        ? "segmented-mode-icon"
                        : entry.id === "SPLIT"
                          ? "split-icon"
                          : entry.id === "BLENDED"
                            ? "blended-icon"
                            : ""
                  }`.trim()
                }
              >
                {entry.icon}
              </span>
              {!collapsed ? <span>{entry.label}</span> : null}
            </button>
          ))}
        </div>
      ) : null}

      {showActions ? (
        <div className="control-actions">
          <button
            type="button"
            className={playing ? "ghost-action stream-action" : "ghost-action stream-action paused"}
            onClick={onTogglePlaying}
            aria-label={playing ? "Pause stream" : "Resume stream"}
            aria-pressed={!playing}
          >
            <span
              className={playing ? "pause-glyph" : "pause-glyph paused"}
              aria-hidden="true"
            >
              <i />
              <i />
            </span>
            {!collapsed ? <span>{playing ? "Pause stream" : "Resume stream"}</span> : null}
          </button>

          {showConfidenceToggle ? (
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
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
