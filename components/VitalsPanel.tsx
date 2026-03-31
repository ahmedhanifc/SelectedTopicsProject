"use client";

import { useEffect, useState } from "react";

type VitalsState = {
  heartRate: number;
  spo2: number;
  bloodPressure: [number, number];
};

const INITIALS: VitalsState = {
  heartRate: 72,
  spo2: 98,
  bloodPressure: [118, 79]
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export default function VitalsPanel() {
  const [vitals, setVitals] = useState<VitalsState>(INITIALS);

  useEffect(() => {
    const interval = window.setInterval(() => {
      setVitals((current) => ({
        heartRate: clamp(current.heartRate + (Math.random() - 0.5) * 2.2, 69, 76),
        spo2: clamp(current.spo2 + (Math.random() - 0.5) * 1, 97, 100),
        bloodPressure: [
          clamp(current.bloodPressure[0] + (Math.random() - 0.5) * 2, 116, 121),
          clamp(current.bloodPressure[1] + (Math.random() - 0.5) * 2, 77, 81)
        ]
      }));
    }, 1500);

    return () => window.clearInterval(interval);
  }, []);

  return (
    <article className="info-card vitals-card">
      <span className="mini-heading">Patient vitals</span>
      <div className="vitals-stack">
        <div className="vital-line">
          <label>Heart rate</label>
          <strong>
            {Math.round(vitals.heartRate)}
            <small>bpm</small>
          </strong>
        </div>
        <div className="vital-line">
          <label>SpO2</label>
          <strong>
            {Math.round(vitals.spo2)}
            <small>%</small>
          </strong>
        </div>
        <div className="vital-line">
          <label>Blood pressure</label>
          <strong className="tight">
            {Math.round(vitals.bloodPressure[0])}/{Math.round(vitals.bloodPressure[1])}
          </strong>
        </div>
      </div>
    </article>
  );
}
