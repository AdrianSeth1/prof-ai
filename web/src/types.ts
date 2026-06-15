export type Screen = 'chat' | 'live' | 'materials' | 'modules' | 'gaps';
export type Mode = 'lecture' | 'awaiting' | 'processing' | 'speaking';

export interface Toast {
  id: string;
  title: string;
  desc?: string;
}

export type HealthState = 'up' | 'degraded' | 'down';

export interface HealthEntry {
  status: HealthState;
  info: string;
}

export interface HealthData {
  backend: HealthEntry;
  ollama: HealthEntry;
  gpu: HealthEntry;
}

export interface PaletteCommand {
  label: string;
  hint?: string;
  group: 'Navigation' | 'Lecture' | 'Actions';
  run: () => void;
}
