import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        base: '#08090a',
        sidebar: '#0a0b0d',
        surface: '#0c0d10',
        surface2: '#0e0f12',
        pop: '#141517',
        toast: '#16171a',
        ink: {
          DEFAULT: '#ededed',
          body: '#dfe1e6',
          soft: '#c4c8cf',
          muted: '#8a8f98',
          faint: '#5a5e66',
          ghost: '#3a3d44',
        },
        accent: {
          DEFAULT: '#5e6ad2',
          hover: '#6e79e0',
          deep: '#5059bd',
          light: '#8b95ea',
          text: '#c5cbff',
        },
        rec: '#ef4d56',
        await: '#e9a23b',
        ok: '#3fb950',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      borderRadius: {
        xs: '4px',
        sm: '6px',
        md: '8px',
        lg: '11px',
        xl: '13px',
      },
      transitionDuration: {
        DEFAULT: '140ms',
      },
    },
  },
  plugins: [],
} satisfies Config
