import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "hsl(var(--bg) / <alpha-value>)",
        surface: "hsl(var(--surface) / <alpha-value>)",
        "surface-2": "hsl(var(--surface-2) / <alpha-value>)",
        "surface-3": "hsl(var(--surface-3) / <alpha-value>)",
        text: "hsl(var(--text) / <alpha-value>)",
        "text-secondary": "hsl(var(--text-secondary) / <alpha-value>)",
        "text-tertiary": "hsl(var(--text-tertiary) / <alpha-value>)",
        border: "hsl(var(--border) / <alpha-value>)",
        "border-subtle": "hsl(var(--border-subtle) / <alpha-value>)",
        primary: "hsl(var(--primary) / <alpha-value>)",
        "primary-light": "hsl(var(--primary-light) / <alpha-value>)",
        "primary-muted": "hsl(var(--primary-muted) / <alpha-value>)",
        accent: "hsl(var(--accent) / <alpha-value>)",
        "accent-light": "hsl(var(--accent-light) / <alpha-value>)",
        success: "hsl(var(--success) / <alpha-value>)",
        "success-muted": "hsl(var(--success-muted) / <alpha-value>)",
        warning: "hsl(var(--warning) / <alpha-value>)",
        "warning-muted": "hsl(var(--warning-muted) / <alpha-value>)",
        danger: "hsl(var(--danger) / <alpha-value>)",
        "danger-muted": "hsl(var(--danger-muted) / <alpha-value>)",
        info: "hsl(var(--info) / <alpha-value>)",
        "info-muted": "hsl(var(--info-muted) / <alpha-value>)"
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        heading: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "Monaco", "monospace"]
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "1rem", letterSpacing: "0.02em" }]
      },
      boxShadow: {
        "soft-lg": "0 12px 40px -12px rgba(0, 0, 0, 0.08)",
        "soft-md": "0 8px 24px -8px rgba(0, 0, 0, 0.06)",
        "soft-sm": "0 2px 8px -2px rgba(0, 0, 0, 0.05)",
        "inner-soft": "inset 0 2px 4px rgba(0, 0, 0, 0.02)"
      },
      borderRadius: {
        "4xl": "2rem"
      },
      transitionTimingFunction: {
        smooth: "cubic-bezier(0.4, 0, 0.2, 1)"
      },
      animation: {
        "fade-in": "fadeIn 0.2s ease-out",
        "slide-up": "slideUp 0.2s ease-out",
        "scale-in": "scaleIn 0.15s ease-out"
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" }
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        scaleIn: {
          "0%": { opacity: "0", transform: "scale(0.98)" },
          "100%": { opacity: "1", transform: "scale(1)" }
        }
      }
    }
  },
  plugins: []
} satisfies Config;