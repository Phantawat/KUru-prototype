import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
    "./src/lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "ku-green": "#006633",
        "ku-green-dark": "#004D26",
        "ku-gold": "#C5B358",
        "ku-mint": "#F0F9F4",
        "ku-mint-border": "#9FE1CB",
      },
    },
  },
};

export default config;
