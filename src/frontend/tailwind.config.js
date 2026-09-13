/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        // All three families ship Cyrillic glyphs (the default campaign language is `uk`).
        display: ['"Cormorant Garamond"', '"PT Serif"', 'Georgia', '"Times New Roman"', 'serif'],
        body: ['"Literata"', '"PT Serif"', 'Georgia', '"Times New Roman"', 'serif'],
        mono: ['"JetBrains Mono"', 'Consolas', '"Courier New"', 'monospace'],
      },
      colors: {
        parchment: '#e8dcc8',
        ink: '#1a1410',
        ember: '#c0522a',
        // 1920s accents for dice outcomes
        brass: '#c9a24a',
        verdigris: '#6f9c86',
        blood: '#9e2b25',
        ash: {
          50: '#f5f3ef',
          100: '#e8e4dc',
          200: '#cdc6b8',
          300: '#afa594',
          400: '#8f8370',
          500: '#6e6252',
          600: '#524938',
          700: '#38301f',
          800: '#231e12',
          900: '#130f08',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.4s ease-out forwards',
        blink: 'blink 1s step-end infinite',
        flicker: 'flicker 3.2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        flicker: {
          '0%, 100%': { opacity: '0.95' },
          '45%': { opacity: '0.55' },
          '55%': { opacity: '0.85' },
        },
      },
    },
  },
  plugins: [],
}
