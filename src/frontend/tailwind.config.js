/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Cinzel"', 'serif'],
        body: ['"Crimson Pro"', 'serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        parchment: '#e8dcc8',
        ink: '#1a1410',
        ember: '#c0522a',
        ash: {
          50:  '#f5f3ef',
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
        'blink': 'blink 1s step-end infinite',
      },
      keyframes: {
        fadeIn: {
          '0%':   { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0' },
        },
      },
    },
  },
  plugins: [],
}
