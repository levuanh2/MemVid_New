/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/typography'),  // Add typography for prose classes (bold, tables, etc.)
  ],
}

export default {
  content: [],
  theme: {
    extend: {},
  },
  plugins: [],
}

