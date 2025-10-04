/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './templates/**/*.html', // Scans all HTML files in the templates folder
    './static/src/**/*.js'    // Scans all JS files if you use JS to build HTML
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}

