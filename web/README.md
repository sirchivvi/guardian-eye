# 🐾 GuardianEye Web App

> **Helping protect wildlife, one signal at a time.**  
> A friendly, modern multimodal conservation web application for park rangers and wildlife researchers.

---

## 🌿 Overview

GuardianEye is an academic proof-of-concept system that combines camera-trap imagery (optical and aerial thermal) with bioacoustic sound recordings (gunshots vs. forest ambiance) to calculate an actionable risk level (**LOW**, **MEDIUM**, **HIGH**).

This web app provides:
- **Home**: Warm hero, three-step "How It Works" walkthrough, headline animated counters, and ranger field perspectives.
- **Try It (Risk Checker)**: An interactive multimodal sandbox using GuardianEye's 3-tier late-fusion rules, probability meters, and camera-trap photo preview.
- **Model Results**: Training and validation convergence curves (loss & accuracy across 30 epochs), 4-class per-class metrics tables, a soft-palette confusion matrix heatmap, dataset composition breakdown, and temperature scaling calibration metrics with a **Train / Validation / Test** split toggle.
- **About & Honesty**: Transparent scientific disclosure about synthetic image-audio pairing, sensor domain shifts, dataset limitations, and ranger-first conservation ethics.

---

## 🎨 Design Philosophy & Palette

- **Mood**: Warm, cute, welcoming nature-conservation charity (like WWF or RSPB), rather than cold tactical defense software.
- **Background**: Soft cream (`#FFF9F0`)
- **Accents**:
  - Sage Green (`#5B8266`, `#E2ECE5`) — Nature, safety, peaceful baseline
  - Warm Terracotta (`#D96B43`, `#F9E5DE`) — Acoustic triggers, warmth
  - Sunny Yellow / Amber (`#F59E0B`, `#FEF3C7`) — Sunlight, cautions
- **Risk States**:
  - `LOW`: Soft Green (`#38A169` / `#EAF7EE`) — Wildlife grazing or peaceful forest
  - `MEDIUM`: Warm Amber (`#DD6B20` / `#FFF5EA`) — Investigation recommended
  - `HIGH`: Coral Red (`#E05344` / `#FFF0EE`) — Urgent intervention alert (clear, not panic-inducing)
- **Typography**: Google Fonts **Nunito** and **Quicksand** (rounded, friendly, highly legible).
- **Animal Touches**: Custom inline SVGs for Fox, Elephant, Owl, Deer, and Sprout.

---

## 🛠️ Technology Stack

- **Framework**: [React 19](https://react.dev/) + [Vite 8](https://vite.dev/)
- **Routing**: [React Router v7](https://reactrouter.com/)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **Motion & Transitions**: [Framer Motion](https://www.framer.com/motion/)
- **Charts & Benchmarks**: [Recharts](https://recharts.org/)
- **Icons**: [Lucide React](https://lucide.dev/) + Custom SVG Animal Illustrations

---

## 🚀 Getting Started

### 1. Prerequisites
- **Node.js**: v18.0.0 or later (tested on Node v25.2.1)
- **npm**: v9.0.0 or later

### 2. Install Dependencies
```bash
cd web
npm install
```

### 3. Start Development Server
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 4. Build for Production
```bash
npm run build
```
The optimized static bundle is output to `web/dist/`.

### 5. Preview Production Build Locally
```bash
npm run preview
```

---

## 📊 Benchmark Data Loading & `/metrics.json`

The app automatically attempts to load benchmark metrics from `/metrics.json` at runtime:
```javascript
fetch('/metrics.json')
  .then(res => res.json())
  .then(data => setMetrics(data))
  .catch(() => setMetrics(fallbackMetrics));
```

If `/metrics.json` is unreachable or the app is run offline, it automatically falls back to `src/data/fallbackMetrics.js` with zero interruption to the user experience.

To export fresh evaluation outputs from training runs, place the generated JSON into `web/public/metrics.json`.

---

## ☁️ Deployment

### Deploying to Vercel

#### Option A: Deploy from Repository Root
GuardianEye includes a root `vercel.json` configured with:
```json
{
  "buildCommand": "cd web && npm install && npm run build",
  "outputDirectory": "web/dist",
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```
1. Push changes to GitHub.
2. Import repository into [Vercel](https://vercel.com).
3. Vercel will automatically build and deploy the SPA.

#### Option B: Deploy `web/` Subdirectory
1. In the Vercel project settings, set **Root Directory** to `web`.
2. The included `web/vercel.json` provides client-side SPA routing (`rewrite to /index.html`).
3. Deploy!

### Deploying to Netlify
For Netlify, add a `web/public/_redirects` file with:
```text
/*    /index.html   200
```

---

## 📁 Project Structure

```text
web/
├── public/
│   ├── favicon.svg          # Nature & wildlife paw/eye favicon
│   └── metrics.json         # Live benchmark data
├── src/
│   ├── components/
│   │   ├── Animals.jsx      # Fox, Elephant, Owl, Deer, Camera, Mic SVGs
│   │   ├── AnimatedCounter.jsx # Smooth count-up animations for stats
│   │   ├── Card.jsx         # Rounded card with gentle hover lift
│   │   ├── Footer.jsx       # Warm conservation footer
│   │   ├── LoadingSkeleton.jsx # Shimmer loading states
│   │   ├── Navbar.jsx       # Sticky glass header with mobile drawer
│   │   └── RiskBadge.jsx    # LOW, MEDIUM, HIGH risk indicators
│   ├── data/
│   │   └── fallbackMetrics.js # Complete offline fallback benchmark dataset
│   ├── pages/
│   │   ├── AboutHonesty.jsx # Academic disclosure, ethics, dataset limits
│   │   ├── Home.jsx         # Hero, 3-step walkthrough, stats, ranger quote
│   │   ├── ModelResults.jsx # Training curves, per-class table, confusion matrix
│   │   ├── NotFound.jsx     # Friendly 404 with lost owl/fox
│   │   └── TryIt.jsx        # Interactive risk playground & image preview
│   ├── App.jsx              # Routing & smooth page transitions
│   ├── index.css            # Tailwind CSS v4 & custom design tokens
│   └── main.jsx             # React entry point with BrowserRouter
├── package.json
├── vercel.json              # SPA routing rules
└── vite.config.js
```

---

## 📜 License & Citation

GuardianEye is an open research project for biodiversity protection.  
Academic evaluation only — not a certified life-safety field alarm.
