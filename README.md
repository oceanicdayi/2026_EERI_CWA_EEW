# Racing the Waves: Earthquake Early Warning in Taiwan

Interactive companion site for the CWA Seismological Center talk at the **EERI Learning From Earthquakes 2026 Travel Study Program** (Taiwan, October 18–24, 2026).

**Live site:** https://oceanicdayi.github.io/2026_EERI_CWA_EEW/

## What's inside
- **Hero seismogram**: the 2025 Dapu M6.4 earthquake seen from Tainan, with the alert issued before the P wave arrives.
- **The race**: warning time versus distance, network generation (2010 / 2016 / 2025) and alert channel.
- **Alert simulator**: click anywhere on Taiwan and watch P and S fronts, alert timing, the blind zone and county-level alerts.
- **Why Taiwan**, **How it works** (with a Pd magnitude calculator) and **25 years of getting faster** (with a blind-zone time machine).
- **Case files**: 2025 Dapu, 2024 Hualien, and the February 2026 false alert from a deep Borneo earthquake.
- **What's next**: FinDer and VS, on-site AI, an observed-intensity Transformer, a Large Earthquake Model (SeisWav2Vec 2.0), LLMs and AI agents.
- **Quiz** and **resources** (GDMS open data, Docker images, studies cited).

## Notes
The race slider and simulator are simplified teaching models (one attenuation relation, no site factors, alert time calibrated to the observed 27.0 / 12.0 / 7.9 s alerts). They are not operational CWA products.

Figures come from CWA Seismological Center presentations and the cited studies; photos are credited on the page. County boundaries: [g0v/twgeojson](https://github.com/g0v/twgeojson). Fonts: Atkinson Hyperlegible and Big Shoulders Display, under the SIL Open Font License (see `assets/fonts`).

## Structure
- `index.html`: the whole site (HTML, CSS and JavaScript, no build step)
- `assets/taiwan.js`: simplified county boundaries
- `assets/img/`: figures and photos
- `assets/fonts/`: self-hosted fonts and licenses
