# EmotiScan v2.0
Dataset-Driven Mental Health Screening

**Healthcare AI Hackathon — "Agents Assemble: The Healthcare AI Endgame Challenge"**

EmotiScan v2.0 is a production-ready AI pipeline for EEG-based mental health screening. We built the AI and visualization layer that sits between any EEG device and the healthcare system. Today we demo with datasets; tomorrow, this connects to the 30M+ consumer EEG devices already in homes.

## 🚀 Features

*   **No Hardware Required**: Built upon public datasets (SEED, DEAP, DREAMER).
*   **Synthetic EEG Generation**: Powered by a Wasserstein GAN + Gradient Penalty (WGAN-GP) to simulate an infinite amount of edge case demo scenarios and protect patient privacy.
*   **3D Brain Visualization**: A stunning interactive visualization built with React & Three.js that demonstrates hemispheric activation and frequency bands in real-time.
*   **A2A Agent Network**: Leveraging an intelligent multi-agent network (Data Curator, Synthetic Generator, Analysis Engine, Visualization) orchestrated via the Model Context Protocol (MCP).
*   **Clinical Integration**: FHIR/SHARP compliant readiness to natively integrate with modern electronic healthcare records.

## 💻 Tech Stack

*   **Frontend**: React 18, TypeScript, Three.js, `@react-three/fiber`, Tailwind CSS, Recharts
*   **Backend**: Python 3.11, FastAPI, PyTorch (Models + WGAN-GP), LIME (Explainability), MNE-Python (EEG Processing)
*   **Agent Protocols**: Anthropic MCP SDK, Google Agent Development Kit (ADK)

## 🏁 Getting Started

### Prerequisites

*   Node.js v18+ & npm
*   Python 3.11+
*   Kaggle CLI (to fetch the initial dataset)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shivangikamat/health-care-ai-hack.git
   cd health-care-ai-hack
   ```

2. **Download the Initial Dataset:**
   ```bash
   pip install kaggle
   kaggle datasets download -d birdy654/eeg-brainwave-dataset-feeling-emotions
   unzip eeg-brainwave-dataset-feeling-emotions.zip -d data/eeg_brainwave/
   ```

3. **Backend Setup:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install fastapi "uvicorn[standard]" torch scipy numpy mne mcp
   # Start the FastAPI server
   ```

4. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   # Start the React App
   npm run dev
   ```

## 🧠 The Science

Our classification models utilize peer-reviewed 4D-CRNN and bi-hemispheric attention methodologies, analyzing:
*   **Differential Entropy (DE)** across Theta, Alpha, Beta, & Gamma frequency bands
*   **Frontal Alpha Asymmetry** (a key clinically validated biomarker for depression)
*   **Hemispheric Imbalances** and complex temporal dynamics
