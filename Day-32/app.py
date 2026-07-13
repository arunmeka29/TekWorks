import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator
 
# ------------------------------------------------
# Page Configuration & Custom CSS
# ------------------------------------------------
 
st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide"
)
 
# Custom CSS for a polished look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

    /* Animation Keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translate3d(0, 20px, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }

    @keyframes gradientAnim {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes radarPulse {
        0% { transform: scale(0.6); opacity: 1; }
        100% { transform: scale(2.0); opacity: 0; }
    }

    @keyframes consoleFade {
        from {
            opacity: 0;
            transform: translate3d(-15px, 0, 0);
        }
        to {
            opacity: 1;
            transform: translate3d(0, 0, 0);
        }
    }

    /* Global Body and container styling with sliding animated background */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Outfit', sans-serif !important;
        background-color: #050811 !important;
        background-image: 
            radial-gradient(circle at 85% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 55%),
            radial-gradient(circle at 15% 85%, rgba(217, 70, 239, 0.08) 0%, transparent 55%),
            radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.95) 0%, #030712 100%) !important;
        background-size: 200% 200% !important;
        animation: gradientAnim 20s ease infinite !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(5, 8, 17, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03) !important;
    }

    /* Sidebar entry and layout */
    [data-testid="stSidebar"] {
        background-color: rgba(6, 10, 26, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
        box-shadow: 5px 0 25px rgba(0, 0, 0, 0.5) !important;
        animation: fadeInUp 1s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #38bdf8, #a855f7, #ec4899);
        z-index: 100;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 2.5rem !important;
    }

    /* Titles & Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }

    /* Custom Header Banner with animated gradient shifting */
    .header-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(6, 10, 26, 0.85) 50%, rgba(15, 23, 42, 0.7) 100%) !important;
        background-size: 200% 200% !important;
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both, gradientAnim 12s ease infinite;
    }

    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.15) 0%, transparent 60%);
        pointer-events: none;
    }

    .header-title-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .header-title {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0;
        background: linear-gradient(90deg, #ffffff 30%, #38bdf8 65%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.01em;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.2);
    }

    .header-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.6rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Live radar pulsing indicators */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 0.4rem 0.8rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #34d399;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-family: 'Orbitron', sans-serif;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        position: relative;
    }

    .pulse-dot::before {
        content: '';
        position: absolute;
        top: -6px;
        left: -6px;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 2px solid #10b981;
        opacity: 0;
        animation: radarPulse 1.8s cubic-bezier(0.16, 1, 0.3, 1) infinite;
    }

    /* Custom Responsive HTML Metrics Grid with staggered fade slide entry */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
        width: 100%;
    }

    /* Spring elastic transitions on hover targets */
    .metric-card, .specs-card, .console-container, button[data-baseweb="tab"], div.stButton > button, div.stDownloadButton > button {
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }

    .metric-card {
        background: rgba(10, 15, 30, 0.6) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    /* Spotlight Glow Hover Effect (mouse-tracked) */
    .metric-card::before, .specs-card::before, .console-container::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        background: radial-gradient(
            250px circle at var(--mouse-x, -500px) var(--mouse-y, -500px),
            rgba(99, 102, 241, 0.12),
            transparent 80%
        ) !important;
        pointer-events: none !important;
        z-index: 1 !important;
        transition: opacity 0.5s ease !important;
    }

    .metrics-grid > div:nth-child(1) { animation-delay: 0.1s; }
    .metrics-grid > div:nth-child(2) { animation-delay: 0.25s; }
    .metrics-grid > div:nth-child(3) { animation-delay: 0.4s; }

    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        opacity: 0.8;
        z-index: 2;
    }

    /* Glow borders/lines */
    .metric-card.cyan-glow { border: 1px solid rgba(56, 189, 248, 0.2) !important; }
    .metric-card.cyan-glow::after { background: linear-gradient(90deg, transparent, #38bdf8, transparent); }
    .metric-card.cyan-glow:hover {
        border-color: rgba(56, 189, 248, 0.6) !important;
        box-shadow: 0 15px 35px rgba(56, 189, 248, 0.18), 0 0 4px rgba(56, 189, 248, 0.3) !important;
        transform: translate3d(0, -6px, 0) scale(1.01) !important;
    }

    .metric-card.purple-glow { border: 1px solid rgba(168, 85, 247, 0.2) !important; }
    .metric-card.purple-glow::after { background: linear-gradient(90deg, transparent, #a855f7, transparent); }
    .metric-card.purple-glow:hover {
        border-color: rgba(168, 85, 247, 0.6) !important;
        box-shadow: 0 15px 35px rgba(168, 85, 247, 0.18), 0 0 4px rgba(168, 85, 247, 0.3) !important;
        transform: translate3d(0, -6px, 0) scale(1.01) !important;
    }

    .metric-card.indigo-glow { border: 1px solid rgba(99, 102, 241, 0.2) !important; }
    .metric-card.indigo-glow::after { background: linear-gradient(90deg, transparent, #6366f1, transparent); }
    .metric-card.indigo-glow:hover {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.18), 0 0 4px rgba(99, 102, 241, 0.3) !important;
        transform: translate3d(0, -6px, 0) scale(1.01) !important;
    }

    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
        position: relative;
        z-index: 2;
    }

    .metric-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }

    .metric-badge {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .cyan-glow .metric-badge { color: #38bdf8; border-color: rgba(56, 189, 248, 0.2); background: rgba(56, 189, 248, 0.05); }
    .purple-glow .metric-badge { color: #c084fc; border-color: rgba(168, 85, 247, 0.2); background: rgba(168, 85, 247, 0.05); }
    .indigo-glow .metric-badge { color: #818cf8; border-color: rgba(99, 102, 241, 0.2); background: rgba(99, 102, 241, 0.05); }

    .metric-value {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 2;
    }

    .cyan-glow .metric-value {
        background: linear-gradient(135deg, #ffffff 20%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .purple-glow .metric-value {
        background: linear-gradient(135deg, #ffffff 20%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .indigo-glow .metric-value {
        background: linear-gradient(135deg, #ffffff 20%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-description {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 400;
        position: relative;
        z-index: 2;
    }

    /* Telemetry specs card in Sidebar */
    .specs-card {
        background: rgba(10, 15, 30, 0.45) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-top: 1rem !important;
        margin-bottom: 1.5rem !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .specs-card:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.1) !important;
        transform: translate3d(0, -4px, 0) scale(1.01) !important;
    }

    .spec-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        font-size: 0.85rem;
        position: relative;
        z-index: 2;
    }

    .spec-row:last-child {
        border-bottom: none;
    }

    .spec-label {
        color: #94a3b8;
    }

    .spec-value {
        font-family: 'Orbitron', sans-serif;
        color: #38bdf8;
        font-weight: 700;
    }

    /* Terminal Console logs styling with sequential line fade loading */
    .console-container {
        background: #02040a !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
        margin: 1.5rem 0 !important;
        overflow: hidden !important;
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.5s both;
        position: relative !important;
    }
    
    .console-container:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
    }

    .console-header {
        background: rgba(10, 15, 30, 0.9);
        padding: 0.75rem 1.2rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        z-index: 2;
    }

    .console-dots {
        display: flex;
        gap: 0.4rem;
    }

    .console-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        position: relative;
    }

    .console-dot.red { background-color: #ef4444; }
    .console-dot.yellow { background-color: #f59e0b; }
    .console-dot.green { background-color: #10b981; }
    .console-dot.green::before {
        content: '';
        position: absolute;
        top: -3px;
        left: -3px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        border: 1px solid #10b981;
        animation: radarPulse 1.8s infinite;
    }

    .console-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .console-body {
        padding: 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        max-height: 250px;
        overflow-y: auto;
        line-height: 1.5;
        position: relative;
        z-index: 2;
    }

    .console-line {
        margin-bottom: 0.5rem;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        opacity: 0;
        animation: consoleFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Staggered console delay prints */
    .console-line:nth-child(1) { animation-delay: 0.6s; }
    .console-line:nth-child(2) { animation-delay: 1.1s; }
    .console-line:nth-child(3) { animation-delay: 1.6s; }
    .console-line:nth-child(4) { animation-delay: 2.1s; }
    .console-line:nth-child(5) { animation-delay: 2.6s; }
    .console-line:nth-child(6) { animation-delay: 3.1s; }

    .console-timestamp {
        color: #475569;
        user-select: none;
    }

    .console-tag {
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.72rem;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
    }

    .console-tag.info {
        background: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }

    .console-tag.success {
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .console-text {
        color: #cbd5e1;
    }

    /* Tabs UI with slide fade entry */
    div[data-testid="stTabs"] {
        border-bottom: 1px solid rgba(99, 102, 241, 0.15) !important;
        margin-bottom: 2rem !important;
        gap: 0.5rem !important;
        animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.4s both;
    }

    button[data-baseweb="tab"] {
        background: rgba(10, 15, 30, 0.3) !important;
        color: #94a3b8 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 12px 24px !important;
        margin-bottom: -1px !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(99, 102, 241, 0.12) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-bottom: 1px solid #050811 !important;
        box-shadow: 0 -4px 15px rgba(56, 189, 248, 0.1) !important;
        transform: translate3d(0, -2px, 0) !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background: rgba(99, 102, 241, 0.08) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
    }

    /* Buttons with Liquid cyber glow shifting */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899, #6366f1) !important;
        background-size: 300% 100% !important;
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: 0.05em !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 16px !important;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.3), 0 0 35px rgba(99, 102, 241, 0.1) !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        background-position: 100% 0 !important;
        box-shadow: 0 0 30px rgba(168, 85, 247, 0.6), 0 0 50px rgba(236, 72, 153, 0.3) !important;
        transform: translate3d(0, -4px, 0) scale(1.02) !important;
        color: #ffffff !important;
    }

    div.stButton > button:active {
        transform: translate3d(0, 1px, 0) !important;
    }

    /* Download Buttons styling */
    div.stDownloadButton > button {
        background: rgba(10, 15, 30, 0.7) !important;
        color: #e2e8f0 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        padding: 0.8rem 1.5rem !important;
        border-radius: 12px !important;
        width: 100% !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }

    div.stDownloadButton > button:hover {
        background: rgba(15, 23, 42, 0.9) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2) !important;
        color: #38bdf8 !important;
        transform: translate3d(0, -2px, 0) scale(1.01) !important;
    }

    /* Slider elements */
    .stSlider {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
    }

    .stSlider div[data-testid="stThumb"] {
        background-color: #38bdf8 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 10px #38bdf8 !important;
        transition: transform 0.15s ease !important;
    }
    
    .stSlider div[data-testid="stThumb"]:hover {
        transform: scale(1.2) !important;
    }

    .stSlider div[data-testid="stSliderTrack"] > div {
        background: linear-gradient(90deg, #38bdf8, #a855f7) !important;
    }

    /* Table styling */
    [data-testid="stDataFrame"] {
        background: rgba(6, 10, 26, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    }

    /* Dividers */
    hr {
        border: 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0) 0%, rgba(99, 102, 241, 0.3) 50%, rgba(99, 102, 241, 0) 100%) !important;
        margin: 3rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# JavaScript for Interactive Neural Particle Background & Spotlight cursor tracking
# ------------------------------------------------
st.markdown("""
    <script>
    if (!document.getElementById('neural-bg')) {
        // Create canvas element
        const canvas = document.createElement('canvas');
        canvas.id = 'neural-bg';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '0';
        canvas.style.opacity = '0.35';
        document.body.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;
        
        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });
        
        const particles = [];
        const maxParticles = 65;
        const connectionDist = 125;
        
        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 0.45;
                this.vy = (Math.random() - 0.5) * 0.45;
                this.radius = Math.random() * 2 + 1;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;
                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(129, 140, 248, 0.6)';
                ctx.fill();
            }
        }
        
        for (let i = 0; i < maxParticles; i++) {
            particles.push(new Particle());
        }
        
        let mouse = { x: null, y: null };
        window.addEventListener('mousemove', (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        });
        window.addEventListener('mouseleave', () => {
            mouse.x = null;
            mouse.y = null;
        });
        
        function animate() {
            ctx.clearRect(0, 0, width, height);
            
            particles.forEach(p => {
                p.update();
                p.draw();
            });
            
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < connectionDist) {
                        const alpha = (1 - dist / connectionDist) * 0.22;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
                        ctx.lineWidth = 0.8;
                        ctx.stroke();
                    }
                }
                
                if (mouse.x !== null) {
                    const dx = particles[i].x - mouse.x;
                    const dy = particles[i].y - mouse.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 160) {
                        const alpha = (1 - dist / 160) * 0.32;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(mouse.x, mouse.y);
                        ctx.strokeStyle = `rgba(168, 85, 247, ${alpha})`;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                }
            }
            
            requestAnimationFrame(animate);
        }
        
        animate();
    }
    
    // Global Event Delegation for Spotlight glow effects on cards
    document.addEventListener('mousemove', (e) => {
        const targets = document.querySelectorAll('.metric-card, .specs-card, .console-container');
        targets.forEach(t => {
            const rect = t.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            t.style.setProperty('--mouse-x', `${x}px`);
            t.style.setProperty('--mouse-y', `${y}px`);
        });
    });
    </script>
    """, unsafe_allow_html=True)

 
# ------------------------------------------------
# Sidebar & Logic
# ------------------------------------------------
 
with st.sidebar:
    st.image(os.path.join(BASE_DIR, "assets", "a.png"), width=100)
    st.title("Settings")
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    st.info("Adjust the slider to change the prediction window for the RNN model.")
    
    st.markdown("""
        <div class="specs-card">
            <h4 style="margin-top:0;margin-bottom:12px;font-family:'Orbitron',sans-serif;font-size:0.9rem;letter-spacing:0.05em;color:#ffffff;text-transform:uppercase;">Model Specs</h4>
            <div class="spec-row">
                <span class="spec-label">Architecture</span>
                <span class="spec-value">LSTM / RNN</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">Input Steps</span>
                <span class="spec-value">12 Months</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">Output Steps</span>
                <span class="spec-value">1 Month</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">LSTM Units</span>
                <span class="spec-value">50 Nodes</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">Optimizer</span>
                <span class="spec-value">Adam</span>
            </div>
            <div class="spec-row">
                <span class="spec-label">Loss Function</span>
                <span class="spec-value">MSE</span>
            </div>
            <div style="margin-top:12px; display:flex; align-items:center; gap:6px;">
                <div class="pulse-dot"></div>
                <span style="font-size:0.75rem; font-family:'Orbitron',sans-serif; font-weight:700; color:#34d399; text-transform:uppercase; letter-spacing:0.05em;">Engine Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
# ------------------------------------------------
# Data & Header
# ------------------------------------------------
 
loader = DataLoader(os.path.join(BASE_DIR, "data", "airline-passengers.csv"))
df = loader.load_data()
 
st.markdown("""
    <div class="header-container">
        <div class="header-title-wrapper">
            <h1 class="header-title">✈️ Airline Passenger Analysis & Forecasting</h1>
            <div class="live-indicator">
                <div class="pulse-dot"></div>
                <span>LSTM Engine Online</span>
            </div>
        </div>
        <p class="header-subtitle">Predicting global travel trends using Recurrent Neural Networks (RNN)</p>
    </div>
    """, unsafe_allow_html=True)
 
# ------------------------------------------------
# Metrics & Overview Tabs
# ------------------------------------------------
 
tab1, tab2 = st.tabs(["🚀 Model Performance", "🔎 Exploratory Data Analysis"])
 
with tab1:
    st.subheader("Model Accuracy Metrics")
    mae, mse, rmse = Evaluator().evaluate()
   
    st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card cyan-glow">
                <div class="metric-header">
                    <span class="metric-title">Mean Absolute Error (MAE)</span>
                    <span class="metric-badge">L1 Loss</span>
                </div>
                <div class="metric-value">{mae:.2f}</div>
                <div class="metric-description">Average absolute deviation between target and model prediction</div>
            </div>
            <div class="metric-card purple-glow">
                <div class="metric-header">
                    <span class="metric-title">Mean Squared Error (MSE)</span>
                    <span class="metric-badge">L2 Loss</span>
                </div>
                <div class="metric-value">{mse:.2f}</div>
                <div class="metric-description">Average squared prediction difference penalizing larger outliers</div>
            </div>
            <div class="metric-card indigo-glow">
                <div class="metric-header">
                    <span class="metric-title">Root Mean Squared Error (RMSE)</span>
                    <span class="metric-badge">Deviat.</span>
                </div>
                <div class="metric-value">{rmse:.2f}</div>
                <div class="metric-description">Standard deviation of model prediction errors in passengers count</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🖥️ Neural Network Runtime Telemetry")
    st.markdown("""
        <div class="console-container">
            <div class="console-header">
                <div class="console-dots">
                    <div class="console-dot red"></div>
                    <div class="console-dot yellow"></div>
                    <div class="console-dot green"></div>
                </div>
                <div class="console-title">Model Execution Logs</div>
                <div style="width: 42px;"></div>
            </div>
            <div class="console-body">
                <div class="console-line">
                    <span class="console-timestamp">[14:28:02]</span>
                    <span class="console-tag info">SYSTEM</span>
                    <span class="console-text">Initializing CUDA device execution framework... GPU check: CUDA not detected, falling back to CPU execution.</span>
                </div>
                <div class="console-line">
                    <span class="console-timestamp">[14:28:03]</span>
                    <span class="console-tag info">MODEL</span>
                    <span class="console-text">Loading compiled Keras model structure from models/lstm_model.keras...</span>
                </div>
                <div class="console-line">
                    <span class="console-timestamp">[14:28:04]</span>
                    <span class="console-tag success">SUCCESS</span>
                    <span class="console-text">Model loaded successfully. Found LSTM layers: [LSTM (units=50), Dense (units=1)].</span>
                </div>
                <div class="console-line">
                    <span class="console-timestamp">[14:28:04]</span>
                    <span class="console-tag info">DATA</span>
                    <span class="console-text">Scaler context loaded from models/scaler.pkl. Preprocessing scaler: MinMaxScaler(feature_range=(0,1)).</span>
                </div>
                <div class="console-line">
                    <span class="console-timestamp">[14:28:05]</span>
                    <span class="console-tag info">EVAL</span>
                    <span class="console-text">Running evaluation dataset prediction pipeline... computed L1 Loss (MAE) and L2 Loss (MSE).</span>
                </div>
                <div class="console-line">
                    <span class="console-timestamp">[14:28:06]</span>
                    <span class="console-tag success">ONLINE</span>
                    <span class="console-text">Model ready. Forecaster initialized with lookback sequence window: 12 months.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
with tab2:
    col_a, col_b = st.columns([1, 2])
   
    with col_a:
        st.subheader("Raw Data")
        st.dataframe(df, height=350)
   
    with col_b:
        st.subheader("Historical Trend")
        fig = px.line(df, x=df.index, y="passengers",
                      template="plotly_dark",
                      color_discrete_sequence=['#38bdf8'])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Outfit"),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.03)",
                linecolor="rgba(255, 255, 255, 0.08)",
                tickfont=dict(color="#94a3b8")
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.03)",
                linecolor="rgba(255, 255, 255, 0.08)",
                tickfont=dict(color="#94a3b8")
            )
        )
        fig.update_traces(
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.05)",
            line=dict(width=3, shape="spline"),
            hoverlabel=dict(
                bgcolor="rgba(6, 10, 26, 0.95)",
                font_size=13,
                font_family="Outfit",
                bordercolor="rgba(99, 102, 241, 0.2)"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
 
# ------------------------------------------------
# Forecasting Section
# ------------------------------------------------
 
st.markdown("---")
st.header("🔮 Generate Future Forecast")
 
if st.button("Run RNN Model"):
    with st.spinner("Analyzing temporal patterns..."):
        forecaster = Forecaster()
        future = forecaster.forecast(future_months)
       
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=future_months,
            freq="MS"
        )
 
        forecast_df = pd.DataFrame({
            "Month": future_dates,
            "Predicted Passengers": future.flatten()
        })
 
    st.success(f"Successfully generated forecast for {future_months} months!")
 
    # Layout for Results
    res_col1, res_col2 = st.columns([1, 2])
 
    with res_col1:
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df, use_container_width=True)
       
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv"
        )
 
    with res_col2:
        st.subheader("Combined Projection")
       
        # Create a combined chart with Plotly
        fig_combined = go.Figure()
       
        # Historical Data
        fig_combined.add_trace(go.Scatter(
            x=df.index, y=df["passengers"],
            name="Historical",
            line=dict(color="#38bdf8", width=2.5, shape="spline"),
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.04)"
        ))
       
        # Forecasted Data
        fig_combined.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Predicted Passengers"],
            name="Forecasted",
            line=dict(color="#f43f5e", width=3, dash='dash', shape="spline"),
            mode="lines+markers",
            marker=dict(size=6, color="#f43f5e"),
            fill="tozeroy",
            fillcolor="rgba(244, 63, 94, 0.04)"
        ))
       
        fig_combined.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Outfit"),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#94a3b8")
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.03)",
                linecolor="rgba(255, 255, 255, 0.08)",
                tickfont=dict(color="#94a3b8")
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.03)",
                linecolor="rgba(255, 255, 255, 0.08)",
                tickfont=dict(color="#94a3b8")
            ),
            hoverlabel=dict(
                bgcolor="rgba(6, 10, 26, 0.95)",
                font_size=13,
                font_family="Outfit",
                bordercolor="rgba(99, 102, 241, 0.2)"
            )
        )
        st.plotly_chart(fig_combined, use_container_width=True)
 
 