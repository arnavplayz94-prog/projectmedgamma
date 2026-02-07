/**
 * MedGemma Login Page - Interactive Features
 * Handles animations, demo login, and navigation
 */

// =====================================================
// Configuration
// =====================================================
const CONFIG = {
    // Healthcare-themed icons for floating animation
    healthcareIcons: ['💊', '🩺', '💉', '❤️', '📋', '🏥', '🩹', '💗', '🧬', '⚕️'],

    // Number of floating icons
    floatingIconCount: 15,

    // Number of particles
    particleCount: 25,

    // Animation duration ranges (in seconds)
    iconDurationMin: 15,
    iconDurationMax: 30,
    particleDurationMin: 8,
    particleDurationMax: 15
};

// =====================================================
// Initialization
// =====================================================
document.addEventListener('DOMContentLoaded', () => {
    initializeFloatingIcons();
    initializeParticles();
    initializeAuthButtons();
});

// =====================================================
// Floating Healthcare Icons
// =====================================================
function initializeFloatingIcons() {
    const container = document.getElementById('animatedBg');
    if (!container) return;

    for (let i = 0; i < CONFIG.floatingIconCount; i++) {
        createFloatingIcon(container, i);
    }
}

function createFloatingIcon(container, index) {
    const icon = document.createElement('div');
    icon.className = 'floating-icon';

    // Random icon selection
    const randomIcon = CONFIG.healthcareIcons[Math.floor(Math.random() * CONFIG.healthcareIcons.length)];
    icon.textContent = randomIcon;

    // Random horizontal position
    const leftPosition = Math.random() * 100;
    icon.style.left = `${leftPosition}%`;

    // Random starting vertical position (staggered for natural effect)
    const startOffset = Math.random() * 100;
    icon.style.bottom = `-${startOffset}px`;

    // Random size variation
    const size = 1.5 + Math.random() * 2;
    icon.style.fontSize = `${size}rem`;

    // Random animation duration
    const duration = CONFIG.iconDurationMin + Math.random() * (CONFIG.iconDurationMax - CONFIG.iconDurationMin);
    icon.style.animationDuration = `${duration}s`;

    // Random animation delay for staggered start
    const delay = Math.random() * duration;
    icon.style.animationDelay = `-${delay}s`;

    // Some icons get glow effect
    if (Math.random() > 0.6) {
        icon.classList.add('glow');
    }

    container.appendChild(icon);
}

// =====================================================
// Particle Effects
// =====================================================
function initializeParticles() {
    const container = document.getElementById('particlesLayer');
    if (!container) return;

    for (let i = 0; i < CONFIG.particleCount; i++) {
        createParticle(container, i);
    }
}

function createParticle(container, index) {
    const particle = document.createElement('div');
    particle.className = 'particle';

    // Random horizontal position
    particle.style.left = `${Math.random() * 100}%`;

    // Random size
    const size = 2 + Math.random() * 4;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;

    // Random color variation (shades of blue)
    const hue = 200 + Math.random() * 40; // Blue range
    const saturation = 70 + Math.random() * 30;
    const lightness = 50 + Math.random() * 20;
    particle.style.background = `hsla(${hue}, ${saturation}%, ${lightness}%, 0.6)`;

    // Random animation duration
    const duration = CONFIG.particleDurationMin + Math.random() * (CONFIG.particleDurationMax - CONFIG.particleDurationMin);
    particle.style.animationDuration = `${duration}s`;

    // Random animation delay
    const delay = Math.random() * duration;
    particle.style.animationDelay = `-${delay}s`;

    container.appendChild(particle);
}

// =====================================================
// Authentication Buttons
// =====================================================
function initializeAuthButtons() {
    const googleBtn = document.getElementById('googleSignInBtn');
    const demoBtn = document.getElementById('demoUserBtn');

    googleBtn?.addEventListener('click', handleGoogleSignIn);
    demoBtn?.addEventListener('click', handleDemoLogin);
}

function handleGoogleSignIn() {
    const btn = document.getElementById('googleSignInBtn');

    // Add loading state
    btn.classList.add('loading');

    // Simulate OAuth redirect (demo only)
    setTimeout(() => {
        console.log('Demo: Google OAuth login initiated');
        navigateToDashboard();
    }, 1500);
}

function handleDemoLogin() {
    const btn = document.getElementById('demoUserBtn');

    // Add loading state
    btn.classList.add('loading');

    // Simulate login process
    setTimeout(() => {
        console.log('Demo: Demo user login successful');
        navigateToDashboard();
    }, 1000);
}

function navigateToDashboard() {
    // Navigate to the main dashboard
    window.location.href = 'index.html';
}

// =====================================================
// Optional: Add subtle mouse parallax effect
// =====================================================
document.addEventListener('mousemove', (e) => {
    const container = document.getElementById('animatedBg');
    if (!container || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const x = (e.clientX / window.innerWidth - 0.5) * 10;
    const y = (e.clientY / window.innerHeight - 0.5) * 10;

    container.style.transform = `translate(${x}px, ${y}px)`;
});

// =====================================================
// Keyboard Accessibility
// =====================================================
document.addEventListener('keydown', (e) => {
    // Allow Enter key on buttons
    if (e.key === 'Enter' && e.target.classList.contains('auth-btn')) {
        e.target.click();
    }
});
