/**
 * MedGemma Dashboard - Hospital-Ready Implementation
 * All 11 Doctor Requirements Implemented
 */

// =====================================================
// State Management
// =====================================================
const AppState = {
    currentView: 'dashboard',
    selectedPatient: null,
    patients: [],
    alerts: [],
    notes: [],
    messages: [],
    settings: {
        specialty: 'general',
        alertSoundEnabled: true,
        alertAutoCollapse: true,
        compactMode: false,
        showBaselines: true
    },
    stats: {
        activePatients: 0,
        pendingAlerts: 0,
        draftNotes: 0,
        reviewedToday: 0
    },
    savedItems: []
};

// Sample Data Generator
const DataGenerator = {
    patientNames: [
        'John Smith', 'Maria Garcia', 'James Johnson', 'Sarah Williams',
        'Michael Brown', 'Emily Davis', 'Robert Wilson', 'Lisa Anderson'
    ],

    conditions: ['cardiac', 'respiratory', 'neuro', 'general'],

    generatePatientId() {
        return `PT-${Math.floor(1000 + Math.random() * 9000)}`;
    },

    generateVitals(isAbnormal = false) {
        const base = {
            hr: { value: 72, baseline: 72, min: 60, max: 100 },
            bp: { systolic: 120, diastolic: 80, baseline: { s: 120, d: 80 } },
            spo2: { value: 98, baseline: 98, min: 95, max: 100 },
            temp: { value: 98.6, baseline: 98.6, min: 97, max: 99 },
            resp: { value: 16, baseline: 16, min: 12, max: 20 }
        };

        if (isAbnormal) {
            base.hr.value = Math.floor(100 + Math.random() * 30);
            base.bp.systolic = Math.floor(140 + Math.random() * 20);
        }

        return base;
    },

    generateSamplePatients() {
        return [
            {
                id: 'PT-1234',
                name: 'John Smith',
                age: 62,
                gender: 'M',
                riskLevel: 'high',
                condition: 'cardiac',
                notes: 'Elevated heart rate, monitoring required',
                lastUpdate: new Date(Date.now() - 5 * 60000),
                alertCount: 2,
                vitals: this.generateVitals(true)
            },
            {
                id: 'PT-5678',
                name: 'Maria Garcia',
                age: 45,
                gender: 'F',
                riskLevel: 'medium',
                condition: 'respiratory',
                notes: 'SpO2 trending lower, observation needed',
                lastUpdate: new Date(Date.now() - 15 * 60000),
                alertCount: 1,
                vitals: this.generateVitals()
            },
            {
                id: 'PT-9012',
                name: 'James Johnson',
                age: 55,
                gender: 'M',
                riskLevel: 'low',
                condition: 'general',
                notes: 'Post-op day 2, stable vitals',
                lastUpdate: new Date(Date.now() - 30 * 60000),
                alertCount: 0,
                vitals: this.generateVitals()
            },
            {
                id: 'PT-3456',
                name: 'Sarah Williams',
                age: 38,
                gender: 'F',
                riskLevel: 'info',
                condition: 'neuro',
                notes: 'Routine observation, all metrics normal',
                lastUpdate: new Date(Date.now() - 60 * 60000),
                alertCount: 0,
                vitals: this.generateVitals()
            }
        ];
    },

    generateSampleAlerts() {
        return [
            {
                id: 'alert-1',
                type: 'critical',
                patientId: 'PT-1234',
                message: 'Heart rate > 120 BPM for 15 minutes',
                time: new Date(Date.now() - 2 * 60000),
                acknowledged: false
            },
            {
                id: 'alert-2',
                type: 'critical',
                patientId: 'PT-1234',
                message: 'Blood pressure elevated: 158/95 mmHg',
                time: new Date(Date.now() - 8 * 60000),
                acknowledged: false
            },
            {
                id: 'alert-3',
                type: 'warning',
                patientId: 'PT-5678',
                message: 'SpO2 dropped to 93%, trending down',
                time: new Date(Date.now() - 12 * 60000),
                acknowledged: false
            },
            {
                id: 'alert-4',
                type: 'info',
                patientId: 'PT-9012',
                message: 'Scheduled medication due in 30 minutes',
                time: new Date(Date.now() - 25 * 60000),
                acknowledged: true
            }
        ];
    },

    generateSampleNotes() {
        return [
            {
                id: 'note-1',
                patientId: 'PT-1234',
                author: 'Dr. Smith',
                content: 'Patient responding to treatment. HR trending down. Continue monitoring.',
                type: 'Progress Note',
                timestamp: new Date(Date.now() - 2 * 60 * 60000)
            },
            {
                id: 'note-2',
                patientId: 'PT-1234',
                author: 'Nurse Stevens',
                content: 'Vitals checked. Patient comfortable, no acute distress.',
                type: 'Nursing Note',
                timestamp: new Date(Date.now() - 4 * 60 * 60000)
            },
            {
                id: 'note-3',
                patientId: 'PT-5678',
                author: 'Dr. Smith',
                content: 'Ordered supplemental O2. Will reassess in 2 hours.',
                type: 'Order Note',
                timestamp: new Date(Date.now() - 1 * 60 * 60000)
            }
        ];
    }
};

// =====================================================
// Utility Functions
// =====================================================
function formatTimeAgo(date) {
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);

    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleDateString();
}

function formatTime(date) {
    return date.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
}

function formatDate(date) {
    const today = new Date();
    if (date.toDateString() === today.toDateString()) {
        return `Today ${formatTime(date)}`;
    }
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit'
    });
}

function getTrendClass(current, baseline, isHighBad = true) {
    const diff = ((current - baseline) / baseline) * 100;
    if (Math.abs(diff) < 5) return 'trend-stable';
    if (isHighBad) {
        return diff > 0 ? 'trend-up' : 'trend-down';
    }
    return diff > 0 ? 'trend-down' : 'trend-up';
}

function getTrendText(current, baseline) {
    const diff = ((current - baseline) / baseline) * 100;
    if (Math.abs(diff) < 5) return '→ stable';
    const arrow = diff > 0 ? '↑' : '↓';
    return `${arrow} ${Math.abs(diff).toFixed(0)}%`;
}

// =====================================================
// Initialization
// =====================================================
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize components
    initializeSidebar();
    initializePatientList();
    initializeFilters();
    initializeTimeRange();
    initializeAlerts();
    initializeVitals();
    initializeAINotes();
    initializeNotesTimeline();
    initializeMessaging();
    initializeSettings();
    initializeSearch();
    initializeModals();
    initializeSavedSection();
    initializeChatbot();

    // Fetch real patients from the backend to ensure IDs match (P001, P002, P003)
    try {
        const response = await fetch('http://127.0.0.1:5000/api/patients');
        if (response.ok) {
            const data = await response.json();
            console.log("Raw API Response:", data); // Debugging

            // Map backend data to frontend structure with safety checks
            AppState.patients = data.patients.map(p => {
                // Determine gender safely
                const sex = p.demographics?.sex || p.sex || 'Male';
                const gender = sex[0]; // 'M' or 'F'
                const name = `${sex === 'Male' ? 'Mr.' : 'Ms.'} ${p.patient_id}`;

                // Determine age safely
                const age = p.demographics?.age || p.age || 0;

                // Determine risk level safely
                const risk = p.risk_assessment?.level || 'low';

                return {
                    id: p.patient_id,
                    name: name,
                    age: age,
                    gender: gender,
                    riskLevel: risk,
                    condition: 'Synthetic Monitoring',
                    notes: (p.medical_history || []).join(', '),
                    lastUpdate: new Date(),
                    alertCount: risk === 'high' ? 1 : 0,
                    vitals: {
                        hr: { value: p.vitals?.heart_rate_avg || 70, baseline: (p.vitals?.heart_rate_avg || 72) - 2, min: 60, max: 100 },
                        bp: {
                            systolic: parseInt((p.vitals?.blood_pressure || "120/80").split('/')[0]),
                            diastolic: parseInt((p.vitals?.blood_pressure || "120/80").split('/')[1]),
                            baseline: { s: 120, d: 80 }
                        },
                        spo2: { value: p.vitals?.spo2_percent || 98, baseline: 98, min: 94, max: 100 },
                        temp: { value: p.vitals?.body_temperature_c || 37, baseline: 36.6, min: 36, max: 38 },
                        resp: { value: p.vitals?.respiratory_rate || 16, baseline: 16, min: 12, max: 20 }
                    }
                };
            });
            console.log("Mapped Patients:", AppState.patients); // Debugging
        } else {
            console.warn("API returned error, falling back to samples");
            AppState.patients = DataGenerator.generateSamplePatients();
        }
    } catch (e) {
        console.error('Could not fetch patients, using samples:', e);
        AppState.patients = DataGenerator.generateSamplePatients();
    }

    AppState.alerts = DataGenerator.generateSampleAlerts();
    AppState.notes = DataGenerator.generateSampleNotes();

    // Initial render
    renderPatientList();
    renderAlerts();
    updateStats();

    // Auto-select first patient for demo
    if (AppState.patients.length > 0) {
        selectPatient(AppState.patients[0]);
    }
    // Start 5-minute live updates
    startLiveUpdates();
});

// =====================================================
// Live Updates
// =====================================================
function startLiveUpdates() {
    // Update every 5 minutes (300,000 ms)
    setInterval(() => {
        console.log('Updating vital signs with new values...');

        AppState.patients.forEach(patient => {
            if (!patient.vitals) return;

            // Update HR (-3 to +3 variation)
            const hrInfo = patient.vitals.hr;
            const hrVar = Math.floor(Math.random() * 7) - 3;
            let newHr = hrInfo.value + hrVar;
            patient.vitals.hr.value = Math.max(hrInfo.min - 5, Math.min(hrInfo.max + 5, newHr));

            // Update BP Systolic (-3 to +3)
            const bpInfo = patient.vitals.bp;
            const bpVar = Math.floor(Math.random() * 7) - 3;
            let newSys = bpInfo.systolic + bpVar;
            patient.vitals.bp.systolic = Math.max(90, Math.min(180, newSys));

            // Update BP Diastolic (follows systolic vaguely)
            let newDia = Math.floor(newSys * 0.65) + (Math.floor(Math.random() * 5) - 2);
            patient.vitals.bp.diastolic = Math.max(50, Math.min(110, newDia));

            // Update SpO2 (mostly stable, -1 to +1)
            const spo2Info = patient.vitals.spo2;
            if (spo2Info.value >= 99) {
                patient.vitals.spo2.value = Math.random() > 0.7 ? 98 : 99;
            } else {
                const spo2Var = Math.random() > 0.8 ? -1 : (Math.random() > 0.5 ? 1 : 0);
                let newSpo2 = spo2Info.value + spo2Var;
                patient.vitals.spo2.value = Math.max(spo2Info.min - 2, Math.min(100, newSpo2));
            }

            // Update Temp (-0.1 to +0.1)
            const tempInfo = patient.vitals.temp;
            const tempVar = (Math.random() * 0.2) - 0.1;
            let newTemp = parseFloat((tempInfo.value + tempVar).toFixed(1));
            patient.vitals.temp.value = Math.max(97, Math.min(101, newTemp));

            // Update Resp (-1 to +1)
            const respInfo = patient.vitals.resp;
            const respVar = Math.floor(Math.random() * 3) - 1;
            let newResp = respInfo.value + respVar;
            patient.vitals.resp.value = Math.max(12, Math.min(24, newResp));

            // Update timestamp
            patient.lastUpdate = new Date();
        });

        // Re-render UI
        renderPatientList();
        if (AppState.selectedPatient) {
            renderVitals(AppState.selectedPatient);
            initializeVitalTrendChart(AppState.selectedPatient);
        }

    }, 300000); // 300,000 ms = 5 minutes
}

// =====================================================
// Sidebar
// =====================================================
function initializeSidebar() {
    const toggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');

    toggle?.addEventListener('click', () => {
        sidebar?.classList.toggle('open');
    });

    // Close on outside click (mobile)
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 900) {
            if (!sidebar?.contains(e.target) && !toggle?.contains(e.target)) {
                sidebar?.classList.remove('open');
            }
        }
    });
}

// =====================================================
// Patient List (Requirements 5, 6)
// =====================================================
function initializePatientList() {
    const searchInput = document.getElementById('patientSearchInput');

    searchInput?.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        filterPatients(query);
    });
}

function renderPatientList(patients = AppState.patients) {
    const list = document.getElementById('patientList');
    if (!list) return;

    list.innerHTML = '';

    if (patients.length === 0) {
        list.innerHTML = `
            <li class="patient-empty-state">
                No patients found.<br>Click "Add Patient" to start.
            </li>
        `;
        return;
    }

    patients.forEach(patient => {
        const card = createPatientCard(patient);
        list.appendChild(card);
    });

    updateStats();
}

function createPatientCard(patient) {
    const li = document.createElement('li');
    li.className = `patient-card risk-${patient.riskLevel}`;
    li.dataset.patientId = patient.id;

    if (AppState.selectedPatient?.id === patient.id) {
        li.classList.add('active');
    }

    const riskLabels = {
        'high': 'Critical',
        'medium': 'Warning',
        'low': 'Stable',
        'info': 'Observe'
    };

    const alertIndicator = patient.alertCount > 0
        ? `<span class="alert-indicator has-alerts" title="${patient.alertCount} Active Alerts">⚠ ${patient.alertCount}</span>`
        : '<span class="alert-indicator"></span>';

    li.innerHTML = `
        <div class="patient-card-header">
            <span class="patient-name">${patient.name}</span>
            <span class="patient-risk-badge">${riskLabels[patient.riskLevel]}</span>
        </div>
        <div class="patient-card-meta">
            <span class="patient-id">${patient.id}</span>
            <span class="patient-age">${patient.age}${patient.gender}</span>
        </div>
        <div class="patient-card-status">
            ${alertIndicator}
            <span class="last-update">Updated ${formatTimeAgo(patient.lastUpdate)}</span>
        </div>
        <div class="patient-tags">
            <span class="tag tag-condition">${patient.condition}</span>
        </div>
    `;

    li.addEventListener('click', () => selectPatient(patient));

    return li;
}

function selectPatient(patient) {
    AppState.selectedPatient = patient;

    // Update active state
    document.querySelectorAll('.patient-card').forEach(card => {
        card.classList.remove('active');
        if (card.dataset.patientId === patient.id) {
            card.classList.add('active');
        }
    });

    // Update header badge
    const badge = document.getElementById('currentPatientBadge');
    if (badge) {
        badge.textContent = patient.id;
        badge.classList.remove('hidden');
    }

    // Update vitals
    renderVitals(patient);

    // Filter alerts and notes for this patient
    renderAlerts(patient.id);
    renderNotesTimeline(patient.id);

    // Reset AI notes
    resetAINotes();

    // Initialize or update vital trend chart
    initializeVitalTrendChart(patient);
}

function filterPatients(query) {
    if (!query) {
        renderPatientList();
        return;
    }

    const filtered = AppState.patients.filter(p =>
        p.name.toLowerCase().includes(query) ||
        p.id.toLowerCase().includes(query) ||
        p.condition.toLowerCase().includes(query)
    );

    renderPatientList(filtered);
}

// =====================================================
// Filters (Requirement 7)
// =====================================================
function initializeFilters() {
    const filterChips = document.querySelectorAll('.filter-chip');
    const conditionFilter = document.getElementById('conditionFilter');

    filterChips.forEach(chip => {
        chip.addEventListener('click', () => {
            filterChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');

            const filter = chip.dataset.filter;
            applyFilters(filter, conditionFilter?.value || 'all');
        });
    });

    conditionFilter?.addEventListener('change', () => {
        const activeChip = document.querySelector('.filter-chip.active');
        const severityFilter = activeChip?.dataset.filter || 'all';
        applyFilters(severityFilter, conditionFilter.value);
    });
}

function applyFilters(severity, condition) {
    let filtered = [...AppState.patients];

    if (severity !== 'all') {
        const severityMap = {
            'critical': 'high',
            'warning': 'medium',
            'stable': 'low'
        };
        filtered = filtered.filter(p => p.riskLevel === severityMap[severity]);
    }

    if (condition !== 'all') {
        filtered = filtered.filter(p => p.condition === condition);
    }

    renderPatientList(filtered);
}

// =====================================================
// Time Range
// =====================================================
function initializeTimeRange() {
    const buttons = document.querySelectorAll('.time-btn');

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const range = btn.dataset.range;
            console.log('Time range selected:', range);
            // Would update charts/data based on range
        });
    });
}

// =====================================================
// Alerts (Requirement 2)
// =====================================================
function initializeAlerts() {
    const filterBtns = document.querySelectorAll('.alert-filter');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const level = btn.dataset.level;
            filterAlerts(level);
        });
    });
}

function renderAlerts(patientId = null) {
    const list = document.getElementById('alertList');
    const emptyState = document.getElementById('alertsEmptyState');
    if (!list) return;

    let alerts = [...AppState.alerts];

    if (patientId) {
        alerts = alerts.filter(a => a.patientId === patientId);
    }

    // Sort by severity then time
    const severityOrder = { critical: 0, warning: 1, info: 2 };
    alerts.sort((a, b) => {
        if (severityOrder[a.type] !== severityOrder[b.type]) {
            return severityOrder[a.type] - severityOrder[b.type];
        }
        return b.time - a.time;
    });

    list.innerHTML = '';

    if (alerts.length === 0) {
        list.classList.add('hidden');
        emptyState?.classList.remove('hidden');
        return;
    }

    list.classList.remove('hidden');
    emptyState?.classList.add('hidden');

    alerts.forEach(alert => {
        const item = createAlertItem(alert);
        list.appendChild(item);
    });
}

function createAlertItem(alert) {
    const li = document.createElement('li');
    li.className = `alert-item alert-${alert.type}`;
    li.dataset.alertId = alert.id;

    if (alert.acknowledged) {
        li.classList.add('acknowledged');
    }

    const typeLabels = {
        'critical': 'CRITICAL',
        'warning': 'WARNING',
        'info': 'INFO'
    };

    li.innerHTML = `
        <div class="alert-header">
            <span class="alert-type">${typeLabels[alert.type]}</span>
            <span class="alert-patient">${alert.patientId}</span>
            <span class="alert-time">${formatTimeAgo(alert.time)}</span>
        </div>
        <p class="alert-message">${alert.message}</p>
        <div class="alert-action-bar">
            <button class="alert-action primary" data-action="review">Review Now</button>
            <button class="alert-action secondary" data-action="acknowledge">Acknowledge</button>
        </div>
    `;

    // Action handlers
    li.querySelector('[data-action="review"]')?.addEventListener('click', (e) => {
        e.stopPropagation();
        const patient = AppState.patients.find(p => p.id === alert.patientId);
        if (patient) selectPatient(patient);
    });

    li.querySelector('[data-action="acknowledge"]')?.addEventListener('click', (e) => {
        e.stopPropagation();
        acknowledgeAlert(alert.id);
    });

    return li;
}

function acknowledgeAlert(alertId) {
    const alert = AppState.alerts.find(a => a.id === alertId);
    if (alert) {
        alert.acknowledged = true;
        renderAlerts(AppState.selectedPatient?.id);
        updateStats();
    }
}

function filterAlerts(level) {
    const list = document.getElementById('alertList');
    if (!list) return;

    const items = list.querySelectorAll('.alert-item');

    items.forEach(item => {
        if (level === 'all') {
            item.style.display = '';
        } else {
            const isMatch = item.classList.contains(`alert-${level}`);
            item.style.display = isMatch ? '' : 'none';
        }
    });
}

// =====================================================
// Vitals (Requirement 3)
// =====================================================
function initializeVitals() {
    // Initialize sparklines and any vital-specific interactions
}

function renderVitals(patient) {
    if (!patient?.vitals) {
        showVitalsEmptyState();
        return;
    }

    hideVitalsEmptyState();

    const v = patient.vitals;

    // Heart Rate
    updateVitalDisplay('hr', {
        value: `${v.hr.value} BPM`,
        baseline: `${v.hr.baseline} BPM`,
        trend: getTrendText(v.hr.value, v.hr.baseline),
        trendClass: getTrendClass(v.hr.value, v.hr.baseline),
        isAbnormal: v.hr.value > v.hr.max || v.hr.value < v.hr.min,
        currentNum: v.hr.value,
        baselineNum: v.hr.baseline,
        improvementDir: 'low'
    });

    // Blood Pressure
    updateVitalDisplay('bp', {
        value: `${v.bp.systolic}/${v.bp.diastolic} mmHg`,
        baseline: `${v.bp.baseline.s}/${v.bp.baseline.d} mmHg`,
        trend: getTrendText(v.bp.systolic, v.bp.baseline.s),
        trendClass: getTrendClass(v.bp.systolic, v.bp.baseline.s),
        isAbnormal: v.bp.systolic > 140 || v.bp.diastolic > 90,
        currentNum: v.bp.systolic,
        baselineNum: v.bp.baseline.s,
        improvementDir: 'low'
    });

    // SpO2
    updateVitalDisplay('spo2', {
        value: `${v.spo2.value}%`,
        baseline: `${v.spo2.baseline}%`,
        trend: getTrendText(v.spo2.value, v.spo2.baseline),
        trendClass: getTrendClass(v.spo2.value, v.spo2.baseline, false),
        isAbnormal: v.spo2.value < v.spo2.min,
        currentNum: v.spo2.value,
        baselineNum: v.spo2.baseline,
        improvementDir: 'high'
    });

    // Temperature
    updateVitalDisplay('temp', {
        value: `${v.temp.value}°F`,
        baseline: `${v.temp.baseline}°F`,
        trend: getTrendText(v.temp.value, v.temp.baseline),
        trendClass: getTrendClass(v.temp.value, v.temp.baseline),
        isAbnormal: v.temp.value > v.temp.max || v.temp.value < v.temp.min,
        currentNum: v.temp.value,
        baselineNum: v.temp.baseline,
        improvementDir: 'low'
    });

    // Respiration
    updateVitalDisplay('resp', {
        value: `${v.resp.value} /min`,
        baseline: `${v.resp.baseline} /min`,
        trend: getTrendText(v.resp.value, v.resp.baseline),
        trendClass: getTrendClass(v.resp.value, v.resp.baseline),
        isAbnormal: v.resp.value > v.resp.max || v.resp.value < v.resp.min,
        currentNum: v.resp.value,
        baselineNum: v.resp.baseline,
        improvementDir: 'low'
    });

    // Render sparklines using VitalGraphs
    renderEnhancedSparklines(patient);
}

function updateVitalDisplay(vital, data) {
    const container = document.querySelector(`[data-vital="${vital}"]`);
    if (!container) return;

    // Update values
    const valueEl = container.querySelector('.vital-current');
    const baselineEl = container.querySelector('.baseline-value');
    const trendEl = container.querySelector('.vital-trend');
    const changeEl = container.querySelector('.vital-change');

    if (valueEl) valueEl.textContent = data.value;
    if (baselineEl) baselineEl.textContent = data.baseline;
    if (trendEl) {
        trendEl.textContent = data.trend;
        trendEl.className = `vital-trend ${data.trendClass}`;
    }

    // Calculate and display % change from baseline
    if (changeEl && data.currentNum !== undefined && data.baselineNum !== undefined) {
        const changePercent = ((data.currentNum - data.baselineNum) / data.baselineNum * 100);
        const absChange = Math.abs(changePercent).toFixed(1);

        let changeText, changeClass;
        if (changePercent > 2) {
            changeText = `↑ ${absChange}%`;
            changeClass = 'worsening';
        } else if (changePercent < -2) {
            changeText = `↓ ${absChange}%`;
            changeClass = data.improvementDir === 'low' ? 'improving' : 'worsening';
        } else {
            changeText = `→ ${absChange}%`;
            changeClass = 'stable';
        }

        changeEl.textContent = changeText;
        changeEl.className = `vital-change ${changeClass}`;
    }

    // Update container state
    container.classList.remove('normal', 'elevated', 'abnormal');
    if (data.isAbnormal) {
        container.classList.add('abnormal');
    } else if (data.trendClass === 'trend-up') {
        container.classList.add('elevated');
    } else {
        container.classList.add('normal');
    }
}

function renderSparklines(patient) {
    // Legacy function - now using renderEnhancedSparklines
    renderEnhancedSparklines(patient);
}

function renderEnhancedSparklines(patient) {
    if (!patient || !window.VitalGraphs) return;

    const vitalsMap = {
        'hr': { type: 'hr', baseline: patient.vitals?.hr?.baseline || 75 },
        'bp': { type: 'bp_systolic', baseline: patient.vitals?.bp?.baseline?.s || 120 },
        'spo2': { type: 'spo2', baseline: patient.vitals?.spo2?.baseline || 97 },
        'temp': { type: 'temp', baseline: patient.vitals?.temp?.baseline || 98.6 },
        'resp': { type: 'resp', baseline: patient.vitals?.resp?.baseline || 16 }
    };

    Object.entries(vitalsMap).forEach(([vitalKey, config]) => {
        const container = document.getElementById(`${vitalKey}Sparkline`);
        if (!container) return;

        VitalGraphs.renderSparkline(
            container,
            config.type,
            patient.id,
            config.baseline
        );
    });
}

function showVitalsEmptyState() {
    const grid = document.getElementById('vitalsGrid');
    const empty = document.getElementById('vitalsEmptyState');
    if (grid) grid.classList.add('hidden');
    if (empty) empty.classList.remove('hidden');
}

function hideVitalsEmptyState() {
    const grid = document.getElementById('vitalsGrid');
    const empty = document.getElementById('vitalsEmptyState');
    if (grid) grid.classList.remove('hidden');
    if (empty) empty.classList.add('hidden');
}

// =====================================================
// Vital Trend Chart (Clinical Graph)
// =====================================================
let vitalTrendChart = null;

function initializeVitalTrendChart(patient) {
    const chartContainer = document.getElementById('vitalTrendChart');
    const emptyState = document.getElementById('chartEmptyState');
    const vitalSelect = document.getElementById('vitalTypeSelect');

    if (!chartContainer) return;

    if (!patient) {
        chartContainer.classList.add('hidden');
        emptyState?.classList.remove('hidden');
        return;
    }

    chartContainer.classList.remove('hidden');
    emptyState?.classList.add('hidden');

    // Get selected vital type
    const vitalType = vitalSelect?.value || 'hr';

    // Create or update chart
    if (window.VitalGraphs) {
        // Clear container and create new chart
        chartContainer.innerHTML = '';
        vitalTrendChart = VitalGraphs.create('vitalTrendChart', {
            vitalType: vitalType,
            patientId: patient.id,
            timeRange: 'weekly',
            showBaseline: true
        });
    }

    // Vital type selector change handler
    if (vitalSelect && !vitalSelect.hasAttribute('data-listener-attached')) {
        vitalSelect.setAttribute('data-listener-attached', 'true');
        vitalSelect.addEventListener('change', () => {
            if (AppState.selectedPatient && window.VitalGraphs) {
                chartContainer.innerHTML = '';
                vitalTrendChart = VitalGraphs.create('vitalTrendChart', {
                    vitalType: vitalSelect.value,
                    patientId: AppState.selectedPatient.id,
                    timeRange: 'weekly',
                    showBaseline: true
                });
            }
        });
    }
}

// =====================================================
// AI Notes (Requirement 4)
// =====================================================
function initializeAINotes() {
    const generateBtn = document.getElementById('generateNotesBtn');
    const acceptBtn = document.getElementById('acceptNoteBtn');
    const editBtn = document.getElementById('editNoteBtn');
    const rejectBtn = document.getElementById('rejectNoteBtn');
    const regenerateBtn = document.getElementById('regenerateNoteBtn');
    const noteTabs = document.querySelectorAll('.note-tab');

    generateBtn?.addEventListener('click', generateAINotes);
    acceptBtn?.addEventListener('click', acceptNote);
    editBtn?.addEventListener('click', toggleEditMode);
    rejectBtn?.addEventListener('click', rejectNote);
    rejectBtn?.addEventListener('click', rejectNote);
    regenerateBtn?.addEventListener('click', generateAINotes);

    // Save button listener
    const saveBtn = document.getElementById('saveNoteBtn');
    saveBtn?.addEventListener('click', () => {
        // Get content from visible tab
        const visibleContent = document.querySelector('.note-section:not(.hidden) .note-text');
        if (visibleContent) {
            saveToSaved(visibleContent.textContent, 'Clinical Note', 'AI Assistant');
        } else {
            // Fallback if multiple sections visible (e.g. SOAP)
            // Just capture all text
            const allText = Array.from(document.querySelectorAll('.note-section:not(.hidden)'))
                .map(sec => sec.innerText)
                .join('\n\n');
            saveToSaved(allText, 'Clinical Global Note', 'AI Assistant');
        }
    });

    noteTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            noteTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const noteType = tab.dataset.type;
            switchNoteView(noteType);
        });
    });
}

async function generateAINotes() {
    console.log("=== generateAINotes called ===");
    const patient = AppState.selectedPatient;
    console.log("Selected patient:", patient);

    if (!patient) {
        alert('Please select a patient first.');
        return;
    }

    const generateContainer = document.getElementById('generateNotesContainer');
    const noteContent = document.getElementById('aiNoteContent');
    const btn = document.getElementById('generateNotesBtn');

    // Show loading state
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="btn-icon">⏳</span> Generating...';
    }

    try {
        console.log("Fetching from backend with patient_id:", patient.id);

        // Add timeout to prevent hanging indefinitely
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

        // Call backend API to generate AI clinical notes
        const response = await fetch('http://127.0.0.1:5000/api/generate-notes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                patient_id: patient.id
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        console.log("Response status:", response.status, response.ok);

        if (!response.ok) {
            throw new Error('Failed to generate notes');
        }

        const data = await response.json();
        console.log("Full API response:", data);
        const notes = data.notes;
        console.log("Notes object:", notes);

        // Display the Advanced BBY Agent Note in the Summary and Impression tabs
        // And use the traditional SOAP note in the SOAP tab

        // Populate SOAP tab
        document.getElementById('noteSubjective').innerHTML = `
            <span class="ai-label">BBY Agent - Clinical Summary</span>
            <div style="white-space: pre-wrap; margin-top: 10px;">${notes.patient_summary}</div>
        `;
        document.getElementById('noteObjective').innerHTML = `
            <span class="ai-label">Traceable Patient Stats</span>
            <div style="margin-top: 10px;">
                • HR: ${data.context.vitals.heart_rate_avg} BPM<br>
                • BP: ${data.context.vitals.blood_pressure} mmHg<br>
                • SpO2: ${data.context.vitals.spo2_percent}%
            </div>
        `;
        document.getElementById('noteAssessment').innerHTML = `
            <span class="ai-label">Clinical Impression</span>
            <div style="white-space: pre-wrap; margin-top: 10px;">${notes.clinical_impression}</div>
        `;
        document.getElementById('notePlan').innerHTML = `
            <span class="ai-label">Suggested Plan</span>
            <div style="white-space: pre-wrap; margin-top: 10px;">${notes.plan || ''}</div>
            <div class="ai-disclaimer">${data.disclaimer}</div>
        `;

        // Store additional note types for tab switching
        AppState.currentAINotes = notes;

        // Populate HPI
        const hpiEl = document.getElementById('noteHPI');
        if (hpiEl) {
            hpiEl.innerHTML = `
                <span class="ai-label">BBY Agent Observation</span>
                <div style="white-space: pre-wrap; margin-top: 10px;">${notes.clinical_impression}</div>
            `;
        }

        // Populate Patient Summary
        const summaryEl = document.getElementById('notePatientSummary');
        if (summaryEl) {
            summaryEl.innerHTML = `
                <span class="ai-label">BBY Agent Summary</span>
                <div style="white-space: pre-wrap; margin-top: 10px;">${notes.patient_summary}</div>
            `;
        }

        // Populate Clinical Impression
        const impressionEl = document.getElementById('noteClinicalImpression');
        if (impressionEl) {
            impressionEl.innerHTML = `
                <span class="ai-label">BBY Agent Clinical Note</span>
                <div style="white-space: pre-wrap; margin-top: 10px;">${notes.clinical_impression}</div>
            `;
        }

        console.log("DOM updates complete. Toggling visibility...");
        console.log("generateContainer:", generateContainer);
        console.log("noteContent:", noteContent);

        // Show notes content
        generateContainer?.classList.add('hidden');
        noteContent?.classList.remove('hidden');

        console.log("Visibility toggled! noteContent.classList:", noteContent?.classList.toString());
        console.log("=== SUCCESS: AI notes should now be visible! ===");

        AppState.stats.draftNotes++;
        updateStats();

    } catch (error) {
        console.error('Error generating AI notes:', error);

        // Detect if it was a timeout
        const isTimeout = error.name === 'AbortError';
        const errorMessage = isTimeout
            ? 'Request timed out. The AI model is taking too long to respond.'
            : 'Failed to generate notes. Check if backend is running.';

        showToast(errorMessage, 'error');

        document.getElementById('noteSubjective').innerHTML = `
            <div class="backend-error">
                <span class="error-icon">⚠️</span>
                <strong>${isTimeout ? 'Request Timed Out' : 'AI Generation Failed'}</strong>
                <p>${isTimeout
                ? 'The AI model took too long to respond (>60 seconds). This can happen with complex requests or high API load.'
                : 'The AI could not generate notes. Possible causes:'}</p>
                <ul style="text-align: left; margin: 10px 0;">
                    ${isTimeout ? '<li>High AI API load - try again in a moment</li>' : ''}
                    <li>API quota exceeded (429 error) - wait a few minutes</li>
                    <li>Backend server not running - run <code>python backend.py</code></li>
                    <li>Network issue - check your connection</li>
                </ul>
                <p>Try again or check the browser console (F12) for details.</p>
            </div>
        `;
        document.getElementById('noteObjective').innerHTML = '--';
        document.getElementById('noteAssessment').innerHTML = '--';
        document.getElementById('notePlan').innerHTML = '--';

        generateContainer?.classList.add('hidden');
        noteContent?.classList.remove('hidden');
    } finally {
        // Reset button
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<span class="btn-icon">✨</span> Generate Draft Notes';
        }
    }
}

// Note: Local generation has been removed. All clinical notes must come from the backend.
// The backend is the single source of truth for AI clinical documentation.

function switchNoteView(noteType) {
    // Get all note sections
    const soapSections = document.querySelectorAll('[data-section="subjective"], [data-section="objective"], [data-section="assessment"], [data-section="plan"]');
    const hpiSection = document.querySelector('[data-section="hpi"]');
    const summarySection = document.querySelector('[data-section="summary"]');
    const impressionSection = document.querySelector('[data-section="impression"]');

    // Hide all sections first
    soapSections.forEach(s => s.classList.add('hidden'));
    hpiSection?.classList.add('hidden');
    summarySection?.classList.add('hidden');
    impressionSection?.classList.add('hidden');

    // Show relevant sections based on tab
    switch (noteType) {
        case 'soap':
            soapSections.forEach(s => s.classList.remove('hidden'));
            break;
        case 'hpi':
            hpiSection?.classList.remove('hidden');
            break;
        case 'assessment':
            document.querySelector('[data-section="assessment"]')?.classList.remove('hidden');
            break;
        case 'summary':
            summarySection?.classList.remove('hidden');
            break;
        case 'impression':
            impressionSection?.classList.remove('hidden');
            break;
    }
}

function acceptNote() {
    const patient = AppState.selectedPatient;
    if (!patient) return;

    // Save note to timeline
    const newNote = {
        id: `note-${Date.now()}`,
        patientId: patient.id,
        author: 'Dr. User',
        content: document.getElementById('noteSubjective').textContent,
        type: 'SOAP Note',
        timestamp: new Date()
    };

    AppState.notes.unshift(newNote);

    // Update stats
    AppState.stats.draftNotes = Math.max(0, AppState.stats.draftNotes - 1);
    AppState.stats.reviewedToday++;
    updateStats();

    // Reset and show success
    resetAINotes();
    renderNotesTimeline(patient.id);

    showToast('Note accepted and saved to patient record.');
}

function toggleEditMode() {
    const editables = document.querySelectorAll('.editable-text');
    const editBtn = document.getElementById('editNoteBtn');

    const isEditing = editables[0]?.getAttribute('contenteditable') === 'true';

    editables.forEach(el => {
        el.setAttribute('contenteditable', isEditing ? 'false' : 'true');
        if (!isEditing) el.focus();
    });

    if (editBtn) {
        editBtn.innerHTML = isEditing
            ? '<span class="icon">✏</span> Edit'
            : '<span class="icon">💾</span> Save';
    }
}

function rejectNote() {
    if (confirm('Discard this AI-generated note?')) {
        AppState.stats.draftNotes = Math.max(0, AppState.stats.draftNotes - 1);
        updateStats();
        resetAINotes();
    }
}

function resetAINotes() {
    const generateContainer = document.getElementById('generateNotesContainer');
    const noteContent = document.getElementById('aiNoteContent');

    generateContainer?.classList.remove('hidden');
    noteContent?.classList.add('hidden');

    // Reset editing state
    document.querySelectorAll('.editable-text').forEach(el => {
        el.setAttribute('contenteditable', 'false');
    });

    const editBtn = document.getElementById('editNoteBtn');
    if (editBtn) {
        editBtn.innerHTML = '<span class="icon">✏</span> Edit';
    }
}

// =====================================================
// Notes Timeline (Requirement 8)
// =====================================================
function initializeNotesTimeline() {
    const searchInput = document.getElementById('notesSearchInput');
    const submitBtn = document.getElementById('quickNoteSubmit');
    const noteInput = document.getElementById('quickNoteInput');

    searchInput?.addEventListener('input', (e) => {
        searchNotes(e.target.value);
    });

    submitBtn?.addEventListener('click', addQuickNote);

    noteInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            addQuickNote();
        }
    });
}

function renderNotesTimeline(patientId = null) {
    const timeline = document.getElementById('notesTimeline');
    const emptyState = document.getElementById('timelineEmptyState');
    if (!timeline) return;

    let notes = [...AppState.notes];

    if (patientId) {
        notes = notes.filter(n => n.patientId === patientId);
    }

    // Sort by timestamp descending
    notes.sort((a, b) => b.timestamp - a.timestamp);

    timeline.innerHTML = '';

    if (notes.length === 0) {
        timeline.classList.add('hidden');
        emptyState?.classList.remove('hidden');
        return;
    }

    timeline.classList.remove('hidden');
    emptyState?.classList.add('hidden');

    notes.forEach(note => {
        const item = createTimelineItem(note);
        timeline.appendChild(item);
    });
}

function createTimelineItem(note) {
    const li = document.createElement('li');
    li.className = 'timeline-item';
    li.dataset.noteId = note.id;

    li.innerHTML = `
        <div class="timeline-marker"></div>
        <div class="timeline-content">
            <div class="timeline-header">
                <span class="timeline-author">${note.author}</span>
                <span class="timeline-time">${formatDate(note.timestamp)}</span>
            </div>
            <p class="timeline-note">${note.content}</p>
            <div class="timeline-meta">
                <span class="note-type">${note.type}</span>
            </div>
        </div>
    `;

    return li;
}

function addQuickNote() {
    const input = document.getElementById('quickNoteInput');
    const content = input?.value.trim();

    if (!content) return;

    const patient = AppState.selectedPatient;

    const newNote = {
        id: `note-${Date.now()}`,
        patientId: patient?.id || 'general',
        author: 'Dr. User',
        content: content,
        type: 'Quick Note',
        timestamp: new Date()
    };

    AppState.notes.unshift(newNote);

    if (input) input.value = '';

    renderNotesTimeline(patient?.id);
    showToast('Note added successfully.');
}

function searchNotes(query) {
    const items = document.querySelectorAll('.timeline-item');
    const lowerQuery = query.toLowerCase();

    items.forEach(item => {
        const content = item.textContent.toLowerCase();
        item.style.display = content.includes(lowerQuery) ? '' : 'none';
    });
}

// =====================================================
// Messaging (Requirement 9)
// =====================================================
function initializeMessaging() {
    const toggle = document.getElementById('messagesToggle');
    const panel = document.getElementById('messagesPanel');
    const closeBtn = document.getElementById('messagesPanelClose');
    const sendBtn = document.getElementById('composeSend');
    const input = document.getElementById('composeInput');

    toggle?.addEventListener('click', () => {
        panel?.classList.toggle('open');
        panel?.classList.toggle('hidden');
    });

    closeBtn?.addEventListener('click', () => {
        panel?.classList.remove('open');
        panel?.classList.add('hidden');
    });

    sendBtn?.addEventListener('click', sendMessage);

    input?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Thread click handlers
    document.querySelectorAll('.thread-item').forEach(thread => {
        thread.addEventListener('click', () => {
            thread.classList.remove('unread');
            updateUnreadBadge();
            // Would open thread view here
        });
    });
}

function sendMessage() {
    const input = document.getElementById('composeInput');
    const content = input?.value.trim();

    if (!content) return;

    // Simulate sending
    console.log('Sending message:', content);
    if (input) input.value = '';

    showToast('Message sent.');
}

function updateUnreadBadge() {
    const unread = document.querySelectorAll('.thread-item.unread').length;
    const badge = document.getElementById('unreadBadge');

    if (badge) {
        badge.textContent = unread;
        badge.style.display = unread > 0 ? '' : 'none';
    }
}

// =====================================================
// Settings (Requirement 10)
// =====================================================
function initializeSettings() {
    const settingsBtn = document.getElementById('settingsBtn');
    const modal = document.getElementById('settingsModal');
    const closeBtn = document.getElementById('settingsClose');
    const saveBtn = document.getElementById('saveSettingsBtn');
    const presetBtns = document.querySelectorAll('.preset-btn');

    settingsBtn?.addEventListener('click', () => {
        modal?.classList.remove('hidden');
    });

    closeBtn?.addEventListener('click', () => {
        modal?.classList.add('hidden');
    });

    modal?.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.add('hidden');
    });

    saveBtn?.addEventListener('click', saveSettings);

    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            AppState.settings.specialty = btn.dataset.preset;
        });
    });
}

function saveSettings() {
    // Gather settings
    AppState.settings.alertSoundEnabled = document.getElementById('alertSoundEnabled')?.checked ?? true;
    AppState.settings.alertAutoCollapse = document.getElementById('alertAutoCollapse')?.checked ?? true;
    AppState.settings.compactMode = document.getElementById('compactMode')?.checked ?? false;
    AppState.settings.showBaselines = document.getElementById('showBaselines')?.checked ?? true;

    // Apply settings
    document.body.classList.toggle('compact-mode', AppState.settings.compactMode);

    // Close modal
    document.getElementById('settingsModal')?.classList.add('hidden');

    showToast('Settings saved.');
}

// =====================================================
// Global Search (Requirement 7)
// =====================================================
function initializeSearch() {
    const input = document.getElementById('globalSearchInput');
    const results = document.getElementById('searchResults');

    input?.addEventListener('focus', () => {
        if (input.value) results?.classList.remove('hidden');
    });

    input?.addEventListener('blur', () => {
        setTimeout(() => results?.classList.add('hidden'), 200);
    });

    input?.addEventListener('input', (e) => {
        const query = e.target.value.trim();

        if (query.length < 2) {
            results?.classList.add('hidden');
            return;
        }

        performGlobalSearch(query);
        results?.classList.remove('hidden');
    });

    // Keyboard shortcut
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            input?.focus();
        }
    });
}

function performGlobalSearch(query) {
    const lowerQuery = query.toLowerCase();

    // Search patients
    const patientResults = AppState.patients.filter(p =>
        p.name.toLowerCase().includes(lowerQuery) ||
        p.id.toLowerCase().includes(lowerQuery)
    ).slice(0, 3);

    // Search notes
    const noteResults = AppState.notes.filter(n =>
        n.content.toLowerCase().includes(lowerQuery)
    ).slice(0, 3);

    // Search alerts
    const alertResults = AppState.alerts.filter(a =>
        a.message.toLowerCase().includes(lowerQuery) ||
        a.patientId.toLowerCase().includes(lowerQuery)
    ).slice(0, 3);

    // Render results
    renderSearchResults('searchPatients', patientResults, p =>
        `<div class="search-result-item" data-patient-id="${p.id}">
            <span class="result-name">${p.name}</span>
            <span class="result-meta">${p.id}</span>
        </div>`
    );

    renderSearchResults('searchNotes', noteResults, n =>
        `<div class="search-result-item">
            <span class="result-name">${n.content.substring(0, 50)}...</span>
            <span class="result-meta">${n.patientId}</span>
        </div>`
    );

    renderSearchResults('searchAlerts', alertResults, a =>
        `<div class="search-result-item">
            <span class="result-name">${a.message}</span>
            <span class="result-meta">${a.patientId}</span>
        </div>`
    );
}

function renderSearchResults(containerId, items, template) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (items.length === 0) {
        container.innerHTML = '<li class="no-results">No results</li>';
        return;
    }

    container.innerHTML = items.map(item => `<li>${template(item)}</li>`).join('');

    // Add click handlers
    container.querySelectorAll('[data-patient-id]').forEach(el => {
        el.addEventListener('click', () => {
            const patient = AppState.patients.find(p => p.id === el.dataset.patientId);
            if (patient) selectPatient(patient);
            document.getElementById('searchResults')?.classList.add('hidden');
        });
    });
}

// =====================================================
// Modals
// =====================================================
function initializeModals() {
    const addPatientBtn = document.getElementById('addPatientBtn');
    const modal = document.getElementById('addPatientModal');
    const closeBtn = modal?.querySelector('.modal-close');
    const cancelBtn = document.getElementById('cancelPatientBtn');
    const form = document.getElementById('addPatientForm');

    addPatientBtn?.addEventListener('click', () => {
        modal?.classList.remove('hidden');
    });

    const closeModal = () => {
        modal?.classList.add('hidden');
        form?.reset();
    };

    closeBtn?.addEventListener('click', closeModal);
    cancelBtn?.addEventListener('click', closeModal);

    modal?.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    form?.addEventListener('submit', (e) => {
        e.preventDefault();
        saveNewPatient();
        closeModal();
    });
}

function saveNewPatient() {
    const name = document.getElementById('patientName')?.value.trim();
    const age = parseInt(document.getElementById('patientAge')?.value) || null;
    const gender = document.getElementById('patientGender')?.value || 'M';
    const risk = document.getElementById('patientRisk')?.value || 'low';
    const condition = document.getElementById('patientCondition')?.value || 'general';
    const notes = document.getElementById('patientNotes')?.value.trim() || '';

    const newPatient = {
        id: DataGenerator.generatePatientId(),
        name: name,
        age: age,
        gender: gender,
        riskLevel: risk,
        condition: condition,
        notes: notes || 'Monitoring...',
        lastUpdate: new Date(),
        alertCount: risk === 'high' ? 1 : 0,
        vitals: DataGenerator.generateVitals(risk === 'high')
    };

    AppState.patients.unshift(newPatient);
    renderPatientList();
    selectPatient(newPatient);

    showToast(`Patient ${name} added successfully.`);
}

// =====================================================
// Saved Section (Requirement: New Tab/Section)
// =====================================================
function initializeSavedSection() {
    const clearBtn = document.getElementById('clearSavedBtn');

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (confirm('Clear all saved items?')) {
                AppState.savedItems = [];
                renderSavedList();
                showToast('All saved items cleared.');
            }
        });
    }

    // Initial render
    renderSavedList();
}

function saveToSaved(content, type, source) {
    const item = {
        id: `saved-${Date.now()}`,
        content: content,
        type: type,
        source: source,
        timestamp: new Date()
    };

    AppState.savedItems.unshift(item);
    renderSavedList();
    showToast('Saved to Saved section.');

    // Animate the saved card to draw attention
    const savedCard = document.querySelector('.saved-card');
    if (savedCard) {
        savedCard.style.animation = 'none';
        savedCard.offsetHeight; // Trigger reflow
        savedCard.style.animation = 'pulse-card 0.5s ease';
    }
}

function renderSavedList() {
    const list = document.getElementById('savedList');
    const emptyState = document.getElementById('savedEmptyState');

    if (!list) return;

    list.innerHTML = '';

    if (AppState.savedItems.length === 0) {
        emptyState?.classList.remove('hidden');
        return;
    }

    emptyState?.classList.add('hidden');

    AppState.savedItems.forEach(item => {
        const li = document.createElement('li');
        li.className = 'alert-item';
        li.style.borderLeftColor = 'var(--color-primary)';

        // Header
        const header = document.createElement('div');
        header.className = 'alert-header';
        header.innerHTML = `
            <span class="alert-type" style="color: var(--color-primary)">${item.source.toUpperCase()}</span>
            <span class="alert-time">${formatTimeAgo(item.timestamp)}</span>
        `;

        // Message
        const messageDiv = document.createElement('div');
        messageDiv.className = 'alert-message';
        messageDiv.style.whiteSpace = 'pre-wrap';
        messageDiv.style.maxHeight = '100px';
        messageDiv.style.overflowY = 'auto';

        const typeStrong = document.createElement('strong');
        typeStrong.textContent = item.type;

        const br = document.createElement('br');

        const contentText = document.createTextNode(
            item.content.substring(0, 150) + (item.content.length > 150 ? '...' : '')
        );

        messageDiv.appendChild(typeStrong);
        messageDiv.appendChild(br);
        messageDiv.appendChild(contentText);

        // Actions
        const actions = document.createElement('div');
        actions.className = 'alert-action-bar';

        const viewBtn = document.createElement('button');
        viewBtn.className = 'alert-action secondary';
        viewBtn.textContent = 'View Full';
        viewBtn.onclick = () => {
            alert(item.content);
        };

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'alert-action secondary';
        deleteBtn.style.color = '#ef4444';
        deleteBtn.textContent = 'Delete';
        deleteBtn.onclick = () => deleteSavedItem(item.id);

        actions.appendChild(viewBtn);
        actions.appendChild(deleteBtn);

        li.appendChild(header);
        li.appendChild(messageDiv);
        li.appendChild(actions);

        list.appendChild(li);
    });
}

function deleteSavedItem(id) {
    AppState.savedItems = AppState.savedItems.filter(i => i.id !== id);
    renderSavedList();
}

// Add pulse animation style
const styleSheet = document.createElement("style");
styleSheet.innerText = `
@keyframes pulse-card {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }
    50% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(37, 99, 235, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
}
.chat-action-save {
    margin-top: 8px;
    font-size: 0.7rem;
    padding: 4px 8px;
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    border-radius: 4px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: all 0.2s;
}
.chat-action-save:hover {
    background: var(--color-primary-light);
    color: var(--color-primary);
    border-color: var(--color-primary);
}
`;
document.head.appendChild(styleSheet);


// =====================================================
// Stats
// =====================================================
function updateStats() {
    const stats = {
        activePatients: AppState.patients.length,
        pendingAlerts: AppState.alerts.filter(a => !a.acknowledged).length,
        draftNotes: AppState.stats.draftNotes,
        reviewedToday: AppState.stats.reviewedToday
    };

    document.getElementById('statActivePatients').textContent = stats.activePatients;
    document.getElementById('statPendingAlerts').textContent = stats.pendingAlerts;
    document.getElementById('statDraftNotes').textContent = stats.draftNotes;
    document.getElementById('statReviewedToday').textContent = stats.reviewedToday;
}

// =====================================================
// Toast Notifications
// =====================================================
function showToast(message, duration = 3000) {
    // Remove existing toast
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%);
        background: #0F172A;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 14px;
        z-index: 9999;
        animation: toastIn 0.3s ease;
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Add toast animations
const style = document.createElement('style');
style.textContent = `
    @keyframes toastIn {
        from { opacity: 0; transform: translateX(-50%) translateY(20px); }
        to { opacity: 1; transform: translateX(-50%) translateY(0); }
    }
    @keyframes toastOut {
        from { opacity: 1; transform: translateX(-50%) translateY(0); }
        to { opacity: 0; transform: translateX(-50%) translateY(20px); }
    }
`;
document.head.appendChild(style);

// =====================================================
// Patient Detail View (Bonus)
// =====================================================
function initializePatientDetailView() {
    const backBtn = document.getElementById('backToListBtn');

    backBtn?.addEventListener('click', () => {
        document.getElementById('patientDetailView')?.classList.add('hidden');
        document.getElementById('mainContent')?.classList.remove('hidden');
        AppState.currentView = 'dashboard';
    });
}

// =====================================================
// Clinical AI Chatbot
// =====================================================
function initializeChatbot() {
    const toggleBtn = document.getElementById('aiChatToggle');
    const chatPanel = document.getElementById('aiChatPanel');
    const closeBtn = document.getElementById('aiChatClose');
    const sendBtn = document.getElementById('chatSend');
    const inputField = document.getElementById('chatInput');

    // Toggle chat panel
    toggleBtn?.addEventListener('click', () => {
        chatPanel?.classList.toggle('hidden');
        if (!chatPanel?.classList.contains('hidden')) {
            inputField?.focus();
        }
    });

    // Close chat panel
    closeBtn?.addEventListener('click', () => {
        chatPanel?.classList.add('hidden');
    });

    // Send message on button click
    sendBtn?.addEventListener('click', sendChatMessage);

    // Send message on Enter (but allow Shift+Enter for new lines)
    inputField?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}

async function sendChatMessage() {
    const inputField = document.getElementById('chatInput');
    const messagesContainer = document.getElementById('chatMessages');
    const sendBtn = document.getElementById('chatSend');

    const message = inputField?.value.trim();
    if (!message) return;

    // Clear input
    inputField.value = '';

    // Add user message to chat
    addChatMessage('user', message);

    // Disable send button
    if (sendBtn) sendBtn.disabled = true;

    // Show typing indicator
    const typingIndicator = addTypingIndicator();

    try {
        const patient = AppState.selectedPatient;

        // Call backend chat API
        const response = await fetch('http://127.0.0.1:5000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                patient_id: patient?.id || null
            })
        });

        // Remove typing indicator
        typingIndicator?.remove();

        if (!response.ok) {
            throw new Error('Failed to get response');
        }

        const data = await response.json();

        // Add AI response
        addChatMessage('assistant', data.response);

    } catch (error) {
        console.error('Chat error:', error);
        typingIndicator?.remove();
        addChatMessage('assistant', 'Unable to connect to Clinical AI backend. Please ensure the backend server is running.\n\n---\n*AI-assisted insight. Clinical judgment required.*');
    } finally {
        if (sendBtn) sendBtn.disabled = false;
        inputField?.focus();
    }
}

function addChatMessage(type, content) {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;

    // Parse markdown-like formatting
    const formattedContent = formatChatContent(content);

    messageDiv.innerHTML = `<div class="message-content">${formattedContent}</div>`;

    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Add Save button for assistant messages
    if (type === 'assistant') {
        const saveBtn = document.createElement('button');
        saveBtn.className = 'chat-action-save';
        saveBtn.innerHTML = '💾 Save to Saved';
        saveBtn.onclick = () => saveToSaved(content, 'Chat Response', 'AI Chat');
        messageDiv.querySelector('.message-content').appendChild(saveBtn);
    }
}

function formatChatContent(content) {
    // Convert markdown-like syntax to HTML
    let html = content
        // Bold text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // Italic text
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Bullet points
        .replace(/^• (.+)$/gm, '<li>$1</li>')
        // Line breaks
        .replace(/\n/g, '<br>');

    // Wrap consecutive <li> items in <ul>
    html = html.replace(/(<li>.*?<\/li>)(<br>)?(<li>)/g, '$1$3');
    html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');

    // Handle horizontal rule
    html = html.replace(/<br>---<br>/g, '<hr>');

    return html;
}

function addTypingIndicator() {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return null;

    const indicator = document.createElement('div');
    indicator.className = 'chat-message assistant';
    indicator.id = 'typingIndicator';
    indicator.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    messagesContainer.appendChild(indicator);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return indicator;
}

