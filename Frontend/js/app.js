/* ==========================================================================
   ASHA-Sahaayak Application Core JavaScript
   ========================================================================== */

const API_BASE_URL = 'http://localhost:5000';

// Default patient database seed
const DEFAULT_PATIENTS = [
  { id: "1", name: "Sunita Devi", age: 26, village: "Karmapur", contact: "+91 98765 43211", month: 7, bloodGroup: "B+", notes: "Feet swelling and mild headache. Blood pressure recorded at 140/90. High-risk maternal monitoring recommended.", risk: "HIGH", date: "2026-05-30" },
  { id: "2", name: "Meena Kumari", age: 23, village: "Bhilai", contact: "+91 98765 43212", month: 5, bloodGroup: "O+", notes: "Reports mild nausea and temperature spikes. Hydration check advised.", risk: "MEDIUM", date: "2026-05-29" },
  { id: "3", name: "Rekha Pandey", age: 29, village: "Durg", contact: "+91 98765 43213", month: 3, bloodGroup: "A+", notes: "Routine checkup. Normal fetal heartbeat and baseline vitals.", risk: "LOW", date: "2026-05-28" },
  { id: "4", name: "Laxmi Bai", age: 32, village: "Raipur", contact: "+91 98765 43214", month: 8, bloodGroup: "AB+", contact: "+91 98765 43214", notes: "Severe dizziness when standing up. Elevated blood pressure history.", risk: "HIGH", date: "2026-05-27" },
  { id: "5", name: "Anita Sahu", age: 24, village: "Simga", contact: "+91 98765 43215", month: 6, bloodGroup: "O-", notes: "Normal checkup, active fetal movements. Hydration and iron supplements check.", risk: "LOW", date: "2026-05-26" }
];

// Initialize and manage localStorage data
function initializeDatabase() {
  if (!localStorage.getItem('patients')) {
    localStorage.setItem('patients', JSON.stringify(DEFAULT_PATIENTS));
  }
}

// Get all patients
function getPatients() {
  initializeDatabase();
  return JSON.parse(localStorage.getItem('patients'));
}

// Save a new patient
function savePatient(patient) {
  const patients = getPatients();
  patient.id = (patients.length + 1).toString();
  patient.date = new Date().toISOString().split('T')[0];
  patients.unshift(patient);
  localStorage.setItem('patients', JSON.stringify(patients));
  return patient;
}

// Sidebar responsive drawer controls
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (sidebar) {
    sidebar.classList.toggle('open');
  }
}

// Format date nicely
function getFormattedDate(dateStr) {
  const date = dateStr ? new Date(dateStr) : new Date();
  return date.toLocaleDateString('en-IN', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
}

// Highlight active sidebar navigation based on pathname
function highlightActiveNav() {
  const path = window.location.pathname.split('/').pop() || 'index.html';
  const navItems = document.querySelectorAll('.sidebar-nav .nav-item');
  
  navItems.forEach(item => {
    item.classList.remove('active');
    const href = item.getAttribute('href');
    if (href === path) {
      item.classList.add('active');
    }
  });
}

// Seeding and setting dropdowns/lists on load
document.addEventListener('DOMContentLoaded', () => {
  initializeDatabase();
  highlightActiveNav();

  // Set today's date in header if placeholder exists
  const dateEl = document.getElementById('currentDate');
  if (dateEl) {
    dateEl.textContent = getFormattedDate();
  }

  // Populate patient dropdown in voice-input screen
  const selectEl = document.getElementById('selectedPatient');
  if (selectEl) {
    selectEl.innerHTML = '<option value="">Choose a patient...</option>';
    const patients = getPatients();
    patients.forEach(p => {
      const option = document.createElement('option');
      option.value = p.id;
      option.textContent = `${p.name} — Village: ${p.village}, Month ${p.month}`;
      selectEl.appendChild(option);
    });

    // Handle preselected patient from URL
    const urlParams = new URLSearchParams(window.location.search);
    const preselectedId = urlParams.get('patientId');
    if (preselectedId) {
      selectEl.value = preselectedId;
    }
  }

  // Set patient registration list inside dashboard
  const patientListEl = document.querySelector('.patient-list');
  if (patientListEl && window.location.pathname.endsWith('dashboard.html')) {
    patientListEl.innerHTML = '';
    const patients = getPatients().slice(0, 5); // display top 5
    patients.forEach(p => {
      const item = document.createElement('div');
      item.className = 'patient-item';
      
      const letter = p.name.charAt(0);
      const gradient = getAvatarGradient(letter);
      const riskClass = `risk-badge--${p.risk.toLowerCase()}`;

      item.innerHTML = `
        <div class="patient-avatar" style="background: ${gradient}">${letter}</div>
        <div class="patient-info">
          <span class="patient-name">${p.name}</span>
          <span class="patient-meta">Village: ${p.village} · Month ${p.month}</span>
        </div>
        <span class="risk-badge ${riskClass}">${p.risk}</span>
      `;
      patientListEl.appendChild(item);
    });
  }
});

// Custom gradient colors for circular patient avatar blocks
function getAvatarGradient(letter) {
  const gradients = [
    'linear-gradient(135deg, #a855f7, #7c3aed)',
    'linear-gradient(135deg, #f59e0b, #d97706)',
    'linear-gradient(135deg, #06b6d4, #0284c7)',
    'linear-gradient(135deg, #ec4899, #be185d)',
    'linear-gradient(135deg, #16a34a, #059669)',
    'linear-gradient(135deg, #3b82f6, #1d4ed8)'
  ];
  const index = letter.charCodeAt(0) % gradients.length;
  return gradients[index];
}

// Show Toast Utility
function showToast(message, type = 'success') {
  let toast = document.getElementById('toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'toast';
    toast.id = 'toast';
    document.body.appendChild(toast);
  }
  
  toast.innerHTML = `
    <div class="toast-icon">${type === 'success' ? '✓' : '⚠️'}</div>
    <div class="toast-message">${message}</div>
  `;
  
  toast.classList.add('show');
  setTimeout(() => {
    toast.classList.remove('show');
  }, 3000);
}

/* ==========================================================================
   API Client & Fallback AI Simulation Engine
   ========================================================================== */

// Local JavaScript Healthcare Engine Simulator (Fallback)
const SIMULATED_SYMPTOMS = {
  fever: { keywords: ['fever', 'bukhar', 'temp', 'temperature', 'জ্বর', 'জ্বর', 'কাযচ্চল', 'તાવ', 'ಜ್ವರ', 'ജ്വരം', 'ଜ୍ୱର', 'ਤਾਵ', 'জ্বৰ'], name: 'Fever' },
  headache: { keywords: ['headache', 'sar dard', 'head pain', 'মাথা ব্যথা', 'தலைவலி', 'માથાનો દુઃખ', 'ತಲೆ ನೋವು', 'തലവേദന', 'ମାଥା ଦୁଃଖ', 'ਸਿਰ ਦਰਦ', 'মূথা ব্যথা'], name: 'Headache' },
  dizziness: { keywords: ['dizziness', 'chakkar', 'dizzy', 'চক্কর', 'தலைசுற்றல்', 'ચક્કર', 'ಚಕ್ಕರ', 'തലസുഴി', 'ଚକର', 'ਚੱਕਰ', 'চকৰ'], name: 'Dizziness' },
  swelling: { keywords: ['swelling', 'sujan', 'swell', 'সোয়া', 'வீக்கம்', 'સૂજન', 'ಹೊಳು', 'വീക്ക്', 'ସୂଜନ', 'ਸੋਜ਼ਨ', 'সুজন'], name: 'Swelling' },
  vomiting: { keywords: ['vomiting', 'ulti', 'nausea', 'বমি', 'வாந்தி', 'ઉલટી', 'ವಾಂತಿ', 'ഛാനകം', 'ବନ୍ତି', 'ਉਲਟੀ', 'বমি'], name: 'Vomiting' },
  bleeding: { keywords: ['bleeding', 'khoon', 'blood', 'রক্তপাত', 'இரத்தப்போகம்', 'લોબો વટવું', 'ರಕ್ತಸ್ರಾವ', 'രക്തസ്രാവം', 'ରକ୍ତସ୍ରାବ', 'ਲਹੂ ਵਟਣਾ', 'রক্তস্ৰাব'], name: 'Bleeding' },
  weakness: { keywords: ['weakness', 'kamzori', 'tired', 'দুর্বলতা', 'பலவீனம்', 'નરમી', 'ದುರ್ಬಲತೆ', 'ദുർബലത', 'ଦୁର୍ବଳତା', 'ਕਮਜ਼ੋਰੀ', 'দুৰ্বলতা'], name: 'Weakness' },
  diarrhoea: { keywords: ['diarrhoea', 'dast', 'loose motion', 'ডায়রিয়া', 'வயிற்றுப்புணர்', 'દસત', 'ಜುರಾಯಿ', 'ജുരായി', 'ଡାଇରିଆ', 'ਦਸਤ', 'ডায়ৰিয়া'], name: 'Diarrhoea' }
};

function simulateHealthcareEngine(text) {
  const lowerText = text.toLowerCase();
  const detected = [];
  
  for (const [key, value] of Object.entries(SIMULATED_SYMPTOMS)) {
    if (value.keywords.some(keyword => lowerText.includes(keyword))) {
      detected.push(value.name);
    }
  }

  // extract month
  const monthMatch = lowerText.match(/(\d+)\s*(month|મહિનો|ମାସ|মাস|மாதம்|నెల|ತಿಂಗಳು|മാസം|महिना|माह|ਮਹੀਨਾ|মাহ)/);
  const pregnancyMonth = monthMatch ? monthMatch[1] : null;

  let riskLevel = 'LOW RISK';
  let guidance = 'No immediate maternal risk factors detected. Continue routine checkups.';

  if (detected.includes('Swelling') && detected.includes('Dizziness')) {
    riskLevel = 'HIGH RISK';
    guidance = 'Potential risk of gestational hypertension or preeclampsia. Regular BP checks, complete bed rest, and immediate PHC (Primary Health Center) referral are strongly recommended.';
  } else if (detected.includes('Bleeding')) {
    riskLevel = 'EMERGENCY';
    guidance = 'Emergency maternal concern detected (antepartum hemorrhage danger). Contact ambulance immediately. Refer patient to nearest emergency hospital without delay.';
  } else if (detected.includes('Fever')) {
    riskLevel = 'MEDIUM RISK';
    guidance = 'Fever detected. Monitor temperature, check hydration levels, suggest PCM, and refer to medical officer if it persists beyond 24 hours.';
  }

  return {
    input_text: text,
    symptoms: detected,
    pregnancy_month: pregnancyMonth,
    risk_level: riskLevel,
    guidance: guidance
  };
}

// Master API Client
const apiClient = {
  // Analyze text via backend or fallback
  async analyzeText(text) {
    // "asha_language" is set by the language selector on the input screen and
    // persisted in localStorage so it survives the navigation to results.
    const language = window.localStorage.getItem('asha_language') || 'en';
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language })
      });
      if (response.ok) {
        return await response.json();
      }
      throw new Error('API server returned error code');
    } catch (e) {
      console.warn('Backend server offline. Running local AI simulation engine.', e);
      return simulateHealthcareEngine(text);
    }
  },

  // Transcribe voice audio via Flask /api/transcribe (Whisper backend)
  async transcribeAudio(audioBlob) {
    const formData = new FormData();
    const ext = audioBlob.type.split('/')[1]?.split(';')[0] || 'webm';
    formData.append('audio', audioBlob, `voice_input.${ext}`);
    // Carry the selected language so the backend can use it for ASR + analysis.
    formData.append('language', window.localStorage.getItem('asha_language') || 'en');

    const response = await fetch(`${API_BASE_URL}/api/transcribe`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      let errMsg = `Server error ${response.status}`;
      try {
        const errData = await response.json();
        errMsg = errData.error || errMsg;
      } catch (_) { /* ignore */ }
      throw new Error(errMsg);
    }

    return await response.json();
  }
};

const LANGUAGE_MAP = {
  en: 'English',
  hi: 'हिंदी',
  bn: 'বাংলা',
  ta: 'தமிழ்',
  gu: 'ગુજરાતી',
  kn: 'ಕನ್ನಡ',
  ml: 'മലയാളം',
  or: 'ଓଡ଼ିଆ',
  pa: 'ਪੰਜਾਬੀ',
  mr: 'मराठी',
  te: 'తెలుగు',
  as: 'অসমীয়া'
};
window.LANGUAGE_MAP = LANGUAGE_MAP;

// Global exports
window.getPatients = getPatients;
window.savePatient = savePatient;
window.showToast = showToast;
window.apiClient = apiClient;
window.toggleSidebar = toggleSidebar;
