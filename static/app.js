// ==========================================================================
// AUDEMO WEB APP CLIENT LOGIC
// ==========================================================================

// Emotion Color System Configuration
const EMOTION_COLORS = {
    happy: { name: 'Happy', hex: '#ffde4d', rgb: [255, 222, 77] },
    sad: { name: 'Sadness', hex: '#4facfe', rgb: [79, 172, 254] },
    angry: { name: 'Angry', hex: '#ff4d4d', rgb: [255, 77, 77] },
    fear: { name: 'Fear', hex: '#b19ffb', rgb: [177, 159, 251] },
    disgust: { name: 'Disgust', hex: '#2ecc71', rgb: [46, 204, 113] },
    neutral: { name: 'Neutral', hex: '#bdc3c7', rgb: [189, 195, 199] },
    ps: { name: 'Surprise', hex: '#ff9f43', rgb: [255, 159, 67] }, // Pleasant Surprise
    surprise: { name: 'Surprise', hex: '#ff9f43', rgb: [255, 159, 67] }
};

// Application States
const AppState = {
    IDLE: 'IDLE',
    FILE_SELECTED: 'FILE_SELECTED',
    ANALYZING: 'ANALYZING',
    RESULT: 'RESULT',
    ERROR: 'ERROR'
};

// State Variables
let currentState = AppState.IDLE;
let selectedFile = null;
let analysisResult = null;
let canvasAnimationId = null;

// DOM Elements Cache
const DOM = {
    heroSection: document.getElementById('heroSection'),
    uploadSection: document.getElementById('uploadSection'),
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    fileDetailsPane: document.getElementById('fileDetailsPane'),
    fileName: document.getElementById('fileName'),
    fileSize: document.getElementById('fileSize'),
    fileDuration: document.getElementById('fileDuration'),
    fileFormat: document.getElementById('fileFormat'),
    clearFileBtn: document.getElementById('clearFileBtn'),
    actionBtnContainer: document.getElementById('actionBtnContainer'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    
    processingSection: document.getElementById('processingSection'),
    stepLoad: document.getElementById('step-load'),
    stepChunk: document.getElementById('step-chunk'),
    stepCnn: document.getElementById('step-cnn'),
    stepGru: document.getElementById('step-gru'),
    stepTimeline: document.getElementById('step-timeline'),
    
    errorSection: document.getElementById('errorSection'),
    errorMessage: document.getElementById('errorMessage'),
    errorRetryBtn: document.getElementById('errorRetryBtn'),
    
    resultsSection: document.getElementById('resultsSection'),
    predictionHeroCard: document.getElementById('predictionHeroCard'),
    predictedEmotion: document.getElementById('predictedEmotion'),
    overallConfidence: document.getElementById('overallConfidence'),
    legendContainer: document.getElementById('legendContainer'),
    
    // Canvas & Tooltip
    timelineCanvas: document.getElementById('timelineCanvas'),
    timelineTooltip: document.getElementById('timelineTooltip'),
    tooltipTime: document.getElementById('tooltipTime'),
    tooltipDot: document.getElementById('tooltipDot'),
    tooltipEmotion: document.getElementById('tooltipEmotion'),
    tooltipConf: document.getElementById('tooltipConf'),
    timelineTimeLabels: document.getElementById('timelineTimeLabels'),
    
    // Audio Player
    mainAudioElement: document.getElementById('mainAudioElement'),
    playPauseBtn: document.getElementById('playPauseBtn'),
    playIcon: document.getElementById('playIcon'),
    pauseIcon: document.getElementById('pauseIcon'),
    timeCurrent: document.getElementById('timeCurrent'),
    timeTotal: document.getElementById('timeTotal'),
    playerProgressBarContainer: document.getElementById('playerProgressBarContainer'),
    playerProgressFill: document.getElementById('playerProgressFill'),
    playerProgressHandle: document.getElementById('playerProgressHandle'),
    playbackSpeedSelect: document.getElementById('playbackSpeedSelect'),
    volumeMuteBtn: document.getElementById('volumeMuteBtn'),
    volumeIcon: document.getElementById('volumeIcon'),
    muteIcon: document.getElementById('muteIcon'),
    volumeSlider: document.getElementById('volumeSlider'),
    
    // Visualizations
    distributionBars: document.getElementById('distributionBars'),
    journeyFlow: document.getElementById('journeyFlow'),
    
    // Technical Details
    techChunkSize: document.getElementById('techChunkSize'),
    techOverlap: document.getElementById('techOverlap'),
    techDuration: document.getElementById('techDuration'),
    techTotalChunks: document.getElementById('techTotalChunks'),
    techSamplingRate: document.getElementById('techSamplingRate'),
    techFeatureDim: document.getElementById('techFeatureDim'),
    techPredictionTime: document.getElementById('techPredictionTime'),
    
    // Reset
    resetBtn: document.getElementById('resetBtn')
};

// ==========================================================================
// STATE MANAGEMENT & WORKFLOWS
// ==========================================================================

function setAppState(state) {
    currentState = state;
    console.log(`State Transition: ${state}`);
    
    // Reset visibilities
    DOM.heroSection.classList.add('hidden');
    DOM.uploadSection.classList.add('hidden');
    DOM.processingSection.classList.add('hidden');
    DOM.errorSection.classList.add('hidden');
    DOM.resultsSection.classList.add('hidden');
    
    // Cancel any timeline animation frame
    if (canvasAnimationId) {
        cancelAnimationFrame(canvasAnimationId);
        canvasAnimationId = null;
    }
    
    switch (state) {
        case AppState.IDLE:
            DOM.heroSection.classList.remove('hidden');
            DOM.uploadSection.classList.remove('hidden');
            DOM.fileDetailsPane.classList.add('hidden');
            DOM.actionBtnContainer.classList.add('hidden');
            DOM.fileInput.value = '';
            selectedFile = null;
            break;
            
        case AppState.FILE_SELECTED:
            DOM.heroSection.classList.remove('hidden');
            DOM.uploadSection.classList.remove('hidden');
            DOM.fileDetailsPane.classList.remove('hidden');
            DOM.actionBtnContainer.classList.remove('hidden');
            break;
            
        case AppState.ANALYZING:
            DOM.processingSection.classList.remove('hidden');
            resetChecklist();
            break;
            
        case AppState.RESULT:
            DOM.resultsSection.classList.remove('hidden');
            break;
            
        case AppState.ERROR:
            DOM.errorSection.classList.remove('hidden');
            break;
    }
}

// Format duration helper
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function handleFileSelected(file) {
    if (!file) return;
    
    selectedFile = file;
    DOM.fileName.textContent = file.name;
    
    // Format size
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    DOM.fileSize.textContent = `${sizeMB} MB`;
    
    // Detect format
    const extension = file.name.split('.').pop().toUpperCase();
    DOM.fileFormat.textContent = `${extension} Audio`;
    
    // Get local duration using Audio metadata loading
    const objectURL = URL.createObjectURL(file);
    const tempAudio = new Audio();
    tempAudio.src = objectURL;
    tempAudio.addEventListener('loadedmetadata', () => {
        DOM.fileDuration.textContent = formatTime(tempAudio.duration);
        DOM.fileDetailsPane.classList.remove('hidden');
        URL.revokeObjectURL(objectURL);
    });
    tempAudio.addEventListener('error', () => {
        DOM.fileDuration.textContent = 'Unknown duration';
        URL.revokeObjectURL(objectURL);
    });
    
    setAppState(AppState.FILE_SELECTED);
}

// Processing checklist helper
function resetChecklist() {
    const items = [DOM.stepLoad, DOM.stepChunk, DOM.stepCnn, DOM.stepGru, DOM.stepTimeline];
    items.forEach(item => {
        item.className = 'checklist-item';
        item.querySelector('.check-icon').textContent = '○';
    });
}

function updateChecklistStep(stepId, state) {
    const el = document.getElementById(stepId);
    if (!el) return;
    
    el.className = `checklist-item ${state}`;
    const checkIcon = el.querySelector('.check-icon');
    if (state === 'done') {
        checkIcon.textContent = '✓';
    } else if (state === 'active') {
        checkIcon.textContent = '▶';
    } else {
        checkIcon.textContent = '○';
    }
}

// Simulate pipeline steps progression during server fetch
function runSimulatedChecklist(onComplete) {
    let currentStep = 0;
    const steps = [
        { id: 'step-load', delay: 100 },
        { id: 'step-chunk', delay: 400 },
        { id: 'step-cnn', delay: 1000 },
        { id: 'step-gru', delay: 1800 },
        { id: 'step-timeline', delay: 2400 }
    ];
    
    resetChecklist();
    
    function runNext() {
        if (currentStep >= steps.length) {
            if (onComplete) onComplete();
            return;
        }
        
        const step = steps[currentStep];
        updateChecklistStep(step.id, 'active');
        
        setTimeout(() => {
            updateChecklistStep(step.id, 'done');
            currentStep++;
            runNext();
        }, step.delay);
    }
    
    runNext();
}

// Post audio for analysis
function analyzeAudioFile() {
    if (!selectedFile) return;
    
    setAppState(AppState.ANALYZING);
    updateChecklistStep('step-load', 'active');
    
    const formData = new FormData();
    formData.append('audio', selectedFile);
    
    let apiCompleted = false;
    let apiData = null;
    let apiError = null;
    
    // Start request
    fetch('/api/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error || 'Server error'); });
        }
        return response.json();
    })
    .then(data => {
        apiData = data;
        apiCompleted = true;
    })
    .catch(err => {
        apiError = err.message;
        apiCompleted = true;
    });
    
    // Manage UI Checklist in sync with network request
    let currentStep = 0;
    const checklistSteps = ['step-load', 'step-chunk', 'step-cnn', 'step-gru', 'step-timeline'];
    
    function tickProgress() {
        if (apiCompleted) {
            // API is done
            if (apiError) {
                DOM.errorMessage.textContent = apiError;
                setAppState(AppState.ERROR);
            } else {
                // Instantly complete all remaining steps and transition to results
                checklistSteps.forEach(stepId => updateChecklistStep(stepId, 'done'));
                setTimeout(() => {
                    displayResults(apiData);
                }, 400);
            }
            return;
        }
        
        // Progress through steps based on approximate times while waiting for API
        if (currentStep < checklistSteps.length - 1) {
            updateChecklistStep(checklistSteps[currentStep], 'done');
            currentStep++;
            updateChecklistStep(checklistSteps[currentStep], 'active');
        }
        
        // Next tick
        setTimeout(tickProgress, 700);
    }
    
    setTimeout(tickProgress, 400);
}

// ==========================================================================
// RESULTS RENDERING & VISUALIZATION
// ==========================================================================

function displayResults(data) {
    analysisResult = data;
    setAppState(AppState.RESULT);
    
    // 1. Overall emotion hero card
    const emotionKey = data.overall_emotion.toLowerCase();
    const emotionConfig = EMOTION_COLORS[emotionKey] || EMOTION_COLORS.neutral;
    
    DOM.predictedEmotion.textContent = emotionConfig.name.toUpperCase();
    DOM.overallConfidence.textContent = `${(data.overall_confidence * 100).toFixed(1)}%`;
    
    // Apply styling to prediction card
    DOM.predictionHeroCard.style.backgroundColor = emotionConfig.hex;
    
    // 2. Render Legends
    renderLegend(data.segments);
    
    // 3. Audio Player Initialization
    DOM.mainAudioElement.src = data.audio_url;
    DOM.mainAudioElement.load();
    DOM.timeTotal.textContent = formatTime(data.duration);
    DOM.playerProgressFill.style.width = '0%';
    DOM.playerProgressHandle.style.left = '0%';
    
    // Reset speed
    DOM.playbackSpeedSelect.value = "1.0";
    DOM.mainAudioElement.playbackRate = 1.0;
    
    // Reset play/pause buttons
    DOM.playIcon.classList.remove('hidden');
    DOM.pauseIcon.classList.add('hidden');
    
    // Set time labels on timeline
    DOM.timelineTimeLabels.querySelector('.mid-time-label').textContent = formatTime(data.duration / 2);
    DOM.timelineTimeLabels.querySelector('.end-time-label').textContent = formatTime(data.duration);
    
    // 4. Render Distribution chart
    renderDistribution(data.segments);
    
    // 5. Render Journey flow
    renderJourneyFlow(data.segments);
    
    // 6. Technical details
    DOM.techChunkSize.textContent = `${data.chunk_duration} seconds`;
    DOM.techOverlap.textContent = `${data.overlap} seconds`;
    DOM.techDuration.textContent = `${data.duration.toFixed(2)} seconds`;
    DOM.techTotalChunks.textContent = data.technical_details.total_chunks;
    DOM.techSamplingRate.textContent = `${data.technical_details.sampling_rate_hz.toLocaleString()} Hz`;
    DOM.techFeatureDim.textContent = `${data.technical_details.feature_dim} (Dense embedding layer)`;
    DOM.techPredictionTime.textContent = `${data.technical_details.prediction_time_seconds}s`;
    
    // Collapse details section initially
    DOM.techDetailsSection.removeAttribute('open');
    
    // 7. Timeline drawing loop
    drawTimelineCanvas();
}

function renderLegend(segments) {
    DOM.legendContainer.innerHTML = '';
    
    // Extract unique emotions present in the segments
    const uniqueEmotions = [...new Set(segments.map(s => s.emotion.toLowerCase()))];
    
    uniqueEmotions.forEach(emotion => {
        const config = EMOTION_COLORS[emotion] || EMOTION_COLORS.neutral;
        
        const item = document.createElement('div');
        item.className = 'legend-item';
        
        const colorSquare = document.createElement('span');
        colorSquare.className = 'legend-color';
        colorSquare.style.backgroundColor = config.hex;
        
        const label = document.createElement('span');
        label.textContent = config.name;
        
        item.appendChild(colorSquare);
        item.appendChild(label);
        DOM.legendContainer.appendChild(item);
    });
}

function renderDistribution(segments) {
    DOM.distributionBars.innerHTML = '';
    
    // Count segments
    const counts = {};
    const total = segments.length;
    
    segments.forEach(s => {
        const emo = s.emotion.toLowerCase();
        counts[emo] = (counts[emo] || 0) + 1;
    });
    
    // Sort emotions by count desc
    const sortedEmotions = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
    
    sortedEmotions.forEach(emo => {
        const count = counts[emo];
        const pct = Math.round((count / total) * 100);
        const config = EMOTION_COLORS[emo] || EMOTION_COLORS.neutral;
        
        const row = document.createElement('div');
        row.className = 'chart-bar-row';
        
        const label = document.createElement('div');
        label.className = 'bar-label';
        label.textContent = config.name;
        
        const track = document.createElement('div');
        track.className = 'bar-track';
        
        const fill = document.createElement('div');
        fill.className = 'bar-fill';
        fill.style.backgroundColor = config.hex;
        // Animate fill width slightly later for aesthetic transition
        setTimeout(() => {
            fill.style.width = `${pct}%`;
        }, 100);
        
        const percentage = document.createElement('div');
        percentage.className = 'bar-percentage';
        percentage.textContent = `${pct}%`;
        
        track.appendChild(fill);
        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(percentage);
        DOM.distributionBars.appendChild(row);
    });
}

function renderJourneyFlow(segments) {
    DOM.journeyFlow.innerHTML = '';
    
    // Compress consecutive segments with same emotion to show journey changes
    const journey = [];
    let current = null;
    
    segments.forEach(s => {
        if (!current || current.emotion !== s.emotion) {
            current = {
                start: s.start,
                emotion: s.emotion,
                confidence: s.confidence
            };
            journey.push(current);
        }
    });
    
    journey.forEach(step => {
        const config = EMOTION_COLORS[step.emotion.toLowerCase()] || EMOTION_COLORS.neutral;
        
        const item = document.createElement('div');
        item.className = 'journey-step';
        
        const timeLabel = document.createElement('span');
        timeLabel.className = 'journey-step-time';
        timeLabel.textContent = formatTime(step.start);
        
        const indicator = document.createElement('span');
        indicator.className = 'journey-step-indicator';
        indicator.style.backgroundColor = config.hex;
        
        const emotionLabel = document.createElement('span');
        emotionLabel.className = 'journey-step-emotion';
        emotionLabel.textContent = config.name.toUpperCase();
        
        item.appendChild(timeLabel);
        item.appendChild(indicator);
        item.appendChild(emotionLabel);
        DOM.journeyFlow.appendChild(item);
    });
}

// Draw timeline canvas including spectrogram and waveform
function drawTimelineCanvas() {
    if (!analysisResult) return;
    
    const canvas = DOM.timelineCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    const segments = analysisResult.segments;
    const numSegments = segments.length;
    const duration = analysisResult.duration;
    const waveform = analysisResult.waveform;
    
    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);
    
    // 1. Draw Spectrogram slices
    const segmentWidth = width / numSegments;
    
    for (let i = 0; i < numSegments; i++) {
        const seg = segments[i];
        const emoKey = seg.emotion.toLowerCase();
        const colorRGB = EMOTION_COLORS[emoKey] ? EMOTION_COLORS[emoKey].rgb : EMOTION_COLORS.neutral.rgb;
        
        const spectrogramData = seg.spectrogram; // 32x32 array (frequencies rows, time cols)
        const cellW = segmentWidth / 32;
        const cellH = height / 32;
        const startX = i * segmentWidth;
        
        // Render 32x32 cells
        for (let col = 0; col < 32; col++) {
            for (let row = 0; row < 32; row++) {
                // Spectrogram array is stored as 32 frequency rows (high indices at bottom or top depending on librosa conversion).
                // Standard spectrogram: row 0 is lowest frequency (drawn at bottom).
                // Let's invert Y axis: index 31 represents highest frequency (drawn at top).
                const val = spectrogramData[31 - row][col];
                
                // Color intensity maps to opacity
                ctx.fillStyle = `rgba(${colorRGB[0]}, ${colorRGB[1]}, ${colorRGB[2]}, ${val * 0.85})`;
                ctx.fillRect(
                    startX + col * cellW,
                    row * cellH,
                    cellW + 0.5, // add subpixel padding to avoid grid line gaps
                    cellH + 0.5
                );
            }
        }
    }
    
    // 2. Draw Waveform envelope outline on top
    if (waveform && waveform.length > 0) {
        ctx.beginPath();
        const centerY = height / 2;
        const stepX = width / waveform.length;
        
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = '#000000'; // thick black outline
        
        // Upper wave line
        ctx.moveTo(0, centerY);
        for (let i = 0; i < waveform.length; i++) {
            const amp = waveform[i] * (height * 0.4); // max amp is 40% of height
            ctx.lineTo(i * stepX, centerY - amp);
        }
        
        // Lower wave line (right to left for solid path)
        for (let i = waveform.length - 1; i >= 0; i--) {
            const amp = waveform[i] * (height * 0.4);
            ctx.lineTo(i * stepX, centerY + amp);
        }
        ctx.closePath();
        
        // Fill waveform with translucent background for readability
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.fill();
        ctx.stroke();
    }
    
    // 3. Draw segment division lines (dashed lines between chunks)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    for (let i = 1; i < numSegments; i++) {
        const x = i * segmentWidth;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    ctx.setLineDash([]); // Reset
    
    // 4. Draw Audio Playhead
    const audio = DOM.mainAudioElement;
    if (audio && audio.duration) {
        const playPercent = audio.currentTime / audio.duration;
        const playheadX = playPercent * width;
        
        // Draw playhead vertical line
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, height);
        ctx.stroke();
        
        // Draw playhead handle at top
        ctx.fillStyle = '#ffde4d'; // yellow
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(playheadX, 0, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
    
    // Request next frame if audio is playing to update playhead smoothly
    if (audio && !audio.paused) {
        canvasAnimationId = requestAnimationFrame(drawTimelineCanvas);
    }
}

// Handle timeline mouse interactions
function getSegmentAtMouse(e) {
    if (!analysisResult) return null;
    
    const canvas = DOM.timelineCanvas;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    
    // Calculate relative x percentage
    const pct = Math.max(0, Math.min(1, x / rect.width));
    const totalTime = analysisResult.duration;
    const hoverTime = pct * totalTime;
    
    // Find matching segment
    const segments = analysisResult.segments;
    const numSegments = segments.length;
    const segmentWidth = rect.width / numSegments;
    const segmentIndex = Math.floor(x / segmentWidth);
    
    if (segmentIndex >= 0 && segmentIndex < numSegments) {
        return {
            segment: segments[segmentIndex],
            x: x,
            hoverTime: hoverTime
        };
    }
    return null;
}

function handleTimelineMouseMove(e) {
    const result = getSegmentAtMouse(e);
    if (!result) {
        DOM.timelineTooltip.classList.add('hidden');
        return;
    }
    
    const { segment, x } = result;
    const config = EMOTION_COLORS[segment.emotion.toLowerCase()] || EMOTION_COLORS.neutral;
    
    // Update tooltip content
    DOM.tooltipTime.textContent = `${formatTime(segment.start)} – ${formatTime(segment.end)}`;
    DOM.tooltipDot.style.backgroundColor = config.hex;
    DOM.tooltipEmotion.textContent = config.name.toUpperCase();
    DOM.tooltipConf.textContent = `${(segment.confidence * 100).toFixed(0)}% Confidence`;
    
    // Position tooltip
    DOM.timelineTooltip.style.left = `${x}px`;
    DOM.timelineTooltip.classList.remove('hidden');
}

function handleTimelineMouseLeave() {
    DOM.timelineTooltip.classList.add('hidden');
}

function handleTimelineClick(e) {
    const result = getSegmentAtMouse(e);
    if (!result) return;
    
    const { segment } = result;
    
    // Jump audio to segment start time
    DOM.mainAudioElement.currentTime = segment.start;
    updatePlayerProgress();
    drawTimelineCanvas();
}

// ==========================================================================
// CUSTOM AUDIO PLAYER CONTROLS
// ==========================================================================

function togglePlayPause() {
    const audio = DOM.mainAudioElement;
    if (audio.paused) {
        audio.play().then(() => {
            DOM.playIcon.classList.add('hidden');
            DOM.pauseIcon.classList.remove('hidden');
            drawTimelineCanvas(); // Start loop
        });
    } else {
        audio.pause();
        DOM.playIcon.classList.remove('hidden');
        DOM.pauseIcon.classList.add('hidden');
        if (canvasAnimationId) {
            cancelAnimationFrame(canvasAnimationId);
            canvasAnimationId = null;
        }
        drawTimelineCanvas(); // Redraw static playhead
    }
}

function updatePlayerProgress() {
    const audio = DOM.mainAudioElement;
    if (!audio || !audio.duration) return;
    
    const current = audio.currentTime;
    const duration = audio.duration;
    
    // Update digital readout
    DOM.timeCurrent.textContent = formatTime(current);
    
    // Update progress bar fill and handle position
    const pct = (current / duration) * 100;
    DOM.playerProgressFill.style.width = `${pct}%`;
    DOM.playerProgressHandle.style.left = `${pct}%`;
}

function handlePlayerProgressClick(e) {
    const audio = DOM.mainAudioElement;
    if (!audio || !audio.duration) return;
    
    const rect = DOM.playerProgressBarContainer.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, x / rect.width));
    
    audio.currentTime = pct * audio.duration;
    updatePlayerProgress();
    drawTimelineCanvas();
}

function handleSpeedChange() {
    const speed = parseFloat(DOM.playbackSpeedSelect.value);
    DOM.mainAudioElement.playbackRate = speed;
}

function handleVolumeChange() {
    const vol = parseFloat(DOM.volumeSlider.value);
    DOM.mainAudioElement.volume = vol;
    DOM.mainAudioElement.muted = (vol === 0);
    updateVolumeIcon(vol, DOM.mainAudioElement.muted);
}

function toggleMute() {
    const audio = DOM.mainAudioElement;
    audio.muted = !audio.muted;
    
    if (audio.muted) {
        DOM.volumeIcon.classList.add('hidden');
        DOM.muteIcon.classList.remove('hidden');
        DOM.volumeSlider.value = 0;
    } else {
        DOM.volumeIcon.classList.remove('hidden');
        DOM.muteIcon.classList.add('hidden');
        DOM.volumeSlider.value = audio.volume;
    }
}

function updateVolumeIcon(vol, isMuted) {
    if (isMuted || vol === 0) {
        DOM.volumeIcon.classList.add('hidden');
        DOM.muteIcon.classList.remove('hidden');
    } else {
        DOM.volumeIcon.classList.remove('hidden');
        DOM.muteIcon.classList.add('hidden');
    }
}

// ==========================================================================
// DRAG & DROP & EVENT LISTENERS SETUP
// ==========================================================================

function setupEventListeners() {
    // 1. Drag and Drop events
    const zone = DOM.uploadZone;
    
    ['dragenter', 'dragover'].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
        }, false);
    });
    
    zone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelected(files[0]);
        }
    }, false);
    
    zone.addEventListener('click', () => {
        DOM.fileInput.click();
    });
    
    DOM.fileInput.addEventListener('change', () => {
        if (DOM.fileInput.files.length > 0) {
            handleFileSelected(DOM.fileInput.files[0]);
        }
    });
    
    // Clear / Reset selections
    DOM.clearFileBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent opening file picker
        setAppState(AppState.IDLE);
    });
    
    DOM.analyzeBtn.addEventListener('click', analyzeAudioFile);
    DOM.errorRetryBtn.addEventListener('click', () => setAppState(AppState.IDLE));
    DOM.resetBtn.addEventListener('click', () => setAppState(AppState.IDLE));
    
    // 2. Custom Audio Player Events
    DOM.playPauseBtn.addEventListener('click', togglePlayPause);
    
    DOM.mainAudioElement.addEventListener('timeupdate', () => {
        updatePlayerProgress();
        // Redraw canvas if paused so playhead moves correctly during manual seeks.
        if (DOM.mainAudioElement.paused) {
            drawTimelineCanvas();
        }
    });
    
    DOM.mainAudioElement.addEventListener('ended', () => {
        DOM.playIcon.classList.remove('hidden');
        DOM.pauseIcon.classList.add('hidden');
        drawTimelineCanvas();
    });
    
    DOM.playerProgressBarContainer.addEventListener('mousedown', (e) => {
        handlePlayerProgressClick(e);
        
        function onMouseMove(moveEvent) {
            handlePlayerProgressClick(moveEvent);
        }
        
        function onMouseUp() {
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
        }
        
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
    });
    
    DOM.playbackSpeedSelect.addEventListener('change', handleSpeedChange);
    DOM.volumeSlider.addEventListener('input', handleVolumeChange);
    DOM.volumeMuteBtn.addEventListener('click', toggleMute);
    
    // 3. Canvas Mouse Events
    DOM.timelineCanvas.addEventListener('mousemove', handleTimelineMouseMove);
    DOM.timelineCanvas.addEventListener('mouseleave', handleTimelineMouseLeave);
    DOM.timelineCanvas.addEventListener('click', handleTimelineClick);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setAppState(AppState.IDLE);
});
