document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    
    const uploadSection = document.getElementById('upload-section');
    const previewZone = document.getElementById('preview-zone');
    const loadingZone = document.getElementById('loading-zone');
    const resultsSection = document.getElementById('results-section');
    
    const imagePreview = document.getElementById('image-preview');
    const resetBtn = document.getElementById('reset-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const newScanBtn = document.getElementById('new-scan-btn');
    
    // Result Elements
    const resConfidence = document.getElementById('res-confidence');
    const resName = document.getElementById('res-name');
    const resType = document.getElementById('res-type');
    const resDesc = document.getElementById('res-desc');
    const resCure = document.getElementById('res-cure');
    const mockWarning = document.getElementById('mock-warning');
    
    let currentFile = null;

    // --- Event Listeners ---

    // Browse Button
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // File Input Change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }, false);

    // Reset Button
    resetBtn.addEventListener('click', resetUI);
    newScanBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        resetUI();
    });

    // Analyze Button
    analyzeBtn.addEventListener('click', uploadAndAnalyze);

    // --- Functions ---

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        currentFile = file;
        
        // Setup preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewZone.classList.remove('hidden');
            previewZone.classList.add('fade-in');
        };
        reader.readAsDataURL(file);
    }

    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        previewZone.classList.add('hidden');
        dropZone.classList.remove('hidden');
        loadingZone.classList.add('hidden');
    }

    async function uploadAndAnalyze() {
        if (!currentFile) return;

        // UI Update to loading state
        previewZone.classList.add('hidden');
        loadingZone.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                displayResults(result);
            } else {
                throw new Error(result.error || 'Something went wrong');
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
            resetUI();
        }
    }

    function displayResults(data) {
        // Hide upload section entirely
        uploadSection.classList.add('hidden');
        
        // Show mock warning if applicable
        if (data.mock_mode) {
            mockWarning.classList.remove('hidden');
        } else {
            mockWarning.classList.add('hidden');
        }

        // Update UI elements
        resName.textContent = data.prediction;
        resConfidence.textContent = data.confidence;
        resType.textContent = data.type;
        resDesc.textContent = data.description;
        resCure.textContent = data.cure;

        // Change colors based on healthy vs disease
        if (data.class_id === 'Potato___healthy') {
            resType.style.color = '#10b981';
            resType.style.background = 'rgba(16, 185, 129, 0.2)';
        } else {
            resType.style.color = '#ef4444';
            resType.style.background = 'rgba(239, 68, 68, 0.2)';
        }

        // Show results section
        resultsSection.classList.remove('hidden');
        resultsSection.classList.add('fade-in');
    }
});
