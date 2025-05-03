// General utility functions
document.addEventListener('DOMContentLoaded', function() {
    // Initialize any general JS needed across pages
});

// Face detection and recognition for attendance capture
if (document.getElementById('attendanceCapture')) {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const captureBtn = document.getElementById('captureBtn');
    const resultDiv = document.getElementById('recognitionResult');
    
    let stream = null;
    let faceBoxes = [];
    
    // Start webcam
    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.play();
        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access webcam. Please ensure permissions are granted.");
        }
    }
    
    // Capture frame and send for recognition
    async function captureFrame() {
        if (!stream) return;
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob and send to server
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'capture.jpg');
            
            try {
                const response = await fetch('/attendance-capture', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="error">${data.error}</div>`;
                    return;
                }
                
                // Display recognition results
                displayResults(data);
            } catch (err) {
                console.error("Error:", err);
                resultDiv.innerHTML = '<div class="error">Failed to process image</div>';
            }
        }, 'image/jpeg', 0.9);
    }
    
    // Display recognition results
    function displayResults(data) {
        resultDiv.innerHTML = '';
        
        if (data.recognized_names.length === 0) {
            resultDiv.innerHTML = '<div class="info">No faces recognized</div>';
            return;
        }
        
        const list = document.createElement('ul');
        data.recognized_names.forEach((name, index) => {
            const item = document.createElement('li');
            item.textContent = name === "Unknown" ? "Unknown person" : `Student: ${name}`;
            list.appendChild(item);
        });
        
        resultDiv.appendChild(list);
    }
    
    // Initialize
    startWebcam();
    captureBtn.addEventListener('click', captureFrame);
}

// Face enrollment page
if (document.getElementById('faceEnrollment')) {
    const video = document.getElementById('enrollmentWebcam');
    const captureBtn = document.getElementById('enrollmentCaptureBtn');
    const previewImg = document.getElementById('enrollmentPreview');
    const studentSelect = document.getElementById('studentSelect');
    const submitBtn = document.getElementById('submitEnrollment');
    
    let stream = null;
    
    // Start webcam
    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.play();
        } catch (err) {
            console.error("Error accessing webcam:", err);
            alert("Could not access webcam. Please ensure permissions are granted.");
        }
    }
    
    // Capture frame for enrollment
    function captureFrame() {
        if (!stream) return;
        
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        previewImg.src = canvas.toDataURL('image/jpeg');
        previewImg.style.display = 'block';
        
        // Stop webcam after capture
        stream.getTracks().forEach(track => track.stop());
        video.style.display = 'none';
        captureBtn.style.display = 'none';
        submitBtn.style.display = 'block';
    }
    
    // Initialize
    startWebcam();
    captureBtn.addEventListener('click', captureFrame);
}