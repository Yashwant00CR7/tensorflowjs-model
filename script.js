const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const status = document.getElementById('status');
const languageSelector = document.getElementById('languageSelector');

let imageDataList = [];
let selectedLanguage = 'en'; // Default language is English

const messages = {
    en: {
        sendingImages: "Sending images...",
        errorSending: "An error occurred while sending the images.",
        prediction: "The detected currency is",
        withConfidence: "with confidence",
    },
    ta: {
        sendingImages: "படங்களை அனுப்புகிறேன்...",
        errorSending: "படங்களை அனுப்புவதில் பிழை ஏற்பட்டது.",
        prediction: "கண்டறியப்பட்ட நோட்டு",
        withConfidence: "விசுவாசத்துடன் உள்ளது",
    }
};

// Initialize webcam feed
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Could not access webcam. Please allow camera access.');
    }
}

// Multilingual Text-to-Speech function
function speakText(key, additionalText = '') {
    const text = messages[selectedLanguage][key] + (additionalText ? ` ${additionalText}` : '');
    const speech = new SpeechSynthesisUtterance(text);
    speech.lang = selectedLanguage === 'en' ? 'en-US' : 'ta-IN';
    speech.pitch = 1;
    speech.rate = 1;
    speech.volume = 1;
    window.speechSynthesis.speak(speech);
}

// Capture image from webcam and store it
async function captureImage() {
    if (imageDataList.length >= 5) {
        return; // Stop capturing once 5 images are reached
    }

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to Base64 string
    const imageData = canvas.toDataURL('image/jpeg');

    // Store captured image data
    imageDataList.push(imageData);
    console.log('Captured Image:', imageDataList);
}

// Function to capture 5 images with 1-second interval and send the images
async function captureAndSendImages() {
    for (let i = 0; i < 5; i++) {
        await captureImage(); // Capture an image
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for 1 second before capturing the next
    }

    console.log('Captured all 5 images');
    sendImages(); // Send images after capturing all
}

// Send images to API automatically after all 5 images are captured
async function sendImages() {
    if (imageDataList.length === 0) {
        alert('No images to send!');
        return;
    }

    status.innerText = messages[selectedLanguage].sendingImages;
    speakText('sendingImages');

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ images: imageDataList }),
        });

        const result = await response.json();
        console.log(result);

        const predictedClass = result.class || 'Unknown';
        const scorePercentage = result.score_percentage || 'N/A';

        status.innerText = `${messages[selectedLanguage].prediction}: ${predictedClass} ${messages[selectedLanguage].withConfidence} ${scorePercentage}`;
        speakText('prediction', `${predictedClass} ${messages[selectedLanguage].withConfidence} ${scorePercentage}`);

        // Reset image data list after sending
        imageDataList = [];
    } catch (error) {
        console.error('Error sending images:', error);
        status.innerText = messages[selectedLanguage].errorSending;
        speakText('errorSending');
    }
}

// Event listener for language selector dropdown
languageSelector.addEventListener('change', (event) => {
    selectedLanguage = event.target.value;
});

// Add event listener for capture button
captureBtn.addEventListener('click', captureAndSendImages);

// Start the webcam on page load
startWebcam();
