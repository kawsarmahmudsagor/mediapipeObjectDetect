import { ObjectDetector, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

const demosSection = document.getElementById("demos");
const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const enableWebcamButton = document.getElementById("webcamButton");

let objectDetector;
let lastVideoTime = -1;
let children = [];

// ---------------- INIT ----------------
async function init() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );

  objectDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "models/v5/logoDetection1.tflite",
      delegate: "GPU",
    },
    scoreThreshold: 0.4,
    runningMode: "VIDEO",
  });

  demosSection.classList.remove("invisible");
}

init();

// ---------------- CAMERA ----------------
enableWebcamButton.addEventListener("click", async () => {
  enableWebcamButton.classList.add("removed");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" } },
    audio: false,
  });

  video.srcObject = stream;
  video.onloadedmetadata = () => {
    video.play();
    requestAnimationFrame(predictWebcam);
  };
});

// ---------------- DETECTION LOOP ----------------
function predictWebcam() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const detections = objectDetector.detectForVideo(
      video,
      performance.now()
    );
    drawResults(detections);
  }

  requestAnimationFrame(predictWebcam);
}

// ---------------- DRAW (FIX MIRROR HERE) ----------------
function drawResults(result) {
  children.forEach(el => el.remove());
  children.length = 0;

  const videoWidth = video.offsetWidth;

  for (const detection of result.detections) {
    const box = detection.boundingBox;

    // 🔥 MIRROR FIX (ONLY X AXIS)
    const x =
      videoWidth - box.originX - box.width;

    const rect = document.createElement("div");
    rect.className = "highlighter";
    rect.style.left = x + "px";
    rect.style.top = box.originY + "px";
    rect.style.width = box.width + "px";
    rect.style.height = box.height + "px";

    const label = document.createElement("p");
    label.className = "info";
    label.innerText =
      `${detection.categories[0].categoryName} ` +
      `${Math.round(detection.categories[0].score * 100)}%`;
    label.style.left = x + "px";
    label.style.top = box.originY + "px";

    liveView.append(rect, label);
    children.push(rect, label);
  }
}
