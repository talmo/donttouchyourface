/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 *
 * This was based off of the demos at:
 * https://github.com/tensorflow/tfjs-models/tree/master/posenet/demos
 *
 * Author: Talmo Pereira <talmo at princeton dot edu>
 */

let videoWidth = 640;
let videoHeight = videoWidth * (1 / 1.777);
let color = 'aqua';

function isAndroid() {
    return /Android/i.test(navigator.userAgent);
}

function isiOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
    return isAndroid() || isiOS();
}

async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
    }
    const mobile = isMobile();
    if (mobile) {
        videoWidth = 320;
        videoHeight = videoWidth * 1.777;
    }
    const video = document.getElementById('video');
    video.width = videoWidth;
    video.height = videoHeight;
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: mobile ? undefined : videoWidth,
            height: mobile ? undefined : videoHeight,
        },
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();
    return video;
}

function toTuple({
    y,
    x
}) {
    return [y, x];
}

function drawPoint(ctx, y, x, r, color) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
    for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i];
        if (keypoint.score < minConfidence) {
            continue;
        }
        const {
            y,
            x
        } = keypoint.position;
        drawPoint(ctx, y * scale, x * scale, 3, color);
    }
}

async function renderToCanvas(a, ctx) {
    const [height, width] = a.shape;
    const imageData = new ImageData(width, height);
    const data = await a.data();
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        const k = i * 3;
        imageData.data[j + 0] = data[k + 0];
        imageData.data[j + 1] = data[k + 1];
        imageData.data[j + 2] = data[k + 2];
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

function renderImageToCanvas(image, size, canvas) {
    canvas.width = size[0];
    canvas.height = size[1];
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
}

function computeDist(keypoint1, keypoint2) {
    return Math.pow(
        Math.pow(keypoint1["position"]["x"] - keypoint2["position"]["x"], 2) +
        Math.pow(keypoint1["position"]["y"] - keypoint2["position"]["y"], 2), 0.5);
}

function detectPoseInRealTime(video, net) {
    const canvas = document.getElementById('output');
    const ctx = canvas.getContext('2d');
    const flipPoseHorizontal = true;
    canvas.width = videoWidth;
    canvas.height = videoHeight;

    async function poseDetectionFrame() {
        let poses = [];
        let minPoseConfidence = 0.1;
        let minPartConfidence = 0.1;
        const pose = await net.estimatePoses(video, {
            flipHorizontal: flipPoseHorizontal,
            decodingMethod: 'single-person'
        });
        poses = poses.concat(pose);

        // Draw video.
        ctx.clearRect(0, 0, videoWidth, videoHeight);
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-videoWidth, 0);
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
        ctx.restore();

        let info_div = document.getElementById('info');
        info_div.innerHTML = '';
        info_div.style.display = 'block';
        let dangerzone_div = document.getElementById('danger-zone');
        poses.forEach(({
            score,
            keypoints
        }) => {
            if (score >= minPoseConfidence) {
                // 0 nose
                // 1 leftEye
                // 2 rightEye
                // 3 leftEar
                // 4 rightEar
                // 9 leftWrist
                // 10 rightWrist
                let min_dist = 10000;
                let touch_threshold = 175;
                let nose = keypoints[0];
                let leftEye = keypoints[1];
                let rightEye = keypoints[2];
                if (nose.score > minPartConfidence) {
                    if (leftEye.score > minPartConfidence) {
                        touch_threshold = computeDist(nose, leftEye) * 4;
                    } else if (rightEye.score > minPartConfidence) {
                        touch_threshold = computeDist(nose, rightEye) * 4;
                    }
                }
                info_div.textContent = '';
                info_div.innerHTML += "Touch threshold: " + touch_threshold + "\n<br>";
                for (const wrist of[keypoints[9], keypoints[10]]) {
                    for (const face of[keypoints[0], keypoints[1], keypoints[2]]) {
                        if (wrist.score > minPartConfidence &&
                        	face.score > minPartConfidence) {
                            let dist = computeDist(wrist, face);
                            info_div.innerHTML += "Distance  (" + wrist["part"];
                            info_div.innerHTML += " <-> " + face["part"] + "): " + dist;
                            info_div.innerHTML += "\n<br>";
                            min_dist = Math.min(min_dist, dist);
                        }
                    }
                }
                if (min_dist < touch_threshold) {
                    canvas.style.border = "5px solid red";
                    color = "red";
                    dangerzone_div.style.display = "table";
                } else {
                    canvas.style.border = "3px solid green";
                    color = "aqua";
                    dangerzone_div.style.display = "none";
                }
                drawKeypoints(keypoints, minPartConfidence, ctx);
            }
        });
        requestAnimationFrame(poseDetectionFrame);
    }
    poseDetectionFrame();
}

async function bindPage() {

    let net;
    if (isMobile()) {
        net = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: {
                width: 640,
                height: 480
            },
            multiplier: 0.75
        });
    } else {
        net = await posenet.load({
            architecture: 'ResNet50',
            outputStride: 32,
            inputResolution: {
                width: 257,
                height: 200
            },
            quantBytes: 2
        });
    }
    let video;
    try {
        video = await loadVideo();
    } catch (e) {
        let info = document.getElementById('info');
        info.textContent = 'this browser does not support video capture,' +
            'or this device does not have a camera';
        info.style.display = 'block';
        throw e;
    }
    detectPoseInRealTime(video, net);
}

bindPage();
