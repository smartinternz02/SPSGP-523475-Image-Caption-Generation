import * as tf from '@tensorflow/tfjs';
import { loadGraphModel, imageToTensor } from './utils.js';

async function runCaptionGenerator() {
  const model = await loadGraphModel('model/model.json');

  const form = document.getElementById('upload-form');
  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const input = document.getElementById('image-input');
    if (input.files.length > 0) {
      const image = input.files[0];
      const tensor = await imageToTensor(image);
      const prediction = await model.predict(tensor);
      const caption = Array.from(prediction.dataSync())
        .map((score, index) => ({ score, index }))
        .sort((a, b) => b.score - a.score)
        .map(({ index }) => tokenizer[index])
        .join(' ');

      const resultDiv = document.getElementById('result');
      resultDiv.textContent = caption;
    }
  });
}

runCaptionGenerator();
