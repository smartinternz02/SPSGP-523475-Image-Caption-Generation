import * as tf from '@tensorflow/tfjs';

export async function loadGraphModel(modelPath) {
  const model = await tf.loadGraphModel(modelPath);
  return model;
}

export async function imageToTensor(image) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = async (event) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 224;
        canvas.height = 224;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        const tensor = tf.tensor3d(imageData, [224, 224, 4], 'int32')
          .slice([0, 0, 0], [224, 224, 3])
          .toFloat()
          .expandDims();

        resolve(tensor);
      };
      img.onerror = reject;
      img.src = event.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(image);
  });
}
